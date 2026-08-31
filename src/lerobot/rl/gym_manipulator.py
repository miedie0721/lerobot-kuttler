# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.model import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    RobotObservation,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
    spacemouse,  # noqa: F401
)
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.import_utils import require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False
    # replay 模式：
    #   "action": 把记录的 ACTION（末端增量，与 policy 输出同空间）喂回动作管线
    #             （干预覆盖 → IK → 关节指令），模拟"policy 输出动作的执行情况"。
    #   "joint":  按记录的实际关节角（observation.state）回放，精确复现录制画面。
    replay_mode: str = "action"  # "action" 或 "joint"


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def reset_follower_position(
    robot_arm: Robot,
    target_position: np.ndarray,
    lift_offset_z: float = 0.0,
    kinematics: RobotKinematics | None = None,
    frozen_joints: list[str] | None = None,
) -> None:
    """Reset robot arm to target position using smooth trajectories.

    If ``lift_offset_z`` > 0 and a kinematics solver is provided, the end-effector
    is first lifted by ``lift_offset_z`` meters along the world Z axis (vertical),
    and only then interpolated back to the target joint position. This is useful for
    peg-in-hole tasks where an inserted object must be pulled out vertically before
    the arm moves sideways.

    ``frozen_joints`` are joint names kept at their current value during the lift
    waypoint (e.g. ``["wrist_flex"]``), so the wrist angle is not disturbed while
    pulling the object out.
    """
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )

    # 复位路径：当前位姿 →（可选）垂直抬升中间位姿 → 目标位姿
    waypoints = [current_position]
    if lift_offset_z > 0.0 and kinematics is not None:
        # placo 运动学求解需要 float64 输入
        current_f64 = current_position.astype(np.float64)
        t_curr = kinematics.forward_kinematics(current_f64)
        t_lift = t_curr.copy()
        t_lift[:3, 3] += np.array([0.0, 0.0, lift_offset_z])
        lift_target = np.asarray(kinematics.inverse_kinematics(current_f64, t_lift), dtype=np.float32)
        # 抬升阶段保持指定关节不变（索引顺序与 current_position_dict 一致）
        frozen = frozen_joints or []
        for i, name in enumerate(current_position_dict):
            if name in frozen:
                lift_target[i] = current_position[i]
        waypoints.append(lift_target)
    waypoints.append(target_position)

    for start, end in zip(waypoints, waypoints[1:], strict=False):
        trajectory = torch.from_numpy(np.linspace(start, end, 50))
        for pose in trajectory:
            action_dict = dict(zip(current_position_dict, pose, strict=False))
            robot_arm.bus.sync_write("Goal_Position", action_dict)
            precise_sleep(0.015)


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        reset_lift_offset_z: float = 0.0,
        reset_kinematics: RobotKinematics | None = None,
        reset_lift_frozen_joints: list[str] | None = None,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
            reset_lift_offset_z: Lift the end-effector vertically by this many meters
                before resetting to ``reset_pose`` (see ``ResetConfig``).
            reset_kinematics: Kinematics solver used to compute the lift waypoint.
            reset_lift_frozen_joints: Joint names kept fixed during the lift waypoint.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras
        self.reset_lift_offset_z = reset_lift_offset_z
        self.reset_kinematics = reset_kinematics
        self.reset_lift_frozen_joints = reset_lift_frozen_joints

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> RobotObservation:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        # Reset the robot
        # self.robot.reset()
        start_time = time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(
                self.robot,
                np.array(self.reset_pose),
                lift_offset_z=self.reset_lift_offset_z,
                kinematics=self.reset_kinematics,
                frozen_joints=self.reset_lift_frozen_joints,
            )
            log_say("Reset the environment done.", play_sounds=True)

        precise_sleep(max(self.reset_time_s - (time.perf_counter() - start_time), 0.0))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render(obs)

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {TeleopEvents.IS_INTERVENTION: False},
        )

    def render(self, observation: dict | None = None) -> None:
        """Display robot camera feeds in OpenCV windows (one per camera).

        Requires a GUI-enabled OpenCV build (``opencv-python``). With the headless
        build (``opencv-python-headless``) the display is skipped after a one-time
        warning so the control loop keeps running.
        """
        import cv2

        current_observation = observation if observation is not None else self._get_observation()
        if current_observation is None:
            return

        pixels = current_observation.get("pixels", {})
        if not isinstance(pixels, dict):
            return

        try:
            for cam_key, img in pixels.items():
                cv2.imshow(cam_key, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except cv2.error:
            if not getattr(self, "_render_gui_warned", False):
                self._render_gui_warned = True
                logging.warning(
                    "OpenCV has no GUI support (opencv-python-headless); camera display disabled. "
                    "Install the GUI build to show camera windows: pip install 'lerobot[opencv-gui]'"
                )

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":  # 仿真环境。真机跳过该分支
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        require_package("gym-hil", extra="hilserl", import_name="gym_hil")
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment。真机环境
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"

    robot = make_robot_from_config(cfg.robot)  # 获取真机机器人实例
    teleop_device = make_teleoperator_from_config(cfg.teleop)  # 获取摇操设备实例
    teleop_device.connect()  # 连接摇操设备

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_pose = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None
    reset_lift_offset_z = cfg.processor.reset.reset_lift_offset_z if cfg.processor.reset is not None else 0.0
    reset_lift_frozen_joints = (
        cfg.processor.reset.reset_lift_frozen_joints if cfg.processor.reset is not None else None
    )

    # Build the kinematics solver used to lift the end-effector before resetting.
    reset_kinematics = None
    if reset_lift_offset_z > 0.0 and cfg.processor.inverse_kinematics is not None:
        motor_names = list(robot.bus.motors.keys())
        reset_kinematics = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_lift_offset_z=reset_lift_offset_z,
        reset_kinematics=reset_kinematics,
        reset_lift_frozen_joints=reset_lift_frozen_joints,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            GymHILAdapterProcessorStep(),
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
        ]

        # Add time limit processor if reset config exists
        if cfg.processor.reset is not None:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
            )

        env_pipeline_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]
        )

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    motor_names = list(env.robot.bus.motors.keys())

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    add_ee_pose = (
        cfg.processor.observation is not None and cfg.processor.observation.add_ee_pose_to_observation
    )
    if kinematics_solver is not None and add_ee_pose:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    # Only add if max_gripper_pos is explicitly configured (required for normalization)
    if (
        cfg.processor.gripper is not None
        and cfg.processor.gripper.use_gripper
        and cfg.processor.max_gripper_pos is not None
    ):
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
        ),
    ]

    # Replace InverseKinematicsProcessor with new kinematic processors
    if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        # Add EE bounds and safety processor
        inverse_kinematics_steps = [
            MapTensorToDeltaActionDictStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
            ),
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
            ),
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            InverseKinematicsRLStep(
                kinematics=kinematics_solver, motor_names=motor_names, initial_guess_current_joints=False
            ),
        ]
        action_pipeline_steps.extend(inverse_kinematics_steps)
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()

    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions

    # Merge env and action-processor info: env wins for str keys, action-processor
    # wins for `TeleopEvents` enum keys
    action_info = processed_action_transition[TransitionKey.INFO]
    new_info = info.copy()
    for key, value in action_info.items():
        if isinstance(key, TeleopEvents):
            new_info[key] = value

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    return new_transition


def reset_and_build_transition(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """Reset env + processors and return the first env-processed transition."""
    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    complementary_data: dict[str, Any] = {}
    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    return env_processor(data=transition)


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Teleop stick: intervene / take control")
    print("- RIGHT button: mark episode successful (reward=1)")
    print("- LEFT button: fail and re-record the episode")
    print("- Press Ctrl+C to exit")
    if cfg.env.processor.observation is not None and cfg.env.processor.observation.display_cameras:
        print("Camera windows (front/side) are shown; close them with a key press or q to quit.")
    print()

    transition = reset_and_build_transition(env, env_processor, action_processor)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        if teleop_device:
            action_features = teleop_device.action_features
        else:
            action_features = {
                "dtype": "float32",
                "shape": (4,),
                "names": ["delta_x", "delta_y", "delta_z", "gripper"],
            }
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    if cfg.mode == "record":
        print(f"=== Episode 1/{cfg.dataset.num_episodes_to_record} START ===")
        print("   Wait for the robot to reset, then do the task.")
        print()

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Create a neutral action (no movement)
            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            if cfg.mode == "record":
                action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                    "teleop_action", transition[TransitionKey.ACTION]
                )
                frame = {
                    **observation,
                    ACTION: action_to_record.cpu(),
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                        "discrete_penalty", 0.0
                    )
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)

            episode_step += 1

            # Handle episode termination
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                info = transition[TransitionKey.INFO]
                success = bool(info.get(TeleopEvents.SUCCESS, False))
                rerecord = bool(info.get(TeleopEvents.RERECORD_EPISODE, False))
                terminate = bool(info.get(TeleopEvents.TERMINATE_EPISODE, False))

                if success:
                    status = "SUCCESS (right button reward=1)"
                elif rerecord or terminate:
                    status = "FAILED (left button) -> re-record"
                elif truncated:
                    status = "TIMEOUT (time limit, reward=0)"
                else:
                    status = "TERMINATED"

                print(
                    f"=== Episode {episode_idx + 1} END: {status}"
                    f" | reward={transition[TransitionKey.REWARD]}"
                    f" | steps={episode_step} time={episode_time:.1f}s ==="
                )
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if rerecord:
                        print(f"   Re-recording episode {episode_idx} (discarding recorded frames)")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        print(f"   Saving episode {episode_idx} to {cfg.dataset.repo_id}")
                        dataset.save_episode()

                # Reset for new episode
                transition = reset_and_build_transition(env, env_processor, action_processor)
                if episode_idx < cfg.dataset.num_episodes_to_record:
                    print(f"\n=== Episode {episode_idx + 1}/{cfg.dataset.num_episodes_to_record} START ===")
                    print("   Wait for the robot to reset, then do the task.")
                    print()

            # Maintain fps timing
            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
    finally:
        if dataset is not None and dataset.writer is not None and dataset.writer.image_writer is not None:
            logging.info("Waiting for image writer to finish...")
            dataset.writer.image_writer.stop()

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Finalizing dataset before pushing to hub")
        dataset.finalize()
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    cfg: GymManipulatorConfig,
) -> None:
    """回放数据集轨迹。

    两种模式（``cfg.dataset.replay_mode``）：

    - ``"action"``（默认）：把记录的 ACTION（末端增量，与 policy 输出同一空间）
      通过 ``step_env_and_process_transition`` 喂回动作管线（干预覆盖 → EE 积分 → IK
      → 关节指令），与 record 完全相同的处理路径，用来模拟"policy 输出动作的执行情况"。
      注意这样复现的是**指令轨迹**；与录制画面（真实电机有跟踪滞后）会有一两度的
      物理差异，属正常。
    - ``"joint"``：不做动作处理，直接把记录的实际关节角（``observation.state``）
      发给机器人，精确复现录制时的画面。
    """
    episode_ids = (
        [cfg.dataset.replay_episode] if cfg.dataset.replay_episode is not None else None
    )  # None 表示加载全部集

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episode_ids,
        download_videos=False,
    )
    motor_names = list(env.robot.bus.motors.keys())
    # 从 episode_index 列推导每集的帧区间 (start, end)（半开区间）
    episode_index = np.asarray(dataset.hf_dataset["episode_index"])
    starts = np.concatenate(([0], np.nonzero(np.diff(episode_index))[0] + 1))
    episodes_indices = list(zip(starts, np.append(starts[1:], len(episode_index)), strict=True))

    mode = cfg.dataset.replay_mode
    if mode not in ("action", "joint"):
        raise ValueError(f"replay_mode must be 'action' or 'joint', got {mode!r}")
    logging.info("Replay mode: %s", mode)

    for _ep_idx, (start, end) in enumerate(episodes_indices):
        if mode == "action":
            # reset 环境 + 重置处理器状态（EEReferenceAndDelta / IK 等跨帧状态清零），
            # 与 record 每集开始时一致
            transition = reset_and_build_transition(env, env_processor, action_processor)
            actions = dataset.select_columns(ACTION)[ACTION][start:end]
            for action_data in actions:
                start_time = time.perf_counter()
                transition = step_env_and_process_transition(
                    env=env,
                    transition=transition,
                    action=action_data,
                    env_processor=env_processor,
                    action_processor=action_processor,
                )
                precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))
        else:  # joint
            _, _ = env.reset()
            states = dataset.select_columns(OBS_STATE)[OBS_STATE][start:end]
            for joint_pos in states:
                start_time = time.perf_counter()
                action = {f"{name}.pos": float(joint_pos[i]) for i, name in enumerate(motor_names)}
                env.robot.send_action(action)
                precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env, teleop_device = make_robot_env(
        cfg.env
    )  # 获取真机环境（获取state、reset环境）、摇操设备(设备连接、断开、数据读取)
    # replay 时不接入实时摇杆（避免干预覆盖录制动作）；record 时正常接入
    teleop_for_processor = None if cfg.mode == "replay" else teleop_device
    # 预先实现数据处理，供control调用：图像/观测的预处理（裁剪→限时→加batch），负责"看到什么"、动作数据管线(把动作意愿经过"干预覆盖 + 边界 + IK 反解"变成真机关节角)
    env_processor, action_processor = make_processors(env, teleop_for_processor, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment action space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    # 让机械臂照着数据集里已录好的轨迹重新走一遍，仅执行选中的episode
    """
    json文件里：
    "mode": "replay",
    "replay_episode": null      // 回放整个数据集所有帧

    命令：
    lerobot.rl.gym_manipulator --config-path=src/lerobot/configs/env_config_so101_spacemouse.json
    """
    if cfg.mode == "replay":
        replay_trajectory(env, env_processor, action_processor, cfg)
        exit()
    # "边摇边录"（人在环采集，写数据集）
    """
    "mode": "record",
    "num_episodes_to_record": 10
    命令同上回放命令
    """
    control_loop(env, env_processor, action_processor, teleop_device, cfg)


if __name__ == "__main__":
    main()
