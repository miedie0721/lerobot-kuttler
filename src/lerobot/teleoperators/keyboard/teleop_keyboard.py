#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import threading
import time
from queue import Queue
from typing import Any

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardRoverTeleopConfig,
    KeyboardSoarmEETeleopConfig,
    KeyboardTeleopConfig,
)
from .kinematics_soarm import lerobot_FK, lerobot_IK

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions; use an X11 session, a "
                "gamepad, or a leader-arm teleoperator instead."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if key == keyboard.Key.up:
                delta_y = -int(val)
            elif key == keyboard.Key.down:
                delta_y = int(val)
            elif key == keyboard.Key.left:
                delta_x = int(val)
            elif key == keyboard.Key.right:
                delta_x = -int(val)
            elif key == keyboard.Key.shift:
                delta_z = -int(val)
            elif key == keyboard.Key.shift_r:
                delta_z = int(val)
            elif key == keyboard.Key.ctrl_r:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = int(val) + 1
            elif key == keyboard.Key.ctrl_l:
                gripper_action = int(val) - 1
            elif val:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            keyboard.Key.shift,
            keyboard.Key.shift_r,
            keyboard.Key.ctrl_r,
            keyboard.Key.ctrl_l,
        ]
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        self.current_pressed.clear()

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }


class KeyboardSoarmEETeleop(KeyboardTeleop):
    """Keyboard end-effector teleoperator for SOARM (SO101) arm.

    Pure end-effector control matching lerobot-kinematics: keys modify
    a target end-effector pose [x, y, z, roll, pitch, yaw], and inverse
    kinematics is solved each frame to produce joint angle actions.

    Key mappings (identical to lerobot-kinematics lerobot_keycon_gpos.py):
        Motion:
            w / s        : X forward / backward
            a / d        : Base rotation (shoulder_pan) - / +
            r / f        : Z up / down
            q / e        : Roll + / -
            g / t        : Pitch + / -
            z / c        : Gripper open / close
        System:
            ESC          : Disconnect
            0            : Reset to home position
    """

    config_class = KeyboardSoarmEETeleopConfig
    name = "keyboard_soarm_ee"

    def __init__(self, config: KeyboardSoarmEETeleopConfig):
        super().__init__(config)
        self.pos_increment = config.pos_increment
        self.rot_increment = config.rot_increment
        self.joint_increment = config.joint_increment
        self.gripper_increment = config.gripper_increment
        self.max_joint_change = config.max_joint_change

        self._initialized = False
        self._init_joint_received = False
        self.needs_feedback = True

        self.lock = threading.Lock()

        # Full 6-DoF internal state
        # current_qpos = actual robot joint state from feedback (radians, used for IK init)
        # target_qpos  = commanded joint state for direct-key axes (radians, accumulates key increments)
        #                Only axes 0 (shoulder_pan) and 5 (gripper) are directly key-controlled.
        #                Axes 1-4 are controlled via IK and target_gpos.
        self.init_qpos_home = np.array(
            [0.0, -1.57, 1.57, 0.0, -1.57, 1.0],
            dtype=np.float64,
        )
        self.current_qpos = self.init_qpos_home.copy()
        self.target_qpos = self.init_qpos_home.copy()
        self.target_gpos = np.zeros(6, dtype=np.float64)

        # Backup for IK failure rollback (matching lerobot-kinematics)
        self.target_gpos_last = np.zeros(6, dtype=np.float64)

        # Workspace limits for target end-effector pose
        # [x_low, y_low, z_low, roll_low, pitch_low, yaw_low]
        self.control_glimit = np.array(
            [[0.125, -0.4, 0.046, -3.1, -0.75, -1.5],
             [0.340,  0.4, 0.23,   2.0,  1.57,  1.5]],
            dtype=np.float64,
        )
        # Joint limits for direct-joint axes (base rotation, gripper)
        # Joints 0-4 in radians, gripper (index 5) in motor range 0-100
        self.control_qlimit = np.array(
            [[-2.1, -3.1, -0.0, -1.375, -1.57, 0.0],
             [ 2.1,  0.0,  3.1,  1.475,  3.1,   100.0]],
            dtype=np.float64,
        )

        # Misc key events queue (non-motion keys)
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "shoulder_pan.pos": 0,
                "shoulder_lift.pos": 1,
                "elbow_flex.pos": 2,
                "wrist_flex.pos": 3,
                "wrist_roll.pos": 4,
                "gripper.pos": 5,
            },
        }

    def _lazy_init_from_feedback(self, obs: dict) -> None:
        """Initialize internal state from the first robot observation (deg -> rad)."""
        if self._init_joint_received:
            return
        if "shoulder_lift.pos" not in obs:
            return
        self.current_qpos = np.array(
            [
                np.deg2rad(obs.get("shoulder_pan.pos", np.rad2deg(self.current_qpos[0]))),
                np.deg2rad(obs.get("shoulder_lift.pos", np.rad2deg(self.current_qpos[1]))),
                np.deg2rad(obs.get("elbow_flex.pos", np.rad2deg(self.current_qpos[2]))),
                np.deg2rad(obs.get("wrist_flex.pos", np.rad2deg(self.current_qpos[3]))),
                np.deg2rad(obs.get("wrist_roll.pos", np.rad2deg(self.current_qpos[4]))),
                obs.get("gripper.pos", self.current_qpos[5]),
            ],
            dtype=np.float64,
        )
        self._init_joint_received = True

    def _sync_target_from_current(self) -> None:
        """Run FK on current arm joints to set the initial target pose and target qpos."""
        arm_q = self.current_qpos[1:5].copy()
        self.target_gpos = lerobot_FK(arm_q)
        self.target_gpos_last = self.target_gpos.copy()
        self.target_qpos = self.current_qpos.copy()

    def _initialize(self, obs: dict | None = None) -> None:
        """One-time initialization on first get_action / send_feedback call."""
        if self._initialized:
            return
        with self.lock:
            if self._initialized:
                return
            if obs is not None:
                self._lazy_init_from_feedback(obs)
            if not self._init_joint_received:
                self.current_qpos = self.init_qpos_home.copy()
            self._sync_target_from_current()
            self._initialized = True

    def _clamp_target_gpos(
        self, axis: int, direction: int, increment: float
    ) -> bool:
        """Check if target_gpos axis can move in direction within control_glimit."""
        if direction > 0:
            return self.target_gpos[axis] <= self.control_glimit[1][axis]
        else:
            return self.target_gpos[axis] >= self.control_glimit[0][axis]

    def _clamp_target_qpos(
        self, axis: int, direction: int, increment: float
    ) -> bool:
        """Check if target_qpos axis can move in direction within control_qlimit."""
        if direction > 0:
            return self.target_qpos[axis] < self.control_qlimit[1][axis] - increment * direction
        else:
            return self.target_qpos[axis] > self.control_qlimit[0][axis] - increment * direction

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        if not self._initialized:
            self._initialize()

        for key_char, is_pressed in list(self.current_pressed.items()):
            if not is_pressed:
                continue

            # --- X: forward/backward ---
            if key_char == "w":
                if self._clamp_target_gpos(0, 1, self.pos_increment):
                    self.target_gpos[0] += self.pos_increment
            elif key_char == "s":
                if self._clamp_target_gpos(0, -1, self.pos_increment):
                    self.target_gpos[0] -= self.pos_increment

            # --- Base rotation (shoulder_pan) ---
            elif key_char == "a":
                if self._clamp_target_qpos(0, -1, self.joint_increment):
                    self.target_qpos[0] -= self.joint_increment
            elif key_char == "d":
                if self._clamp_target_qpos(0, 1, self.joint_increment):
                    self.target_qpos[0] += self.joint_increment

            # --- Z: up/down ---
            elif key_char == "r":
                if self._clamp_target_gpos(2, 1, self.pos_increment):
                    self.target_gpos[2] += self.pos_increment
            elif key_char == "f":
                if self._clamp_target_gpos(2, -1, self.pos_increment):
                    self.target_gpos[2] -= self.pos_increment

            # --- Roll ---
            elif key_char == "q":
                if self._clamp_target_gpos(3, 1, self.rot_increment):
                    self.target_gpos[3] += self.rot_increment
            elif key_char == "e":
                if self._clamp_target_gpos(3, -1, self.rot_increment):
                    self.target_gpos[3] -= self.rot_increment

            # --- Pitch ---
            elif key_char == "g":
                if self._clamp_target_gpos(4, 1, self.rot_increment):
                    self.target_gpos[4] += self.rot_increment
            elif key_char == "t":
                if self._clamp_target_gpos(4, -1, self.rot_increment):
                    self.target_gpos[4] -= self.rot_increment

            # --- Gripper ---
            elif key_char == "z":
                if self._clamp_target_qpos(5, 1, self.gripper_increment):
                    self.target_qpos[5] += self.gripper_increment
            elif key_char == "c":
                if self._clamp_target_qpos(5, -1, self.gripper_increment):
                    self.target_qpos[5] -= self.gripper_increment

            # --- Reset ---
            elif key_char == "0":
                self.current_qpos = self.init_qpos_home.copy()
                self._sync_target_from_current()

            else:
                self.misc_keys_queue.put(key_char)

        arm_joints = self.current_qpos[1:5].copy()
        q_result, ik_ok = lerobot_IK(
            arm_joints,
            self.target_gpos,
            ilimit=10,
            slimit=2,
            tol=1e-3,
            max_joint_change=self.max_joint_change,
        )
        if ik_ok:
            self.current_qpos[1:5] = q_result
            self.target_gpos_last = self.target_gpos.copy()
        else:
            self.target_gpos = self.target_gpos_last.copy()

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "shoulder_pan.pos": float(np.rad2deg(self.target_qpos[0])),
            "shoulder_lift.pos": float(np.rad2deg(self.current_qpos[1])),
            "elbow_flex.pos": float(np.rad2deg(self.current_qpos[2])),
            "wrist_flex.pos": float(np.rad2deg(self.current_qpos[3])),
            "wrist_roll.pos": float(np.rad2deg(self.current_qpos[4])),
            "gripper.pos": float(self.target_qpos[5]),
        }

    def send_feedback(self, feedback: dict) -> None:
        was_init = self._init_joint_received
        self._lazy_init_from_feedback(feedback)
        if self._init_joint_received and not was_init:
            self._sync_target_from_current()
        if self._init_joint_received:
            with self.lock:
                self.current_qpos = np.array(
                    [
                        np.deg2rad(feedback.get("shoulder_pan.pos", np.rad2deg(self.current_qpos[0]))),
                        np.deg2rad(feedback.get("shoulder_lift.pos", np.rad2deg(self.current_qpos[1]))),
                        np.deg2rad(feedback.get("elbow_flex.pos", np.rad2deg(self.current_qpos[2]))),
                        np.deg2rad(feedback.get("wrist_flex.pos", np.rad2deg(self.current_qpos[3]))),
                        np.deg2rad(feedback.get("wrist_roll.pos", np.rad2deg(self.current_qpos[4]))),
                        feedback.get("gripper.pos", self.current_qpos[5]),
                    ],
                    dtype=np.float64,
                )
        if not self._initialized:
            self._initialize(feedback)

    def get_teleop_events(self) -> dict:
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        movement_keys = set("wsadrfqegtzc0")
        is_intervention = any(
            self.current_pressed.get(k, False) for k in movement_keys
        )

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: False,
            TeleopEvents.SUCCESS: False,
            TeleopEvents.RERECORD_EPISODE: False,
        }


class KeyboardRoverTeleop(KeyboardTeleop):
    """
    Keyboard teleoperator for mobile robots like EarthRover Mini Plus.

    Provides intuitive WASD-style controls for driving a mobile robot:
    - Linear movement (forward/backward)
    - Angular movement (turning/rotation)
    - Speed adjustment
    - Emergency stop

    Keyboard Controls:
        Movement:
            - W: Move forward
            - S: Move backward
            - A: Turn left (with forward motion)
            - D: Turn right (with forward motion)
            - Q: Rotate left in place
            - E: Rotate right in place
            - X: Emergency stop

        Speed Control:
            - +/=: Increase speed
            - -: Decrease speed

        System:
            - ESC: Disconnect teleoperator

    Attributes:
        config: Teleoperator configuration
        current_linear_speed: Current linear velocity magnitude
        current_angular_speed: Current angular velocity magnitude

    Example:
        ```python
        from lerobot.teleoperators.keyboard import KeyboardRoverTeleop, KeyboardRoverTeleopConfig

        teleop = KeyboardRoverTeleop(
            KeyboardRoverTeleopConfig(linear_speed=1.0, angular_speed=1.0, speed_increment=0.1)
        )
        teleop.connect()

        while teleop.is_connected:
            action = teleop.get_action()
            robot.send_action(action)
        ```
    """

    config_class = KeyboardRoverTeleopConfig
    name = "keyboard_rover"

    def __init__(self, config: KeyboardRoverTeleopConfig):
        super().__init__(config)
        # Add rover-specific speed settings
        self.current_linear_speed = config.linear_speed
        self.current_angular_speed = config.angular_speed

    @property
    def action_features(self) -> dict:
        """Return action format for rover (linear and angular velocities)."""
        return {
            "linear_velocity": float,
            "angular_velocity": float,
        }

    @property
    def is_calibrated(self) -> bool:
        """Rover teleop doesn't require calibration."""
        return True

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue without clearing held keys"""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                # Only remove key if it's being released
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """
        Get the current action based on pressed keys.

        Returns:
            RobotAction with 'linear_velocity' and 'angular_velocity' keys.
        """
        before_read_t = time.perf_counter()

        self._drain_pressed_keys()

        linear_velocity = 0.0
        angular_velocity = 0.0

        # Check which keys are currently pressed (not released)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        # Linear movement (W/S) - these take priority
        if "w" in active_keys:
            linear_velocity = self.current_linear_speed
        elif "s" in active_keys:
            linear_velocity = -self.current_linear_speed

        # Turning (A/D/Q/E)
        if "d" in active_keys:
            angular_velocity = -self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "a" in active_keys:
            angular_velocity = self.current_angular_speed
            if linear_velocity == 0:  # If not moving forward/back, add slight forward motion
                linear_velocity = self.current_linear_speed * self.config.turn_assist_ratio
        elif "q" in active_keys:
            angular_velocity = self.current_angular_speed
            linear_velocity = 0  # Rotate in place
        elif "e" in active_keys:
            angular_velocity = -self.current_angular_speed
            linear_velocity = 0  # Rotate in place

        # Stop (X) - overrides everything
        if "x" in active_keys:
            linear_velocity = 0
            angular_velocity = 0

        # Speed adjustment
        if "+" in active_keys or "=" in active_keys:
            self.current_linear_speed += self.config.speed_increment
            self.current_angular_speed += self.config.speed_increment * self.config.angular_speed_ratio
            logging.info(
                f"Speed increased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )
        if "-" in active_keys:
            self.current_linear_speed = max(
                self.config.min_linear_speed, self.current_linear_speed - self.config.speed_increment
            )
            self.current_angular_speed = max(
                self.config.min_angular_speed,
                self.current_angular_speed - self.config.speed_increment * self.config.angular_speed_ratio,
            )
            logging.info(
                f"Speed decreased: linear={self.current_linear_speed:.2f}, angular={self.current_angular_speed:.2f}"
            )

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
