#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""SOARM(SO101) 机械臂的 SpaceMouse 末端摇操器。

通过 3Dconnexion SpaceMouse 实现 6 自由度末端连续控制，控制方案参考
`lerobot-teleoperator-spacemouse`：

- 每帧排空 HID 缓冲，只取最新一帧状态，降低摇操延迟；
- 输入超时（input_timeout_s）：超过该时长未收到新的 HID 状态即视为无输入，
  目标位姿立即冻结、机械臂停止移动（松手即停）；
- 摇杆输入按固定步长积分到目标末端位姿；积分在 4x4 齐次变换矩阵上进行
  （平移直接相加、旋转用矩阵复合），避免欧拉角在 pitch≈90°（SO101 的
  home 位姿 wrist_flex=86° 已接近）时的万向锁翻转导致的抖动；
- IK 用 placo（URDF 运动学）求解，姿态权重远低于位置权重（0.01 vs 1.0），
  因 SO101 无偏航关节、6D 目标本就不可达，低姿态权重避免 IK 在不可达
  姿态上"打架"振荡。

使用 pyspacemouse 并自定义设备参数，修正为右手定则的轴映射：
    X:     末端前后平移
    Y:     底座旋转 (shoulder_pan)
    Z:     末端上下平移
    Roll:  末端横滚（绕局部 X 轴旋转）——本设备 pitch 传感器（物理 X 轴倾斜）驱动，方向已校正
    Pitch: 末端俯仰（绕局部 Y 轴旋转）——本设备 roll 传感器（物理 Y 轴倾斜）驱动，方向已校正
    Yaw:   忽略（SO101 无偏航关节）
    左键:  缓速复位到推荐初始位姿（home_duration 秒内线性插值，到位后保持，操作摇杆即恢复）
    右键:  无（夹爪固定不动，不提供夹爪按键）
"""

import contextlib
import logging
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from lerobot.lerobot_types import RobotAction
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..teleoperator import Teleoperator
from .configuration_spacemouse import SpaceMouseSoarmEETeleopConfig

try:
    import pyspacemouse
    from pyspacemouse.config_helpers import create_device_info

    _pyspacemouse_available = True
except ImportError:
    _pyspacemouse_available = False

logger = logging.getLogger(__name__)

# 自定义设备参数：修正为右手定则的轴映射。
# pyspacemouse 默认映射存在 roll/pitch 互换、yaw 方向相反的问题。
_SPACEMOUSE_DEVICE_NAME = "SpaceMouseCompact"
_SPACEMOUSE_VENDOR_ID = 0x256F
_SPACEMOUSE_PRODUCT_ID = 0xC635
_SPACEMOUSE_CORRECTED_MAPPINGS = {
    "x":     (1, 1, 2,  1),
    "y":     (1, 3, 4, -1),
    "z":     (1, 5, 6, -1),
    "roll":  (2, 1, 2,  1),
    "pitch": (2, 3, 4, -1),
    "yaw":   (2, 5, 6, -1),
}
_SPACEMOUSE_BUTTONS = {
    "LEFT":  (3, 1, 0),
    "RIGHT": (3, 1, 1),
}

# 目标末端位姿的位置工作空间限制（按硬件真实关节范围采样 FK 得到的可达范围，留余量）。
# 实测 4 臂关节（pan=0）末端 x 可达 [-0.34, 0.48]，这里 X 上限取 0.40 以便向前伸展。
# 每行为 [x, y, z]，第一行下限、第二行上限。
_POS_LIMIT = np.array([[-0.22, -0.4, -0.15], [0.40, 0.4, 0.31]], dtype=np.float64)

# IK 用到的 4 个臂关节（shoulder_pan 由摇杆直接控制，不参与 IK）
_ARM_JOINT_NAMES = ["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
_URDF_PATH = str(Path(__file__).parent / "urdfs" / "so101_new_calib.urdf")


class SpaceMouseSoarmEETeleop(Teleoperator):
    """SOARM(SO101) 机械臂的 SpaceMouse 末端摇操器。"""

    config_class = SpaceMouseSoarmEETeleopConfig
    name = "spacemouse_soarm_ee"

    def __init__(self, config: SpaceMouseSoarmEETeleopConfig):
        super().__init__(config)
        self.config = config
        self.translation_step_m = config.translation_step_m
        self.rotation_step_rad = config.rotation_step_rad
        self.deadzone = config.deadzone
        self.input_timeout_s = config.input_timeout_s
        self.read_drain_count = config.read_drain_count
        self.max_joint_change = config.max_joint_change
        self.home_duration = config.home_duration
        self.home_hold_duration = config.home_hold_duration
        self.home_on_start = config.home_on_start

        self._device = None
        self._kinematics = None
        self._initialized = False
        self._init_joint_received = False
        self._homing = False
        self.needs_feedback = True

        # 内部状态（关节 0-4 使用弧度，关节 5 夹爪使用 0-100 范围）。
        # 推荐初始位姿（pan=-3.09°, lift=-43.43°, elbow=54.43°, wrist_flex=47.08°,
        # wrist_roll=-86.17°, gripper=5.66）作为"左键复位"目标，
        # 以及未收到反馈时的兜底初始位姿。
        self.init_qpos_home = np.array(
            [-0.061, -0.692, 1.011, 1.156, -1.500, 5.732],
            dtype=np.float64,
        )
        # current_qpos: 从动臂当前关节角（IK 的起始点，由反馈更新）
        self.current_qpos = self.init_qpos_home.copy()
        # target_qpos: 目标关节角（shoulder_pan 由摇杆控制，夹爪固定不动）
        self.target_qpos = self.init_qpos_home.copy()
        # target_T: 目标末端位姿（4x4 齐次变换矩阵，臂关节坐标系）
        self.target_T = np.eye(4, dtype=np.float64)
        # 回 home 流程状态
        self._home_T = np.eye(4, dtype=np.float64)
        self._homing_start_qpos = np.zeros(6, dtype=np.float64)
        self._homing_progress = 0.0
        self._homing_hold_remaining: float | None = 0.0
        self._homing_phase = "move"

        self._prev_button_left = False
        # 上一帧是否有摇杆输入（用于检测"锁死 → 输入"的恢复时刻）
        self._last_input_active = False
        # 记录相邻两次 get_action 之间的真实墙钟间隔（仅用于回 home 插值进度）
        self._last_frame_time = 0.0

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

    @property
    def feedback_features(self) -> dict:
        return {}

    @check_if_already_connected
    def connect(self) -> None:
        """连接 SpaceMouse 设备并初始化 IK 求解器。"""
        if not _pyspacemouse_available:
            require_package("pyspacemouse", extra="spacemouse")

        # 扫描已连接的 SpaceMouse 设备
        found = pyspacemouse.get_connected_devices()
        if not found:
            raise RuntimeError("No SpaceMouse device found. Is it connected?")

        # 用修正后的轴映射创建设备参数，非阻塞方式打开
        self._device_spec = create_device_info(
            name=_SPACEMOUSE_DEVICE_NAME,
            vendor_id=_SPACEMOUSE_VENDOR_ID,
            product_id=_SPACEMOUSE_PRODUCT_ID,
            mappings=_SPACEMOUSE_CORRECTED_MAPPINGS,
            buttons=_SPACEMOUSE_BUTTONS,
        )
        self._device = pyspacemouse.open(device_spec=self._device_spec, nonblocking=True)
        logger.info("SpaceMouse connected: %s", found[0])

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self._device is not None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _init_kinematics(self) -> None:
        """初始化 placo IK 求解器（4 个臂关节，姿态权重 0.01）。"""
        if self._kinematics is None:
            require_package("placo", extra="placo-dep")
            self._kinematics = RobotKinematics(
                urdf_path=_URDF_PATH,
                target_frame_name="gripper_frame_link",
                joint_names=_ARM_JOINT_NAMES,
            )

    def _arm_fk(self) -> np.ndarray:
        """当前臂关节角的末端位姿（4x4 矩阵，pan 置 0 即臂关节坐标系）。"""
        arm_deg = np.rad2deg(self.current_qpos[1:5])
        return self._kinematics.forward_kinematics(arm_deg)

    def _arm_ik(self) -> None:
        """对目标末端位姿求解 IK，更新 current_qpos 的臂关节(1-4)。"""
        self._init_kinematics()
        q_cur_deg = np.rad2deg(self.current_qpos[1:5])
        q_target_deg = self._kinematics.inverse_kinematics(
            q_cur_deg,
            self.target_T,
            position_weight=1.0,
            orientation_weight=0.01,
        )
        # 安全钳制：单关节每帧最大变化量（placo 输出本身平滑，仅在异常时生效）
        delta_deg = np.clip(
            q_target_deg - q_cur_deg,
            -np.degrees(self.max_joint_change),
            np.degrees(self.max_joint_change),
        )
        self.current_qpos[1:5] = np.deg2rad(q_cur_deg + delta_deg)

    def _lazy_init_from_feedback(self, obs: dict) -> None:
        """首次收到从动臂反馈时，用其真实关节角初始化内部状态。"""
        if self._init_joint_received:
            return
        if "shoulder_lift.pos" not in obs:
            return
        # 反馈为角度值（度），转换为弧度；夹爪为 0-100 范围，不转换
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
        """把当前关节角对应的末端位姿同步为初始目标，避免启动时跳变。"""
        self._init_kinematics()
        self.target_T = self._arm_fk()
        self.target_qpos = self.current_qpos.copy()

    def _start_homing(self) -> None:
        """左键触发：进入回初始位姿模式（关节空间匀速线性插值 + 到位后保持）。"""
        self._init_kinematics()
        self._home_T = self._kinematics.forward_kinematics(np.rad2deg(self.init_qpos_home[1:5]))
        # 起始关节目标：底座取 target_qpos，臂关节取 current_qpos，夹爪固定不动
        self._homing_start_qpos = np.array(
            [
                self.target_qpos[0],
                self.current_qpos[1],
                self.current_qpos[2],
                self.current_qpos[3],
                self.current_qpos[4],
                self.target_qpos[5],
            ],
            dtype=np.float64,
        )
        self._homing_progress = 0.0
        self._homing_hold_remaining = self.home_hold_duration
        self._homing_phase = "move"
        self._homing = True

    def _update_homing(self, frame_dt: float) -> bool:
        """每帧推进回 home 流程：阶段一匀速线性插值到初始位姿，阶段二持续保持。

        home_hold_duration 为 None 时无限保持（由摇杆操作结束），为有限值时到点
        结束。返回 True 表示移动 + 保持全部完成。
        """
        self._homing_progress += frame_dt
        if self._homing_progress < self.home_duration:
            # 阶段一：关节空间匀速线性插值（与 move_soarm_to_state 一致，不经 IK）
            self._homing_phase = "move"
            t = self._homing_progress / self.home_duration
            self.target_qpos[0] = self._homing_start_qpos[0] + (
                self.init_qpos_home[0] - self._homing_start_qpos[0]
            ) * t
            self.current_qpos[1:5] = self._homing_start_qpos[1:5] + (
                self.init_qpos_home[1:5] - self._homing_start_qpos[1:5]
            ) * t
            # 同步末端位姿目标（供回位结束后 IK 无缝衔接）
            self.target_T = self._arm_fk()
        else:
            # 阶段二：到位后持续保持初始位姿
            self._homing_phase = "hold"
            self.target_qpos[0] = self.init_qpos_home[0]
            self.current_qpos[1:5] = self.init_qpos_home[1:5]
            self.target_T = self._home_T.copy()
            # 有限保持时长时到点自动结束
            if self._homing_hold_remaining is not None:
                self._homing_hold_remaining -= frame_dt
                if self._homing_hold_remaining <= 0.0:
                    return True
        return False

    def _joystick_moved(self, state) -> bool:
        """任一轴读数超过死区即视为用户操作了摇杆。"""
        axes = [state.x, state.y, state.z, state.roll, state.pitch, state.yaw]
        return any(abs(float(axis)) > self.deadzone for axis in axes)

    def _read_latest_state(self):
        """排空 HID 缓冲，返回最新一帧状态（读到重复时间戳即认为已是最新）。"""
        state = None
        last_t = object()
        for _ in range(max(1, self.read_drain_count)):
            next_state = self._device.read()
            if next_state is None:
                return state
            state = next_state
            t = getattr(state, "t", None)
            if t == last_t:
                break
            last_t = t
        return state

    def _state_is_stale(self, state, now: float) -> bool:
        """输入超时判断：超过 input_timeout_s 未收到新的 HID 状态即视为无输入。"""
        t = getattr(state, "t", None)
        return (
            t is not None
            and t >= 0.0
            and self.input_timeout_s > 0.0
            and now - float(t) > self.input_timeout_s
        )

    def _axis_value(self, raw: float) -> float:
        """死区 + 重缩放：死区内归零，死区外重映射到 [0,1]，不损失小幅操作灵敏度。"""
        value = float(raw)
        if abs(value) <= self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _integrate_input(self, state) -> None:
        """把摇杆输入按固定步长积分到目标末端位姿（4x4 矩阵，步长与帧间隔无关）。

        平移直接加到位置列，旋转用矩阵复合（绕当前目标姿态的局部轴旋转），
        避免欧拉角加法积分的万向锁问题。
        """
        # Y 轴 → 底座旋转 (shoulder_pan)
        self.target_qpos[0] -= self._axis_value(state.y) * self.rotation_step_rad
        # X/Z 平移
        self.target_T[:3, 3] += np.array(
            [
                self._axis_value(state.x) * self.translation_step_m,
                0.0,
                self._axis_value(state.z) * self.translation_step_m,
            ]
        )
        # Roll/Pitch 旋转（局部轴）：
        # 本设备上 roll 传感器对应物理 Y 轴倾斜（驱动臂的俯仰）、pitch 传感器对应物理 X 轴倾斜（驱动臂的横滚）。
        # 负号：使俯仰/横滚的物理方向正确（推摇杆向某侧时末端朝对应方向转动）。
        delta_r = Rotation.from_rotvec(
            [
                self._axis_value(state.pitch) * self.rotation_step_rad,
                -self._axis_value(state.roll) * self.rotation_step_rad,
                0.0,  # Yaw 忽略：SO101 无偏航关节
            ]
        )
        self.target_T[:3, :3] = self.target_T[:3, :3] @ delta_r.as_matrix()
        # 限制在工作空间范围内
        self.target_T[:3, 3] = np.clip(self.target_T[:3, 3], _POS_LIMIT[0], _POS_LIMIT[1])

    def _initialize(self, obs: dict | None = None) -> None:
        """初始化内部状态（优先用反馈，其次用预设 home 位姿）。"""
        if self._initialized:
            return
        if obs is not None:
            self._lazy_init_from_feedback(obs)
        if not self._init_joint_received:
            self.current_qpos = self.init_qpos_home.copy()
        self._sync_target_from_current()
        # 启动时缓速移动到推荐初始位姿（回 home 逻辑与左键复位共用）
        if self.home_on_start:
            self._start_homing()
        self._initialized = True

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """读取 SpaceMouse 最新状态，积分目标末端位姿，经 IK 求解后返回关节目标。"""
        now = time.perf_counter()

        # 首次调用时初始化内部状态
        if not self._initialized:
            self._initialize()

        # 相邻帧间隔（仅用于回 home 插值进度）
        dt = now - self._last_frame_time if self._last_frame_time > 0 else 1 / 60
        self._last_frame_time = now
        dt = min(dt, 0.1)

        # 排空 HID 缓冲取最新一帧；超时未收到新状态视为无输入
        state = self._read_latest_state()
        has_input = (
            state is not None and not self._state_is_stale(state, now) and self._joystick_moved(state)
        )

        # 从锁死状态恢复输入时，先把目标末端位姿同步到当前关节角，
        # 避免机械臂从"上一帧追赶时遗留的陈旧目标"突然起跳
        if has_input and not self._last_input_active:
            self.target_T = self._arm_fk()

        if state is not None:
            # 回位保持阶段：用户操作摇杆（新鲜输入）→ 立即结束保持，恢复正常摇操
            if self._homing and self._homing_phase == "hold" and has_input:
                self._homing = False
            if not self._homing and has_input:
                self._integrate_input(state)
            # 移动阶段（move）忽略摇杆输入，避免与复位目标打架

            # 按键处理（边沿触发，只在按下瞬间生效一次）
            if state.buttons is not None and len(state.buttons) >= 2:
                btn_left = bool(state.buttons[0])
                # 左键：缓速复位到初始位姿
                if btn_left and not self._prev_button_left:
                    self._start_homing()
                self._prev_button_left = btn_left
                # 右键不使用：夹爪固定不动

        # 回 home：每帧向初始位姿缓速逼近（与摇杆状态无关，HID 丢帧也能继续）
        if self._homing and self._update_homing(dt):
            self._homing = False

        # 仅在【有摇杆输入】且【非回位】时求解 IK；无输入时锁死上一帧关节角
        if not self._homing and has_input:
            self._arm_ik()

        # 记录本帧输入状态（供下一帧检测锁死→输入的恢复时刻）
        self._last_input_active = has_input

        # 输出关节目标（弧度转角度；夹爪为 0-100 范围不转换）
        return {
            "shoulder_pan.pos": float(np.rad2deg(self.target_qpos[0])),
            "shoulder_lift.pos": float(np.rad2deg(self.current_qpos[1])),
            "elbow_flex.pos": float(np.rad2deg(self.current_qpos[2])),
            "wrist_flex.pos": float(np.rad2deg(self.current_qpos[3])),
            "wrist_roll.pos": float(np.rad2deg(self.current_qpos[4])),
            "gripper.pos": float(self.target_qpos[5]),
        }

    def send_feedback(self, feedback: dict) -> None:
        """接收从动臂反馈，更新内部状态（底座与夹爪直接跟随实际值）。"""
        was_init = self._init_joint_received
        self._lazy_init_from_feedback(feedback)
        if self._init_joint_received and not was_init:
            self._sync_target_from_current()
        if self._init_joint_received:
            # 底座角与夹爪位置由反馈直接更新，臂关节(1-4)交给 IK 自主追踪
            self.current_qpos[0] = np.deg2rad(
                feedback.get("shoulder_pan.pos", np.rad2deg(self.current_qpos[0]))
            )
            self.current_qpos[5] = feedback.get("gripper.pos", self.current_qpos[5])
        if not self._initialized:
            self._initialize(feedback)

    def disconnect(self) -> None:
        """关闭 SpaceMouse 设备并重置内部状态。"""
        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
            self._device = None
        self._initialized = False
        self._init_joint_received = False
