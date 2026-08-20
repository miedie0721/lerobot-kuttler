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
"""SOARM(SO101) 机械臂的 SpaceMouse 末端遥操作器配置。"""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse_soarm_ee")
@dataclass
class SpaceMouseSoarmEETeleopConfig(TeleoperatorConfig):
    """SOARM(SO101) 机械臂的 SpaceMouse 末端遥操作器配置。

    摇杆输入按固定步长积分到目标末端位姿，每帧由逆运动学(IK)求解关节角。
    按键：左键 → 复位到初始位姿（夹爪固定不动，不提供夹爪按键）。

    属性:
        translation_step_m: 推杆满偏转时每帧末端的平移量 (m)。60Hz 下默认 0.001m/帧 ≈ 0.06 m/s。
        rotation_step_rad: 推杆满偏转时每帧末端的旋转量 (rad)。60Hz 下默认 0.003rad/帧 ≈ 0.18 rad/s。
        deadzone: 摇杆死区，轴读数绝对值低于该值视为无输入。
        input_timeout_s: 超过该时长未收到新的 HID 状态即视为无输入（目标位姿冻结，松手即停）。
        read_drain_count: 每次 get_action 最多读取的 HID 状态数（排空缓冲，取最新一帧）。
        max_joint_change: 每帧 IK 求解后单个关节的最大角变化量 (radians)。
        home_duration: 左键复位/启动回位时的移动时长（秒）。
        home_hold_duration: 回位到位后的保持时长（秒），None 表示无限保持（操作摇杆即结束）。
        home_on_start: 摇操启动时是否自动缓速移动到推荐初始位姿。
    """

    # 推杆满偏转时每帧末端的平移量 (m)。
    translation_step_m: float = 0.001
    # 推杆满偏转时每帧末端的旋转量 (rad)。60Hz 下 0.003rad/帧 ≈ 0.18 rad/s ≈ 10.3°/s。
    rotation_step_rad: float = 0.003
    # 摇杆死区，轴读数绝对值低于该值视为无输入（滤除 HID 静止噪声）。
    deadzone: float = 0.10
    # 输入超时（秒）：超过该时长未收到新的 HID 状态即视为无输入。
    input_timeout_s: float = 0.08
    # 每次 get_action 最多读取的 HID 状态数。
    read_drain_count: int = 32
    # 单次 get_action 中单个关节允许的最大角变化量 (radians)。
    # 用于限制机械臂单帧运动幅度、保证安全；值越大跟随越灵敏，但运动越激进。
    max_joint_change: float = 0.2
    # 左键复位/启动回位时的移动时长（秒）。
    home_duration: float = 2.0
    # 回位到位后的保持时长（秒）。None 表示不限时长：用户操作摇杆即结束保持。
    home_hold_duration: float | None = None
    # 摇操启动时是否自动缓速移动到推荐初始位姿（init_qpos_home）。
    home_on_start: bool = True
