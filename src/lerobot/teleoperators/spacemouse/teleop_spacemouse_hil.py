# !/usr/bin/env python

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
"""SOARM(SO101) 机械臂的 SpaceMouse HIL-SERL 摇操器。

面向 HIL-SERL（人在环路的样本高效强化学习）的 SpaceMouse 遥操器，与
``SpaceMouseSoarmEETeleop``（关节空间末端摇操）不同，本类输出**末端平移增量**
（delta_x/delta_y/delta_z），与 gamepad/keyboard_ee 遥操器保持同一动作协议，
可直接接入 ``lerobot.rl`` 的处理器流水线（``InterventionActionProcessorStep``
→ ``MapTensorToDeltaActionDictStep`` → IK 链）。逆运动学由 RL 流水线统一求解，
遥操器自身无状态、不依赖机械臂反馈。

轴映射（3 自由度平移，roll/pitch/yaw 忽略）：
    X: delta_x（末端前后平移）
    Y: delta_y（末端横向平移，由 IK 解 shoulder_pan 实现）
    Z: delta_z（末端上下平移）
    左键: 失败/重录本集（TERMINATE_EPISODE + RERECORD_EPISODE）
    右键: 标记成功（SUCCESS → reward=1，按配置自动结束 episode）

按键采用边沿触发：在 ``get_action()`` 中检测按下瞬间并置标志，
``get_teleop_events()`` 消费该标志（两个方法每个环境步都会被调用一次）。
"""

import time

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.decorators import check_if_not_connected

from .configuration_spacemouse import SpaceMouseSoarmHILTeleopConfig
from .teleop_spacemouse import SpaceMouseSoarmEETeleop


class SpaceMouseSoarmHILTeleop(SpaceMouseSoarmEETeleop):
    """SOARM(SO101) 机械臂的 SpaceMouse HIL-SERL 摇操器（末端平移增量 + 奖励按键）。"""

    config_class = SpaceMouseSoarmHILTeleopConfig
    name = "spacemouse_soarm_hil"

    def __init__(self, config: SpaceMouseSoarmHILTeleopConfig):
        super().__init__(config)
        # HIL 模式输出无状态增量动作，不需要机械臂反馈
        self.needs_feedback = False
        # 按键边沿触发状态（get_action 置位，get_teleop_events 消费）
        self._prev_button_right = False
        self._success_pressed = False
        self._failure_pressed = False

    @property
    def action_features(self) -> dict:
        # 与 GamepadTeleop / KeyboardEndEffectorTeleop 一致的动作协议，
        # 保证录制数据集的 action schema 与策略动作空间（3 维 delta）对齐。
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """读取最新 SpaceMouse 状态，返回末端平移增量（各轴经死区重缩放后 ∈ [-1,1]）。"""
        now = time.perf_counter()
        state = self._read_latest_state()
        has_input = state is not None and not self._state_is_stale(state, now) and self._joystick_moved(state)

        delta_x, delta_y, delta_z = 0.0, 0.0, 0.0
        if has_input:
            delta_x = self._axis_value(state.x)
            delta_y = self._axis_value(state.y)
            delta_z = self._axis_value(state.z)
        self._last_input_active = has_input

        # 按键边沿检测（按下瞬间触发一次；左键=失败/重录，右键=成功）
        if state is not None and state.buttons is not None and len(state.buttons) >= 2:
            btn_left = bool(state.buttons[0])
            btn_right = bool(state.buttons[1])
            if btn_right and not self._prev_button_right:
                self._success_pressed = True
            if btn_left and not self._prev_button_left:
                self._failure_pressed = True
            self._prev_button_left = btn_left
            self._prev_button_right = btn_right

        return {
            "delta_x": float(delta_x),
            "delta_y": float(delta_y),
            "delta_z": float(delta_z),
        }

    def get_teleop_events(self) -> dict[str, bool]:
        """返回摇操控制事件，供 ``AddTeleopEventsAsInfoStep`` 消费。

        返回:
            - is_intervention: 摇杆当前是否有新鲜输入（干预接管）
            - terminate_episode: 左键按下（失败终止）
            - success: 右键按下（reward=1）
            - rerecord_episode: 左键按下（录制模式下重录本集）
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        success = self._success_pressed
        failure = self._failure_pressed
        self._success_pressed = False
        self._failure_pressed = False

        return {
            TeleopEvents.IS_INTERVENTION: bool(self._last_input_active),
            TeleopEvents.TERMINATE_EPISODE: failure,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: failure,
        }

    def send_feedback(self, feedback: dict) -> None:
        # HIL 模式输出无状态增量动作，机械臂反馈无意义
        pass
