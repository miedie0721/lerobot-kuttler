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
"""Configuration for SpaceMouse teleoperator."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("spacemouse_soarm_ee")
@dataclass
class SpaceMouseSoarmEETeleopConfig(TeleoperatorConfig):
    """Configuration for SpaceMouse end-effector teleoperator for SOARM (SO101).

    Uses a 3Dconnexion SpaceMouse for continuous 6-DOF end-effector control.
    The SpaceMouse provides velocity-like inputs that are integrated into a
    target end-effector pose, with IK solved each frame.

    Buttons:
        Left button  → Gripper close
        Right button → Gripper open

    Attributes:
        translation_speed: Max translation speed at full deflection (m/s).
        rotation_speed: Max rotation speed at full deflection (rad/s).
        gripper_increment: Gripper position increment per button press.
        max_joint_change: Max per-joint angular change per IK step (radians).
    """

    translation_speed: float = 0.1
    rotation_speed: float = 1.0
    gripper_increment: float = 1.0
    max_joint_change: float = 0.1
