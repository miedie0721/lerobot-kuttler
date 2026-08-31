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

from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep


def test_map_delta_action_with_gripper():
    step = MapDeltaActionToRobotActionStep()
    action = step.action({"delta_x": 0.5, "delta_y": -0.2, "delta_z": 0.1, "gripper": 2.0})
    assert action["enabled"] is True
    assert action["target_x"] == 0.5
    assert action["target_y"] == -0.2
    assert action["target_z"] == 0.1
    assert action["gripper_vel"] == 2.0


def test_map_delta_action_without_gripper_defaults_to_stay():
    """use_gripper=False teleops (e.g. SpaceMouseSoarmHILTeleop) omit the gripper key.

    The gripper velocity must default to 1.0 ("stay") so the gripper is frozen at
    its current position instead of raising a KeyError.
    """
    step = MapDeltaActionToRobotActionStep()
    action = step.action({"delta_x": 0.5, "delta_y": 0.0, "delta_z": -0.3})
    assert action["gripper_vel"] == 1.0


def test_map_delta_action_enabled_flag():
    step = MapDeltaActionToRobotActionStep()
    action = step.action({"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0})
    assert action["enabled"] is False
