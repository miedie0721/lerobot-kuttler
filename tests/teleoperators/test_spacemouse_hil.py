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

from lerobot.processor.hil_processor import _check_teleop_with_events
from lerobot.teleoperators.spacemouse import SpaceMouseSoarmHILTeleop, SpaceMouseSoarmHILTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents, make_teleoperator_from_config


def test_factory_creates_hil_teleop():
    teleop = make_teleoperator_from_config(SpaceMouseSoarmHILTeleopConfig())
    assert isinstance(teleop, SpaceMouseSoarmHILTeleop)
    assert not teleop.needs_feedback
    assert teleop.is_connected is False


def test_action_features_match_delta_protocol():
    teleop = SpaceMouseSoarmHILTeleop(SpaceMouseSoarmHILTeleopConfig())
    features = teleop.action_features
    assert features["dtype"] == "float32"
    assert features["shape"] == (3,)
    assert features["names"] == {"delta_x": 0, "delta_y": 1, "delta_z": 2}


def test_get_teleop_events_when_disconnected():
    teleop = SpaceMouseSoarmHILTeleop(SpaceMouseSoarmHILTeleopConfig())
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.IS_INTERVENTION] is False
    assert events[TeleopEvents.SUCCESS] is False
    assert events[TeleopEvents.TERMINATE_EPISODE] is False
    assert events[TeleopEvents.RERECORD_EPISODE] is False


def test_teleop_satisfies_hil_events_protocol():
    teleop = SpaceMouseSoarmHILTeleop(SpaceMouseSoarmHILTeleopConfig())
    _check_teleop_with_events(teleop)
