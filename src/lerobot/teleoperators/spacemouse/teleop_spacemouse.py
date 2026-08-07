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
"""SpaceMouse end-effector teleoperator for SOARM (SO101) arm.

Provides continuous 6-DOF end-effector control using a 3Dconnexion SpaceMouse.
The SpaceMouse's velocity-like inputs are integrated into a target end-effector
pose each frame, and inverse kinematics produces joint angle actions for the
SO101 follower robot.

Uses the pyspacemouse library with a custom device spec that corrects the
axis mapping for right-hand-rule convention:
    X:     end-effector forward/backward translation
    Y:     base rotation (shoulder_pan)
    Z:     end-effector up/down translation
    Roll:  end-effector rotation about X
    Pitch: end-effector rotation about Y
    Yaw:   end-effector rotation about Z
    Left button:  reset to home position
    Right button: toggle gripper open/close
"""

import contextlib
import logging
import threading
import time

import numpy as np

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..keyboard.kinematics_soarm import lerobot_FK, lerobot_IK
from ..teleoperator import Teleoperator
from .configuration_spacemouse import SpaceMouseSoarmEETeleopConfig

try:
    import pyspacemouse
    from pyspacemouse.config_helpers import create_device_info

    _pyspacemouse_available = True
except ImportError:
    _pyspacemouse_available = False

logger = logging.getLogger(__name__)

# Custom device spec: corrected axis mapping for right-hand-rule convention.
# The default pyspacemouse mapping has roll/pitch swapped and yaw sign wrong.
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

# Workspace limits for target end-effector pose (matching lerobot-kinematics).
_CONTROL_GLIMIT = np.array(
    [[0.125, -0.4, 0.046, -3.1, -0.75, -1.5],
     [0.340,  0.4, 0.23,   2.0,  1.57,  1.5]],
    dtype=np.float64,
)


class SpaceMouseSoarmEETeleop(Teleoperator):
    """SpaceMouse end-effector teleoperator for SOARM (SO101) arm."""

    config_class = SpaceMouseSoarmEETeleopConfig
    name = "spacemouse_soarm_ee"

    def __init__(self, config: SpaceMouseSoarmEETeleopConfig):
        super().__init__(config)
        self.config = config
        self.translation_speed = config.translation_speed
        self.rotation_speed = config.rotation_speed
        self.gripper_increment = config.gripper_increment
        self.max_joint_change = config.max_joint_change

        self._device = None
        self._initialized = False
        self._init_joint_received = False
        self.needs_feedback = True
        self.logs = {}

        self.lock = threading.Lock()

        # Internal state (radians for joints 0-4, motor range for gripper at 5)
        self.init_qpos_home = np.array(
            [0.0, -1.57, 1.57, 0.0, -1.57, 1.0],
            dtype=np.float64,
        )
        self.current_qpos = self.init_qpos_home.copy()
        self.target_qpos = self.init_qpos_home.copy()
        self.target_gpos = np.zeros(6, dtype=np.float64)
        self.target_gpos_last = np.zeros(6, dtype=np.float64)

        self._prev_button_left = False
        self._prev_button_right = False

        # Track real wall-clock time between get_action calls for dt computation.
        self._last_action_time = 0.0

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
        if not _pyspacemouse_available:
            require_package("pyspacemouse", extra="spacemouse")

        found = pyspacemouse.get_connected_devices()
        if not found:
            raise RuntimeError("No SpaceMouse device found. Is it connected?")

        self._device_spec = create_device_info(
            name=_SPACEMOUSE_DEVICE_NAME,
            vendor_id=_SPACEMOUSE_VENDOR_ID,
            product_id=_SPACEMOUSE_PRODUCT_ID,
            mappings=_SPACEMOUSE_CORRECTED_MAPPINGS,
            buttons=_SPACEMOUSE_BUTTONS,
        )
        self._device = pyspacemouse.open(
            device_spec=self._device_spec, nonblocking=True
        )
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

    def _lazy_init_from_feedback(self, obs: dict) -> None:
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
        arm_q = self.current_qpos[1:5].copy()
        self.target_gpos = lerobot_FK(arm_q)
        self.target_gpos_last = self.target_gpos.copy()
        self.target_qpos = self.current_qpos.copy()

    def _initialize(self, obs: dict | None = None) -> None:
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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        now = time.perf_counter()

        if not self._initialized:
            self._initialize()

        # Compute real dt from wall-clock time between consecutive get_action calls.
        loop_dt = now - self._last_action_time if self._last_action_time > 0 else 1.0 / 60.0
        self._last_action_time = now
        # Clamp to prevent integration spikes after pauses.
        loop_dt = min(loop_dt, 0.1)

        state = self._device.read()

        if state is not None:
            # Y axis → shoulder_pan base rotation (negative: left push = left rotation)
            self.target_qpos[0] -= float(state.y) * self.rotation_speed * loop_dt
            # X/Z/Roll/Pitch/Yaw → target end-effector pose
            self.target_gpos[0] += float(state.x) * self.translation_speed * loop_dt
            self.target_gpos[2] += float(state.z) * self.translation_speed * loop_dt
            self.target_gpos[3] -= float(state.roll) * self.rotation_speed * loop_dt
            self.target_gpos[4] += float(state.pitch) * self.rotation_speed * loop_dt
            # Yaw is intentionally ignored: SO101 is a 5-DOF arm with no yaw joint.

            self.target_gpos = np.clip(
                self.target_gpos, _CONTROL_GLIMIT[0], _CONTROL_GLIMIT[1]
            )

            if state.buttons is not None and len(state.buttons) >= 2:
                btn_left = bool(state.buttons[0])
                btn_right = bool(state.buttons[1])
                # LEFT button: reset to home position
                if btn_left and not self._prev_button_left:
                    self.target_qpos = self.init_qpos_home.copy()
                    self.target_gpos = lerobot_FK(self.init_qpos_home[1:5])
                    self.target_gpos_last = self.target_gpos.copy()
                # RIGHT button: toggle gripper
                if btn_right and not self._prev_button_right:
                    if self.target_qpos[5] > 50:
                        self.target_qpos[5] = 0.0
                    else:
                        self.target_qpos[5] = 100.0
                self._prev_button_left = btn_left
                self._prev_button_right = btn_right

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

        self.logs["read_pos_dt_s"] = time.perf_counter() - now

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
                self.current_qpos[0] = np.deg2rad(
                    feedback.get("shoulder_pan.pos", np.rad2deg(self.current_qpos[0]))
                )
                self.current_qpos[5] = feedback.get("gripper.pos", self.current_qpos[5])
        if not self._initialized:
            self._initialize(feedback)

    def disconnect(self) -> None:
        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.close()
            self._device = None
        self._initialized = False
        self._init_joint_received = False
