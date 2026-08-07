#!/usr/bin/env python
"""Kinematics for SO101 robot arm, ported from lerobot-kinematics.

Provides forward kinematics (FK) and inverse kinematics (IK) for the
4-joint SO101 arm (shoulder pitch, elbow, wrist pitch, wrist roll),
using only numpy and scipy.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# 4x4 homogeneous transformation primitives
# ---------------------------------------------------------------------------

def _Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[1, 1] = c
    T[1, 2] = -s
    T[2, 1] = s
    T[2, 2] = c
    return T


def _Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 2] = s
    T[2, 0] = -s
    T[2, 2] = c
    return T


def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4, dtype=np.float64)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    return T


def _translate_x(d: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = d
    return T


def _translate_y(d: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[1, 3] = d
    return T


def _translate_z(d: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[2, 3] = d
    return T


# ---------------------------------------------------------------------------
# SO101 kinematics chain definition (from lerobot-kinematics create_so101)
#
# Each entry: ("t" | "r",  axis,  value-or-joint-index)
#   "t" / "r" = translation / revolute
#   axis      = "x" | "y" | "z"
#   For fixed transforms: value = translation (m) or rotation (rad)
#   For revolute joints:   value = joint index (0-3)
# ---------------------------------------------------------------------------

SO101_CHAIN: list[tuple[str, str, int | float]] = [
    ("t", "x", 0.02943),
    ("t", "z", 0.05504),
    ("r", "y", 0),        # joint 0: shoulder pitch
    ("t", "z", 0.1127),
    ("t", "x", 0.02798),
    ("r", "y", 1),        # joint 1: elbow
    ("t", "x", 0.13504),
    ("t", "z", 0.00519),
    ("r", "y", 2),        # joint 2: wrist pitch
    ("t", "x", 0.0593),
    ("t", "z", 0.00996),
    ("r", "x", 3),        # joint 3: wrist roll
]

SO101_NUM_JOINTS = 4

SO101_QLIM = np.array(
    [
        [-1.57, -1.57, -1.5, -3.14158],  # lower bounds
        [1.57, 1.57, 1.5, 3.14158],      # upper bounds
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def _apply_transform(
    T: np.ndarray, elem_type: str, axis: str, value: float
) -> np.ndarray:
    """Apply a single elementary transform to T and return the new 4x4 matrix."""
    if elem_type == "t":
        if axis == "x":
            return T @ _translate_x(value)
        elif axis == "y":
            return T @ _translate_y(value)
        elif axis == "z":
            return T @ _translate_z(value)
        else:
            raise ValueError(f"Unknown translation axis: {axis}")
    elif elem_type == "r":
        if axis == "x":
            return T @ _Rx(value)
        elif axis == "y":
            return T @ _Ry(value)
        elif axis == "z":
            return T @ _Rz(value)
        else:
            raise ValueError(f"Unknown rotation axis: {axis}")
    else:
        raise ValueError(f"Unknown element type: {elem_type}")


def compute_fk(q: np.ndarray) -> np.ndarray:
    """Forward kinematics: joint angles -> 4x4 homogeneous transform.

    Args:
        q: (4,) joint angles in radians
            [shoulder_pitch, elbow, wrist_flex, wrist_roll].

    Returns:
        4x4 homogeneous transformation matrix of end-effector in base frame.
    """
    T = np.eye(4, dtype=np.float64)
    for elem_type, axis, val in SO101_CHAIN:
        if elem_type == "r":
            T = _apply_transform(T, elem_type, axis, float(q[int(val)]))
        else:
            T = _apply_transform(T, elem_type, axis, float(val))
    return T


def lerobot_FK(q: np.ndarray) -> np.ndarray:
    """Forward kinematics returning end-effector pose as [x,y,z,roll,pitch,yaw].

    Uses XYZ intrinsic Euler convention for orientation, matching the
    convention used by lerobot_IK. This ensures FK→IK round-trip
    consistency: the rotation matrix reconstructed by IK from the FK
    output matches the original forward kinematics matrix.

    All values are rounded to 3 decimal places to match lerobot-kinematics.

    Args:
        q: (4,) joint angles in radians.

    Returns:
        (6,) array: [X, Y, Z, roll, pitch, yaw] rounded to 3 decimals.
    """
    T = compute_fk(q)
    x = round(float(T[0, 3]), 3)
    y = round(float(T[1, 3]), 3)
    z = round(float(T[2, 3]), 3)
    rot = R.from_matrix(T[:3, :3])
    roll, pitch, yaw = rot.as_euler("xyz")
    roll = round(float(roll), 3)
    pitch = round(float(pitch), 3)
    yaw = round(float(yaw), 3)
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# 6D angle-axis error between two poses
# ---------------------------------------------------------------------------


def angle_axis_error(T_curr: np.ndarray, T_target: np.ndarray) -> np.ndarray:
    """Compute 6D error vector between current and target poses.

    First 3 elements: position error (target - current).
    Last 3 elements: angle-axis rotation error in the base frame.

    Args:
        T_curr: 4x4 current end-effector transform.
        T_target: 4x4 target end-effector transform.

    Returns:
        (6,) error vector.
    """
    e = np.empty(6, dtype=np.float64)
    e[:3] = T_target[:3, 3] - T_curr[:3, 3]

    R_rel = T_target[:3, :3] @ T_curr[:3, :3].T
    li = np.array(
        [
            R_rel[2, 1] - R_rel[1, 2],
            R_rel[0, 2] - R_rel[2, 0],
            R_rel[1, 0] - R_rel[0, 1],
        ]
    )
    li_norm = float(np.linalg.norm(li))

    if li_norm < 1e-10:
        a = np.zeros(3) if np.trace(R_rel) > 0 else np.pi / 2.0 * (np.diag(R_rel) + 1.0)
    else:
        theta = np.arctan2(li_norm, np.trace(R_rel) - 1.0)
        a = theta * li / li_norm

    e[3:] = a
    return e


# ---------------------------------------------------------------------------
# Numerical Jacobian
# ---------------------------------------------------------------------------


def numerical_jacobian(q: np.ndarray, T_curr: np.ndarray) -> np.ndarray:
    """Compute 6x4 geometric Jacobian numerically.

    Uses finite-difference with epsilon = 1e-5 to compute the spatial
    velocity of the end-effector with respect to each joint.

    Args:
        q: (4,) current joint angles in radians.
        T_curr: 4x4 current end-effector transform (from compute_fk(q)).

    Returns:
        (6, 4) geometric Jacobian matrix (base frame).
    """
    eps = 1e-5
    J = np.zeros((6, SO101_NUM_JOINTS), dtype=np.float64)
    p_curr = T_curr[:3, 3].copy()

    for j in range(SO101_NUM_JOINTS):
        q_plus = q.copy()
        q_plus[j] += eps
        T_plus = compute_fk(q_plus)
        p_plus = T_plus[:3, 3]

        J[:3, j] = (p_plus - p_curr) / eps

        R_plus = T_plus[:3, :3] @ T_curr[:3, :3].T
        omega = np.array(
            [
                R_plus[2, 1] - R_plus[1, 2],
                R_plus[0, 2] - R_plus[2, 0],
                R_plus[1, 0] - R_plus[0, 1],
            ]
        ) / 2.0

        J[3:, j] = omega / eps

    return J


# ---------------------------------------------------------------------------
# Levenberg-Marquardt IK solver
# ---------------------------------------------------------------------------


def lerobot_IK(
    q_now: np.ndarray,
    target_pose: np.ndarray,
    ilimit: int = 10,
    slimit: int = 2,
    tol: float = 1e-3,
    max_joint_change: float = 0.1,
) -> tuple[np.ndarray, bool]:
    """Inverse kinematics using Levenberg-Marquardt with Chan damping.

    Given a target end-effector pose [x,y,z,roll,pitch,yaw] (XYZ Euler),
    solve for joint angles using LM numerical optimization.

    Args:
        q_now: (4,) current joint angles (radians) - used as initial guess.
        target_pose: (6,) target end-effector pose [x,y,z,roll,pitch,yaw]
                     where roll,pitch,yaw are XYZ intrinsic Euler angles.
        ilimit: Max iterations per search attempt.
        slimit: Max search attempts (random restarts on failure).
        tol: Convergence tolerance on scalar error E.
        max_joint_change: Max per-joint change per step (radians).

    Returns:
        (q_result, success): q_result is (4,) joint angles, success is bool.
    """
    x, y, z, roll, pitch, yaw = target_pose
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    R_mat = r.as_matrix()
    T_target = np.eye(4, dtype=np.float64)
    T_target[:3, :3] = R_mat
    T_target[:3, 3] = [x, y, z]

    best_q = None
    best_E = float("inf")

    rng = np.random.default_rng()

    for search_i in range(slimit):
        if search_i == 0:
            q = q_now.astype(np.float64).copy()
        else:
            q = rng.uniform(SO101_QLIM[0], SO101_QLIM[1]).astype(np.float64)

        for iteration_i in range(ilimit):
            T_curr = compute_fk(q)
            e = angle_axis_error(T_curr, T_target)
            E = 0.5 * float(np.sum(e**2))

            if iteration_i > 0 and tol > E:
                q = _smooth_joint_motion(q_now, q, max_joint_change)
                return q, True

            if best_E > E:
                best_E = E
                best_q = q.copy()

            J = numerical_jacobian(q, T_curr)
            We = np.eye(6, dtype=np.float64)
            Wn = E * np.eye(SO101_NUM_JOINTS, dtype=np.float64)

            JT_We_J = J.T @ We @ J
            JT_We_e = J.T @ We @ e

            try:
                delta_q = np.linalg.solve(
                    JT_We_J + Wn + 1e-6 * np.eye(SO101_NUM_JOINTS), JT_We_e
                )
            except np.linalg.LinAlgError:
                break

            q += delta_q

            q = np.clip(q, SO101_QLIM[0], SO101_QLIM[1])

    if best_q is not None and best_E < tol:
        best_q = _smooth_joint_motion(q_now, best_q, max_joint_change)
        q = np.clip(best_q, SO101_QLIM[0], SO101_QLIM[1])
        return q, True

    return -np.ones(SO101_NUM_JOINTS, dtype=np.float64), False


def _smooth_joint_motion(
    q_now: np.ndarray, q_new: np.ndarray, max_change: float = 0.1
) -> np.ndarray:
    """Clamp per-joint change so no joint moves more than max_change per step."""
    q_smooth = q_new.copy()
    for i in range(len(q_smooth)):
        delta = q_smooth[i] - q_now[i]
        if abs(delta) > max_change:
            delta = np.sign(delta) * max_change
        q_smooth[i] = q_now[i] + delta
    return q_smooth
