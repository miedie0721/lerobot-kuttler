# 当前机械臂实际 state（弧度，夹爪为 0-100；角度换算：pan=-2.24° 等）：
# {'shoulder_pan.pos': -0.0391, 'shoulder_lift.pos': -0.7871, 'elbow_flex.pos': 1.1400,
#  'wrist_flex.pos': 0.8216, 'wrist_roll.pos': -1.4139, 'gripper.pos': 5.66}

import time

import numpy as np

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig

# SO101 的 5 个机械臂关节（夹爪使用 0-100 范围，不参与弧度转换）
_ARM_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]


def _deg_to_rad_state(state: dict[str, float]) -> dict[str, float]:
    """把 from动臂观测（角度值，度）转换为弧度；夹爪 0-100 不转换。"""
    return {
        key: (np.deg2rad(value) if key in _ARM_JOINT_KEYS else value)
        for key, value in state.items()
    }


def _rad_to_deg_state(state: dict[str, float]) -> dict[str, float]:
    """把弧度 state 转换为度数（供 send_action 使用，夹爪 0-100 不转换）。"""
    return {
        key: (np.rad2deg(value) if key in _ARM_JOINT_KEYS else value)
        for key, value in state.items()
    }


def get_soarm_state(robot_port: str, robot_id: str) -> dict[str, float]:
    """读取 SO101 当前关节角，返回弧度（夹爪为 0-100）。"""
    robot = make_robot_from_config(SO101FollowerConfig(port=robot_port, id=robot_id))
    robot.connect()
    try:
        observation = robot.get_observation()
        return _deg_to_rad_state(
            {key: value for key, value in observation.items() if key.endswith(".pos")}
        )
    finally:
        robot.disconnect()


def move_soarm_to_state(
    state: dict[str, float],
    robot_port: str = "/dev/ttyACM0",
    robot_id: str = "kuttler_soarm",
    duration: float = 2.0,
    hold_duration: float = 3.0,
    fps: int = 60,
) -> None:
    """把 SO101 机械臂平滑移动到指定 state，并保持一段时间。

    参数 state 的关节角单位为弧度（与 get_soarm_state 输出一致），夹爪为 0-100；
    缺省的关节保持当前值。机械臂按线性插值在 duration 秒内逐步逼近目标，
    到位后继续以同样频率持续发送目标指令 hold_duration 秒，以保持该 state。
    内部将弧度转换为度数后发送给从动臂。
    """
    robot = make_robot_from_config(SO101FollowerConfig(port=robot_port, id=robot_id))
    robot.connect()
    try:
        observation = robot.get_observation()
        current = _deg_to_rad_state(
            {key: value for key, value in observation.items() if key.endswith(".pos")}
        )
        # 目标 state 补全缺失关节（缺省保持当前值）
        target = dict(current)
        for key, value in state.items():
            if key in current:
                target[key] = value

        current_deg = _rad_to_deg_state(current)
        target_deg = _rad_to_deg_state(target)

        # 阶段一：线性插值移动到目标 state
        n_steps = max(int(duration * fps), 1)
        for i in range(1, n_steps + 1):
            frame_start = time.perf_counter()
            t = i / n_steps
            goal = {
                key: current_deg[key] + (target_deg[key] - current_deg[key]) * t
                for key in current_deg
            }
            robot.send_action({key: float(value) for key, value in goal.items()})
            # 精确限频：睡满剩余时间，保证帧率正好为 fps（发送耗时不计入）
            time.sleep(max(1 / fps - (time.perf_counter() - frame_start), 0.0))

        # 阶段二：保持目标 state（持续发送指令，防止机械臂松手）
        n_hold = max(int(hold_duration * fps), 1)
        for _ in range(n_hold):
            frame_start = time.perf_counter()
            robot.send_action({key: float(value) for key, value in target_deg.items()})
            time.sleep(max(1 / fps - (time.perf_counter() - frame_start), 0.0))
    finally:
        robot.disconnect()


# 当前机械臂实际 state（弧度），2026-07-24 读取自 /dev/ttyACM0
INIT_STATE = {
    "shoulder_pan.pos": -0.054,
    "shoulder_lift.pos": -0.758,
    "elbow_flex.pos": 0.950,
    "wrist_flex.pos": 0.8216,
    "wrist_roll.pos": -1.504,
    "gripper.pos": 5.66,
}


if __name__ == "__main__":
    # 读取当前 state（弧度），print 保留 3 位小数
    state = get_soarm_state(robot_port="/dev/ttyACM0", robot_id="kuttler_soarm")
    print({key: f"{value:.3f}" for key, value in state.items()})

    # 移动到推荐初始 state（弧度），平滑过渡，用时 2 秒
    move_soarm_to_state(INIT_STATE, robot_port="/dev/ttyACM0", robot_id="kuttler_soarm", duration=2.0)
    print(get_soarm_state(robot_port="/dev/ttyACM0", robot_id="kuttler_soarm"))
