# 实时print spacemouse的6维位姿数据
import signal
import sys
import time

import pyspacemouse
from pyspacemouse.config_helpers import create_device_info

DEVICE_NAME = "SpaceMouseCompact"
VENDOR_ID = 0x256F
PRODUCT_ID = 0xC635

CORRECTED_MAPPINGS = {
    "x":     (1, 1, 2,  1),
    "y":     (1, 3, 4, -1),
    "z":     (1, 5, 6, -1),
    "roll":  (2, 1, 2,  1),
    "pitch": (2, 3, 4, -1),
    "yaw":   (2, 5, 6, -1),
}

RUNNING = True


def _signal_handler(sig, frame):
    global RUNNING
    RUNNING = False
    print("\nShutting down...", flush=True)


def print_spacemouse_data():
    signal.signal(signal.SIGINT, _signal_handler)

    found = pyspacemouse.get_connected_devices()
    if not found:
        print("No SpaceMouse device found. Is it connected?")
        sys.exit(1)
    print(f"  Found: {found[0]}", flush=True)

    device_spec = create_device_info(
        name=DEVICE_NAME,
        vendor_id=VENDOR_ID,
        product_id=PRODUCT_ID,
        mappings=CORRECTED_MAPPINGS,
        buttons={
            "LEFT": (3, 1, 0),
            "RIGHT": (3, 1, 1),
        },
    )

    with pyspacemouse.open(device_spec=device_spec, nonblocking=True) as device:
        print(
            "Connected. 位姿增量: X/Y/Z 平移, Roll/Pitch/Yaw 旋转 (右手定则)\n"
            "Move the SpaceMouse or press Ctrl+C to exit.\n",
            flush=True,
        )
        last_print_time = 0.0
        while RUNNING:
            state = device.read()
            if state is not None:
                now = time.perf_counter()
                if now - last_print_time >= 1 / 60.0:
                    last_print_time = now
                    print(
                        f"X: {state.x: 7.3f}  Y: {state.y: 7.3f}  Z: {state.z: 7.3f}  "
                        f"Roll: {state.roll: 7.3f}  Pitch: {state.pitch: 7.3f}  Yaw: {state.yaw: 7.3f}  "
                        f"LEFT: {int(state.buttons[0]) if state.buttons is not None else 0}  "
                        f"RIGHT: {int(state.buttons[1]) if state.buttons is not None else 0}",
                        flush=True,
                    )


if __name__ == "__main__":
    print_spacemouse_data()
