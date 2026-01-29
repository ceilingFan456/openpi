import sys
import time
import tty
import termios
import numpy as np

from franka_interface import FrankaInterface, MockRobot

# --- CONFIG ---
USE_MOCK = False          # True = no hardware required
IP = "192.168.1.111"
PORT = 4242
POLL_HZ = 5               # for auto mode
# -------------

HELP = """
Keys:
  p    : print state once
  a    : toggle auto-print at POLL_HZ
  h    : help
  q    : quit
"""

def get_key(fd, old_settings):
    """SSH-friendly single key reader."""
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # possible arrows etc.
            ch += sys.stdin.read(2)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def print_state(robot):
    joints = robot.get_joint_positions()
    ee_pose = robot.get_ee_pose()
    gripper = robot.get_gripper_position()

    print("\n--- STATE ---", flush=True)
    print("Joints:", np.array2string(np.asarray(joints), precision=4, separator=", "), flush=True)
    print("EE xyz :", np.asarray(ee_pose)[:3], flush=True)
    print("Gripper:", float(np.asarray(gripper)[0]), "m", flush=True)
    print("-----------", flush=True)

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("[console] starting...", flush=True)
    print(f"[console] stdin isatty={sys.stdin.isatty()} stdout isatty={sys.stdout.isatty()}", flush=True)
    if not sys.stdin.isatty():
        print("[console] ERROR: stdin is not a TTY. Use `ssh -t user@host`.", flush=True)
        sys.exit(1)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    robot = MockRobot() if USE_MOCK else FrankaInterface(ip=IP, port=PORT)

    # Quick connectivity test (read-only)
    print("[console] testing RPC/read...", flush=True)
    print_state(robot)

    print(HELP, flush=True)

    auto = False
    last = 0.0
    dt = 1.0 / max(1, POLL_HZ)

    try:
        while True:
            if auto and (time.time() - last) >= dt:
                print_state(robot)
                last = time.time()

            # Non-blocking-ish: just block on key when not auto.
            if auto:
                # small sleep so we don't spin
                time.sleep(0.02)
                # try to read key only if input is ready (simple approach: skip)
                # If you want true non-blocking, we can add `select`.
                continue

            key = get_key(fd, old)

            if key in ("q", "\x03"):
                break
            elif key == "h":
                print(HELP, flush=True)
            elif key == "p":
                print_state(robot)
            elif key == "a":
                auto = not auto
                print(f"[console] auto={auto} (POLL_HZ={POLL_HZ})", flush=True)

    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass
        try:
            robot.close()
        except Exception:
            pass
        print("[console] exit.", flush=True)

if __name__ == "__main__":
    main()
