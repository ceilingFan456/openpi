import sys
import time
import tty
import termios
import select
import numpy as np

from franka_interface import FrankaInterface, MockRobot

# --- CONFIG ---
USE_MOCK = True          # True = no hardware required
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

def get_key(fd):
    """Reads a key if available, otherwise returns None."""
    # check if stdin has data waiting (timeout=0 makes it non-blocking)
    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
        return sys.stdin.read(1)
    return None

def print_state(robot):
    try:
        joints = robot.get_joint_positions()
        ee_pose = robot.get_ee_pose()
        gripper = robot.get_gripper_position()

        # Using \r\n explicitly to ensure the cursor returns to the start of the line
        print("\r\n--- STATE ---", flush=True)
        print(f"Joints : {np.array2string(np.asarray(joints), precision=4, separator=', ')}\r", flush=True)
        print(f"EE xyz : {np.asarray(ee_pose)[:3]}\r", flush=True)
        print(f"Gripper: {float(np.asarray(gripper)[0]):.4f} m\r", flush=True)
        print("-----------\r", flush=True)
    except Exception as e:
        print(f"\r\n[ERROR] Could not read robot state: {e}\r", flush=True)

def main():
    if not sys.stdin.isatty():
        print("[console] ERROR: stdin is not a TTY. Use `ssh -t user@host`.", flush=True)
        sys.exit(1)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    robot = MockRobot() if USE_MOCK else FrankaInterface(ip=IP, port=PORT)
    
    print("[console] testing RPC/read...", flush=True)
    print_state(robot)
    print(HELP, flush=True)

    auto = False
    last_print_time = 0.0
    dt = 1.0 / max(1, POLL_HZ)

    try:
        # Set terminal to raw mode once
        tty.setraw(sys.stdin.fileno())
        
        while True:
            current_time = time.time()

            # 1. Handle Auto-Printing
            if auto and (current_time - last_print_time) >= dt:
                # Temporarily return to cooked mode to print cleanly, or just print
                # Printing in raw mode can sometimes mess up newlines (\r\n)
                print_state(robot)
                last_print_time = current_time

            # 2. Check for Keyboard Input
            key = get_key(fd)

            if key:
                if key in ("q", "\x03"): # 'q' or Ctrl+C
                    break
                elif key == "h":
                    print(HELP, flush=True)
                elif key == "p":
                    print_state(robot)

                ## move ee in the x direction.
                elif key == "w":
                    robot.set
                elif key == "a":
                    auto = not auto
                    print(f"\r\n[console] auto={auto}\r\n", flush=True)

            # 3. Prevent CPU hogging
            time.sleep(0.01)

    finally:
        # ALWAYS restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        try:
            robot.close()
        except:
            pass
        print("\n[console] exit.", flush=True)

if __name__ == "__main__":
    main()