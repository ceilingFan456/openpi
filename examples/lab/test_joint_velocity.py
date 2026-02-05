import time
import numpy as np

from franka_interface import FrankaInterface  # adjust import if needed

## using this script to test the joints velocity control interface. 

def main():
    robot = FrankaInterface(ip="192.168.1.112", port=4242)

    try:
        print("Current joint positions:")
        print(robot.get_joint_positions())

        # 7D joint velocity vector
        vel = np.zeros(7)

        # Rotate joint 0 slowly (rad/s)
        vel[0] = 0.1   # VERY conservative, safe value
        # vel[1] = 0.02
        # vel[2] = -0.03

        print("Starting joint velocity control...")
        robot.start_joint_velocity_control(vel)

        # Run for 3 seconds, keep updating command
        start = time.time()
        while time.time() - start < 3.0:
            robot.update_desired_joint_velocities(vel)
            time.sleep(0.02)  # ~50 Hz

        print("Stopping...")
        robot.update_desired_joint_velocities(np.zeros(7))
        time.sleep(0.5)

    finally:
        robot.terminate_current_policy()
        robot.close()
        print("Done.")

if __name__ == "__main__":
    main()
