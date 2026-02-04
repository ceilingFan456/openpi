#!/usr/bin/env python3
"""
go_home_client.py

Simple ZeroRPC client to send the robot to a home joint configuration via the server's
move_to_joint_positions RPC.

Usage examples:
  # Use default host/port and default home joints
  python3 go_home_client.py

  # Specify host/port and time-to-go
  python3 go_home_client.py --host 127.0.0.1 --port 4242 --time 3.0

  # Provide a custom 7-element home using comma-separated values
  python3 go_home_client.py --home "0, -0.785398, 0, -2.356194, 0, 1.5708, 0.785398" --time 4.0
"""

import argparse
import zerorpc
import numpy as np
import sys


def parse_home_string(s: str):
    """Parse a comma-separated or Python-list-like string into a 7-element float list."""
    try:
        # allow both "1,2,3" and "[1,2,3]"
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            vals = eval(s)
        else:
            vals = [x.strip() for x in s.split(',') if x.strip() != '']
            vals = [float(x) for x in vals]
        return [float(x) for x in vals]
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid home format: {e}")


def main():
    parser = argparse.ArgumentParser(description="Send robot to home joint positions via ZeroRPC server")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='ZeroRPC server host')
    parser.add_argument('--port', type=int, default=4242, help='ZeroRPC server port')
    parser.add_argument('--time', type=float, default=3.0, help='Time to reach the home pose (seconds)')
    parser.add_argument('--home', type=parse_home_string, default=None,
                        help='Comma-separated 7 joint values or Python list, e.g. "0, -0.785, 0, -2.356, 0, 1.571, 0.785"')
    parser.add_argument('--timeout', type=float, default=5.0, help='Connection timeout (seconds)')

    args = parser.parse_args()

    # Sensible default home pose (7 joints) -- adjust for your robot if needed.
    # default_home = [0.019, -0.541, -0.019, -2.751, -0.027, 2.201, 0.820]  # cihai_realrob
    default_home = [0.0, -0.4058451, 0.0, -2.6068573, 0.0, 2.1711833, 0.86116207]
    # default_home = [-0.12658207, 0.22434965, -0.20203676, -2.06723024, 0.28471853, 2.09440176, -0.84372037]
    # default_home = [-0.22394875 , 0.4347141 , -0.07857539, -2.18918258 , 0.65292927 , 2.53198843, -1.27714176]

    home = args.home if args.home is not None else default_home

    if len(home) != 7:
        print(f"Error: home must be 7 joint values, got {len(home)}: {home}")
        sys.exit(2)

    print(f"Connecting to ZeroRPC server at tcp://{args.host}:{args.port} ...")
    client = zerorpc.Client(heartbeat=10, timeout=int(max(1, args.timeout)))
    try:
        client.connect(f"tcp://{args.host}:{args.port}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        sys.exit(1)

    try:
        print(f"Sending move_to_joint_positions with time_to_go={args.time}s and home={home}")
        resp = client.move_to_joint_positions(home, float(args.time))
        print("Server response:", resp)
        
        # Open the gripper
        print("Opening gripper...")
        gripper_resp = client.control_gripper(False)  # False = open
        print("Gripper response:", gripper_resp)
    except Exception as e:
        print("RPC call failed:", e)
        raise
    finally:
        try:
            client.close()
        except Exception:
            pass

    print("Command sent. Monitor robot state on the robot host.")


if __name__ == '__main__':
    main()
