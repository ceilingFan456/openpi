#!/usr/bin/env python3
"""
Dual Franka arm ZeroRPC server for bimanual policy execution.

This script runs on the Franka host PC (NUC) and:
1. Directly controls two Franka arms via Polymetis RobotInterface
2. Hosts a ZeroRPC server exposing both arms via separate namespaces
3. Allows remote clients (e.g., workstation) to control both robots simultaneously

Compatible with franka_interface.py client (connect to different ports for each arm).
"""

import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import argparse
from typing import Optional


class DualArmServer:
    """ZeroRPC server for dual Franka arm control."""
    
    def __init__(self, 
                 left_robot_port=50051,
                 left_gripper_port=50052,
                 right_robot_port=50054,
                 right_gripper_port=50055):
        """
        Initialize dual arm server.
        
        Args:
            left_robot_port: Polymetis port for left robot arm
            left_gripper_port: Polymetis port for left gripper
            right_robot_port: Polymetis port for right robot arm
            right_gripper_port: Polymetis port for right gripper
        """
        print(f"Connecting to left robot on port {left_robot_port}...")
        self.left_robot = RobotInterface(ip_address='localhost', port=left_robot_port)
        
        print(f"Connecting to left gripper on port {left_gripper_port}...")
        self.left_gripper = GripperInterface(ip_address='localhost', port=left_gripper_port)
        
        print(f"Connecting to right robot on port {right_robot_port}...")
        self.right_robot = RobotInterface(ip_address='localhost', port=right_robot_port)
        
        print(f"Connecting to right gripper on port {right_gripper_port}...")
        self.right_gripper = GripperInterface(ip_address='localhost', port=right_gripper_port)
        
        print(f"Left robot initialized. EE pose: {self._format_ee_pose(self.left_robot.get_ee_pose())}")
        print(f"Left gripper width: {self.left_gripper.get_state().width:.4f}m")
        print(f"Right robot initialized. EE pose: {self._format_ee_pose(self.right_robot.get_ee_pose())}")
        print(f"Right gripper width: {self.right_gripper.get_state().width:.4f}m")
    
    def _format_ee_pose(self, pose_data):
        """Convert Polymetis pose to [x, y, z, qx, qy, qz, qw] format."""
        pos = pose_data[0].numpy()
        quat_xyzw = pose_data[1].numpy()
        return np.concatenate([pos, quat_xyzw])
    
    # ========= LEFT ARM METHODS =========
    
    def left_get_ee_pose(self):
        """Get left arm end-effector pose [x, y, z, qx, qy, qz, qw]."""
        return self._format_ee_pose(self.left_robot.get_ee_pose()).tolist()
    
    def left_get_joint_positions(self):
        """Get left arm joint positions (7D)."""
        return self.left_robot.get_joint_positions().numpy().tolist()
    
    def left_get_joint_velocities(self):
        """Get left arm joint velocities (7D)."""
        return self.left_robot.get_joint_velocities().numpy().tolist()
    
    def left_get_force_torque(self):
        """Get left arm force/torque readings (6D)."""
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def left_get_gripper_position(self):
        """Get left gripper position."""
        width = self.left_gripper.get_state().width
        return [width]
    
    def left_set_gripper_position(self, pos):
        """Set left gripper position."""
        pos = float(np.clip(pos, 0.0, 0.08))
        self.left_gripper.goto(width=pos, speed=0.1, force=20.0, blocking=False)
        return f"Left gripper moving to {pos:.4f}m"
    
    def left_control_gripper(self, gripper_action):
        """Control left gripper (True=close, False=open)."""
        if gripper_action:
            self.left_gripper.grasp(speed=0.1, force=20.0, blocking=False)
            return "Closing left gripper"
        else:
            self.left_gripper.goto(width=0.08, speed=0.1, force=20.0, blocking=False)
            return "Opening left gripper"
    
    def left_move_to_joint_positions(self, positions, time_to_go):
        """Move left arm to target joint positions."""
        positions = torch.Tensor(positions)
        self.left_robot.move_to_joint_positions(positions=positions, time_to_go=float(time_to_go))
        return f"Left arm moving to joint positions in {time_to_go}s"
    
    def left_start_cartesian_impedance(self, Kx, Kxd):
        """Start left arm Cartesian impedance controller."""
        Kx = torch.Tensor(Kx) if Kx is not None else None
        Kxd = torch.Tensor(Kxd) if Kxd is not None else None
        self.left_robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
        return "Left arm Cartesian impedance started"
    
    def left_start_joint_impedance(self, Kq, Kqd):
        """Start left arm joint impedance controller."""
        Kq = torch.Tensor(Kq) if Kq is not None else None
        Kqd = torch.Tensor(Kqd) if Kqd is not None else None
        self.left_robot.start_joint_impedance(Kq=Kq, Kqd=Kqd)
        return "Left arm joint impedance started"
    
    def left_update_desired_ee_pose(self, pose):
        """Update left arm desired end-effector pose."""
        pose = np.array(pose)
        position = torch.Tensor(pose[:3])
        orientation = torch.Tensor(pose[3:7])  # quaternion in xyzw format
        self.left_robot.update_desired_ee_pose(position=position, orientation=orientation)
        return "Left arm desired EE pose updated"
    
    def left_update_desired_joint_pos(self, pos):
        """Update left arm desired joint positions."""
        pos = torch.Tensor(pos)
        self.left_robot.update_desired_joint_positions(pos)
        return "Left arm desired joint positions updated"
    
    def left_terminate_current_policy(self):
        """Terminate left arm current policy."""
        self.left_robot.terminate_current_policy()
        return "Left arm policy terminated"
    
    # ========= RIGHT ARM METHODS =========
    
    def right_get_ee_pose(self):
        """Get right arm end-effector pose [x, y, z, qx, qy, qz, qw]."""
        return self._format_ee_pose(self.right_robot.get_ee_pose()).tolist()
    
    def right_get_joint_positions(self):
        """Get right arm joint positions (7D)."""
        return self.right_robot.get_joint_positions().numpy().tolist()
    
    def right_get_joint_velocities(self):
        """Get right arm joint velocities (7D)."""
        return self.right_robot.get_joint_velocities().numpy().tolist()
    
    def right_get_force_torque(self):
        """Get right arm force/torque readings (6D)."""
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def right_get_gripper_position(self):
        """Get right gripper position."""
        width = self.right_gripper.get_state().width
        return [width]
    
    def right_set_gripper_position(self, pos):
        """Set right gripper position."""
        pos = float(np.clip(pos, 0.0, 0.08))
        self.right_gripper.goto(width=pos, speed=0.1, force=20.0, blocking=False)
        return f"Right gripper moving to {pos:.4f}m"
    
    def right_control_gripper(self, gripper_action):
        """Control right gripper (True=close, False=open)."""
        if gripper_action:
            self.right_gripper.grasp(speed=0.1, force=20.0, blocking=False)
            return "Closing right gripper"
        else:
            self.right_gripper.goto(width=0.08, speed=0.1, force=20.0, blocking=False)
            return "Opening right gripper"
    
    def right_move_to_joint_positions(self, positions, time_to_go):
        """Move right arm to target joint positions."""
        positions = torch.Tensor(positions)
        self.right_robot.move_to_joint_positions(positions=positions, time_to_go=float(time_to_go))
        return f"Right arm moving to joint positions in {time_to_go}s"
    
    def right_start_cartesian_impedance(self, Kx, Kxd):
        """Start right arm Cartesian impedance controller."""
        Kx = torch.Tensor(Kx) if Kx is not None else None
        Kxd = torch.Tensor(Kxd) if Kxd is not None else None
        self.right_robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
        return "Right arm Cartesian impedance started"
    
    def right_start_joint_impedance(self, Kq, Kqd):
        """Start right arm joint impedance controller."""
        Kq = torch.Tensor(Kq) if Kq is not None else None
        Kqd = torch.Tensor(Kqd) if Kqd is not None else None
        self.right_robot.start_joint_impedance(Kq=Kq, Kqd=Kqd)
        return "Right arm joint impedance started"
    
    def right_update_desired_ee_pose(self, pose):
        """Update right arm desired end-effector pose."""
        pose = np.array(pose)
        position = torch.Tensor(pose[:3])
        orientation = torch.Tensor(pose[3:7])  # quaternion in xyzw format
        self.right_robot.update_desired_ee_pose(position=position, orientation=orientation)
        return "Right arm desired EE pose updated"
    
    def right_update_desired_joint_pos(self, pos):
        """Update right arm desired joint positions."""
        pos = torch.Tensor(pos)
        self.right_robot.update_desired_joint_positions(pos)
        return "Right arm desired joint positions updated"
    
    def right_terminate_current_policy(self):
        """Terminate right arm current policy."""
        self.right_robot.terminate_current_policy()
        return "Right arm policy terminated"
    
    # ========= BIMANUAL METHODS =========
    
    def get_both_ee_poses(self):
        """Get both arms' end-effector poses."""
        return {
            'left': self.left_get_ee_pose(),
            'right': self.right_get_ee_pose()
        }
    
    def get_both_joint_positions(self):
        """Get both arms' joint positions."""
        return {
            'left': self.left_get_joint_positions(),
            'right': self.right_get_joint_positions()
        }
    
    def get_both_joint_velocities(self):
        """Get both arms' joint velocities."""
        return {
            'left': self.left_get_joint_velocities(),
            'right': self.right_get_joint_velocities()
        }
    
    def get_both_gripper_positions(self):
        """Get both grippers' positions."""
        return {
            'left': self.left_get_gripper_position(),
            'right': self.right_get_gripper_position()
        }
    
    def terminate_all_policies(self):
        """Terminate policies on both arms."""
        self.left_robot.terminate_current_policy()
        self.right_robot.terminate_current_policy()
        return "All policies terminated"
    
    def close(self):
        """Close connections to both robots."""
        return "Connections closed"


class LeftArmProxy:
    """Proxy that exposes left arm with standard FrankaInterface API."""
    
    def __init__(self, dual_server):
        self.server = dual_server
    
    def get_ee_pose(self):
        return self.server.left_get_ee_pose()
    
    def get_joint_positions(self):
        return self.server.left_get_joint_positions()
    
    def get_joint_velocities(self):
        return self.server.left_get_joint_velocities()
    
    def get_force_torque(self):
        return self.server.left_get_force_torque()
    
    def get_gripper_position(self):
        return self.server.left_get_gripper_position()
    
    def set_gripper_position(self, pos):
        return self.server.left_set_gripper_position(pos)
    
    def control_gripper(self, gripper_action):
        return self.server.left_control_gripper(gripper_action)
    
    def move_to_joint_positions(self, positions, time_to_go):
        return self.server.left_move_to_joint_positions(positions, time_to_go)
    
    def start_cartesian_impedance(self, Kx, Kxd):
        return self.server.left_start_cartesian_impedance(Kx, Kxd)
    
    def start_joint_impedance(self, Kq, Kqd):
        return self.server.left_start_joint_impedance(Kq, Kqd)
    
    def update_desired_ee_pose(self, pose):
        return self.server.left_update_desired_ee_pose(pose)
    
    def update_desired_joint_pos(self, pos):
        return self.server.left_update_desired_joint_pos(pos)
    
    def terminate_current_policy(self):
        return self.server.left_terminate_current_policy()
    
    def close(self):
        return self.server.close()


class RightArmProxy:
    """Proxy that exposes right arm with standard FrankaInterface API."""
    
    def __init__(self, dual_server):
        self.server = dual_server
    
    def get_ee_pose(self):
        return self.server.right_get_ee_pose()
    
    def get_joint_positions(self):
        return self.server.right_get_joint_positions()
    
    def get_joint_velocities(self):
        return self.server.right_get_joint_velocities()
    
    def get_force_torque(self):
        return self.server.right_get_force_torque()
    
    def get_gripper_position(self):
        return self.server.right_get_gripper_position()
    
    def set_gripper_position(self, pos):
        return self.server.right_set_gripper_position(pos)
    
    def control_gripper(self, gripper_action):
        return self.server.right_control_gripper(gripper_action)
    
    def move_to_joint_positions(self, positions, time_to_go):
        return self.server.right_move_to_joint_positions(positions, time_to_go)
    
    def start_cartesian_impedance(self, Kx, Kxd):
        return self.server.right_start_cartesian_impedance(Kx, Kxd)
    
    def start_joint_impedance(self, Kq, Kqd):
        return self.server.right_start_joint_impedance(Kq, Kqd)
    
    def update_desired_ee_pose(self, pose):
        return self.server.right_update_desired_ee_pose(pose)
    
    def update_desired_joint_pos(self, pos):
        return self.server.right_update_desired_joint_pos(pos)
    
    def terminate_current_policy(self):
        return self.server.right_terminate_current_policy()
    
    def close(self):
        return self.server.close()


def main():
    parser = argparse.ArgumentParser(
        description='Dual Franka arm ZeroRPC server for bimanual policy execution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--left-robot-port', type=int, default=50051,
                        help='Polymetis port for left robot arm')
    parser.add_argument('--left-gripper-port', type=int, default=50052,
                        help='Polymetis port for left gripper')
    parser.add_argument('--right-robot-port', type=int, default=50054,
                        help='Polymetis port for right robot arm')
    parser.add_argument('--right-gripper-port', type=int, default=50055,
                        help='Polymetis port for right gripper')
    parser.add_argument('--left-zerorpc-port', type=int, default=4242,
                        help='ZeroRPC server port for left arm')
    parser.add_argument('--right-zerorpc-port', type=int, default=4243,
                        help='ZeroRPC server port for right arm')
    parser.add_argument('--dual-zerorpc-port', type=int, default=4244,
                        help='ZeroRPC server port for dual arm (bimanual methods)')
    parser.add_argument('--ip', type=str, default='0.0.0.0',
                        help='IP address to bind ZeroRPC servers (0.0.0.0 for all interfaces)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DUAL FRANKA ARM SERVER")
    print("=" * 70)
    print(f"Left robot port:   {args.left_robot_port}")
    print(f"Left gripper port: {args.left_gripper_port}")
    print(f"Right robot port:  {args.right_robot_port}")
    print(f"Right gripper port: {args.right_gripper_port}")
    print(f"Left ZeroRPC:      {args.ip}:{args.left_zerorpc_port}")
    print(f"Right ZeroRPC:     {args.ip}:{args.right_zerorpc_port}")
    print(f"Dual ZeroRPC:      {args.ip}:{args.dual_zerorpc_port}")
    print("=" * 70)
    
    # Create dual arm server
    dual_server = DualArmServer(
        left_robot_port=args.left_robot_port,
        left_gripper_port=args.left_gripper_port,
        right_robot_port=args.right_robot_port,
        right_gripper_port=args.right_gripper_port
    )
    
    # Create proxy servers for individual arms
    left_proxy = LeftArmProxy(dual_server)
    right_proxy = RightArmProxy(dual_server)
    
    print(f"\nStarting THREE ZeroRPC servers:")
    print(f"  1. Left arm:  {args.ip}:{args.left_zerorpc_port}  (FrankaInterface API)")
    print(f"  2. Right arm: {args.ip}:{args.right_zerorpc_port} (FrankaInterface API)")
    print(f"  3. Dual arm:  {args.ip}:{args.dual_zerorpc_port} (Bimanual methods)")
    print("\nLeft Arm API (port {}):".format(args.left_zerorpc_port))
    print("  get_ee_pose(), get_joint_positions(), get_joint_velocities()")
    print("  get_gripper_position(), set_gripper_position(), control_gripper()")
    print("  move_to_joint_positions(), start_joint_impedance(), etc.")
    print("\nRight Arm API (port {}):".format(args.right_zerorpc_port))
    print("  get_ee_pose(), get_joint_positions(), get_joint_velocities()")
    print("  get_gripper_position(), set_gripper_position(), control_gripper()")
    print("  move_to_joint_positions(), start_joint_impedance(), etc.")
    print("\nDual Arm API (port {}):".format(args.dual_zerorpc_port))
    print("  left_*, right_* (all methods prefixed)")
    print("  get_both_ee_poses(), get_both_joint_positions()")
    print("  terminate_all_policies()")
    print("\nServers ready. Press Ctrl+C to stop.\n")
    
    import threading
    
    # Start left arm server in thread
    def run_left_server():
        server = zerorpc.Server(left_proxy, heartbeat=20)
        server.bind(f"tcp://{args.ip}:{args.left_zerorpc_port}")
        server.run()
    
    # Start right arm server in thread
    def run_right_server():
        server = zerorpc.Server(right_proxy, heartbeat=20)
        server.bind(f"tcp://{args.ip}:{args.right_zerorpc_port}")
        server.run()
    
    # Start dual arm server in main thread
    def run_dual_server():
        server = zerorpc.Server(dual_server, heartbeat=20)
        server.bind(f"tcp://{args.ip}:{args.dual_zerorpc_port}")
        server.run()
    
    try:
        left_thread = threading.Thread(target=run_left_server, daemon=True)
        right_thread = threading.Thread(target=run_right_server, daemon=True)
        
        left_thread.start()
        right_thread.start()
        
        # Run dual server in main thread (blocks until Ctrl+C)
        run_dual_server()
        
    except KeyboardInterrupt:
        print("\n\nShutting down all servers...")
    finally:
        print("Servers stopped")


if __name__ == '__main__':
    main()
