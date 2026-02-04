#!/usr/bin/env python3
"""
Single Franka arm ZeroRPC server for policy execution.

This script runs on the Franka host PC (NUC) and:
1. Directly controls one Franka arm via Polymetis RobotInterface
2. Hosts a ZeroRPC server that exposes the FrankaInterface API
3. Allows remote clients (e.g., workstation) to control the robot

Compatible with franka_interface.py client.
"""

import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import argparse
from typing import Optional


class SingleArmServer:
    """ZeroRPC server for single Franka arm control."""
    
    def __init__(self, 
                 robot_port=50051,
                 gripper_port=50052):
        """
        Initialize single arm server.
        
        Args:
            robot_port: Polymetis port for robot arm
            gripper_port: Polymetis port for gripper
        """
        print(f"Connecting to robot on port {robot_port}...")
        self.robot = RobotInterface(ip_address='localhost', port=robot_port)
        
        print(f"Connecting to gripper on port {gripper_port}...")
        self.gripper = GripperInterface(ip_address='localhost', port=gripper_port)
        
        # Print current robot state at startup
        current_ee_pose = self._format_ee_pose(self.robot.get_ee_pose())
        current_joint_pos = self.robot.get_joint_positions().numpy()
        
        print(f"\nRobot initialized successfully!")
        print(f"Current EE pose [x, y, z, qx, qy, qz, qw]:")
        print(f"  {current_ee_pose}")
        print(f"Current joint positions [q1-q7] (radians):")
        print(f"  {current_joint_pos}")
        print(f"Gripper width: {self.gripper.get_state().width:.4f}m")
        print(f"Gripper state: {self.gripper.get_state()}")
    
    def _format_ee_pose(self, pose_data):
        """Convert Polymetis pose to [x, y, z, qx, qy, qz, qw] format."""
        pos = pose_data[0].numpy()
        quat_xyzw = pose_data[1].numpy()
        return np.concatenate([pos, quat_xyzw])
    
    # ========= State query methods (FrankaInterface API) =========
    
    def get_ee_pose(self):
        """Get end-effector pose [x, y, z, qx, qy, qz, qw]."""
        return self._format_ee_pose(self.robot.get_ee_pose()).tolist()
    
    def get_joint_positions(self):
        """Get current joint positions (7D)."""
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        """Get current joint velocities (7D)."""
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def get_force_torque(self):
        """Get current force/torque readings (6D)."""
        # Polymetis doesn't expose F/T directly, return zeros for now
        # If you have F/T sensor, implement this properly
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # ========= Gripper methods (FrankaInterface API) =========
    
    def get_gripper_position(self):
        """Get current gripper position (opening width in meters)."""
        width = self.gripper.get_state().width
        return [width]
    
    def get_gripper_prev_cmd_success(self):
        """Get whether previous gripper command succeeded."""
        return self.gripper.get_state().prev_command_successful
    
    def set_gripper_position(self, pos):
        """
        Set gripper to specific position.
        
        Args:
            pos: Gripper width in meters (0.0 = closed, ~0.08 = open)
        """
        pos = float(np.clip(pos, 0.0, 0.08))
        self.gripper.goto(width=pos, speed=0.1, force=20.0, blocking=False)
        return f"Gripper moving to {pos:.4f}m"
    
    def control_gripper(self, gripper_action):
        """
        Control gripper (binary: 0: open/ 1: close).
        
        Args:
            gripper_action: True to close, False to open
        """
        if gripper_action:
            # Close/grasp
            self.gripper.grasp(speed=0.5, force=20.0, blocking=False)

            print("Gripper grasp command!")
            return "Closing gripper"
        else:
            # Open
            self.gripper.goto(width=0.08, speed=0.5, force=20.0, blocking=False)
            # self.gripper.grasp(grasp_width=0.08, speed=0.15, force=20.0, blocking=False)
            print("Gripper open command!")
            return "Opening gripper"
    
    # ========= Motion control methods (FrankaInterface API) =========
    
    def move_to_joint_positions(self, positions, time_to_go):
        """
        Move to target joint positions.
        
        Args:
            positions: Target joint positions (7D list/array)
            time_to_go: Time to reach target (seconds)
        """
        positions = torch.Tensor(positions)
        self.robot.move_to_joint_positions(positions=positions, time_to_go=float(time_to_go))
        return f"Moving to joint positions in {time_to_go}s"
    
    def start_cartesian_impedance(self, Kx, Kxd):
        """
        Start Cartesian impedance controller.
        
        Args:
            Kx: Position stiffness (6D list/array)
            Kxd: Velocity damping (6D list/array)
        """
        print("[rpc]start_cartesian_impedance called", Kx, Kxd)
        Kx = torch.Tensor(Kx) if Kx is not None else None
        Kxd = torch.Tensor(Kxd) if Kxd is not None else None
        self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
        return "Cartesian impedance controller started"
    
    def start_joint_impedance(self, Kq, Kqd):
        """
        Start joint impedance controller.
        
        Args:
            Kq: Joint stiffness (7D list/array) or None for defaults
            Kqd: Joint damping (7D list/array) or None for defaults
        """
        Kq = torch.Tensor(Kq) if Kq is not None else None
        Kqd = torch.Tensor(Kqd) if Kqd is not None else None
        self.robot.start_joint_impedance(Kq=Kq, Kqd=Kqd)
        return "Joint impedance controller started"
    
    def update_desired_ee_pose(self, pose):
        """
        Update desired end-effector pose (for Cartesian impedance control).
        
        Args:
            pose: Desired end-effector pose [x, y, z, qx, qy, qz, qw] (7D list/array)
        """
        pose = np.array(pose)
        position = torch.Tensor(pose[:3])
        orientation = torch.Tensor(pose[3:7])  # quaternion in xyzw format
        self.robot.update_desired_ee_pose(position=position, orientation=orientation)
        return "Updated desired EE pose"
    
    def update_desired_joint_pos(self, pos):
        """
        Update desired joint positions (for joint impedance control).
        
        Args:
            pos: Desired joint positions (7D list/array)
        """
        pos = torch.Tensor(pos)
        self.robot.update_desired_joint_positions(pos)
        return "Updated desired joint positions"

    def start_joint_velocity_control(
        self, joint_vel_desired, hz=None, Kq=None, Kqd=None, **kwargs
    ):
        """Starts joint velocity control mode.
        Runs a non-blocking joint velocity controller.
        The desired joint velocities can be updated using `update_desired_joint_velocities`
        """
        ## hz, Kq, Kqd can all be none since there are default values in the original method
        ## https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/polymetis/robot_interface.py#L608
        
        ## convert the format to torch 
        joint_vel_desired = torch.Tensor(joint_vel_desired)
        hz = float(hz) if hz is not None else None
        Kq = torch.Tensor(Kq) if Kq is not None else None
        Kqd = torch.Tensor(Kqd) if Kqd is not None else None
        ## use the original methods
        return self.robot.start_joint_velocity_control(
            joint_vel_desired=joint_vel_desired, hz=hz, Kq=Kq, Kqd=Kqd, **kwargs
        )

    def update_desired_joint_velocities(self, velocities):
        """Update the desired joint velocities used by the joint velocities control mode.
        Requires starting a joint velocities controller with `start_joint_velocity_control` beforehand.
        """
        ## adapted from https://github.com/facebookresearch/fairo/blob/main/polymetis/polymetis/python/polymetis/robot_interface.py#L608
        velocities = torch.Tensor(velocities)
        return self.robot.update_desired_joint_velocities(velocities)
    
    def terminate_current_policy(self):
        """Terminate the currently running policy/controller."""
        self.robot.terminate_current_policy()
        return "Terminated current policy"
    
    def close(self):
        """Close the connection to the robot."""
        # Polymetis RobotInterface doesn't have explicit close
        return "Connection closed"


def main():
    parser = argparse.ArgumentParser(
        description='Single Franka arm ZeroRPC server for policy execution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--robot-port', type=int, default=50051,
                        help='Polymetis port for robot arm')
    parser.add_argument('--gripper-port', type=int, default=50052,
                        help='Polymetis port for gripper')
    parser.add_argument('--zerorpc-port', type=int, default=4242,
                        help='ZeroRPC server port for remote control')
    parser.add_argument('--ip', type=str, default='0.0.0.0',
                        help='IP address to bind ZeroRPC server (0.0.0.0 for all interfaces)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SINGLE FRANKA ARM SERVER")
    print("=" * 70)
    print(f"Robot port:        {args.robot_port}")
    print(f"Gripper port:      {args.gripper_port}")
    print(f"ZeroRPC bind:      {args.ip}:{args.zerorpc_port}")
    print("=" * 70)
    
    # Create server
    arm_server = SingleArmServer(
        robot_port=args.robot_port,
        gripper_port=args.gripper_port
    )
    
    # Start ZeroRPC server
    print(f"\nStarting ZeroRPC server on {args.ip}:{args.zerorpc_port}...")
    print("\nAvailable RPC methods (FrankaInterface API):")
    print("  State Queries:")
    print("    - get_ee_pose() -> [x, y, z, qx, qy, qz, qw]")
    print("    - get_joint_positions() -> [q1, ..., q7]")
    print("    - get_joint_velocities() -> [dq1, ..., dq7]")
    print("    - get_force_torque() -> [fx, fy, fz, tx, ty, tz]")
    print("  Gripper Control:")
    print("    - get_gripper_position() -> [width]")
    print("    - set_gripper_position(pos)")
    print("    - control_gripper(gripper_action)  # True=close, False=open")
    print("  Motion Control:")
    print("    - move_to_joint_positions(positions, time_to_go)")
    print("    - start_cartesian_impedance(Kx, Kxd)")
    print("    - start_joint_impedance(Kq, Kqd)")
    print("    - update_desired_ee_pose(pose)  # pose=[x,y,z,qx,qy,qz,qw]")
    print("    - update_desired_joint_pos(pos)")
    print("    - terminate_current_policy()")
    print("\nServer ready. Press Ctrl+C to stop.\n")
    
    try:
        server = zerorpc.Server(arm_server, heartbeat=20)
        server.bind(f"tcp://{args.ip}:{args.zerorpc_port}")
        server.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        print("Server stopped")


if __name__ == '__main__':
    main()
