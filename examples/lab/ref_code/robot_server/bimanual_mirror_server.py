#!/usr/bin/env python3
"""
Bimanual mirroring server that:
1. Directly controls two Franka arms via polymetis RobotInterface
2. Mirrors lead arm movements to follower arm at high frequency
3. Hosts a ZeroRPC server for external clients to query arm states
"""

import zerorpc
from polymetis import RobotInterface, GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch
import threading
import time
import argparse
from typing import Optional
import math

# Define home positions
HOME_JOINTS = [0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4]

class BimanualMirrorController:
    """Controller that mirrors lead arm to follower arm and provides state query interface."""
    
    def __init__(self, 
                 lead_port=50054, 
                 follower_port=50051,
                 follower_gripper_port=50052,
                 mirror_frequency=200.0,
                 position_threshold=0.001,
                 joint_kp_scale=1.0,
                 return_to_home=False):
        """
        Initialize bimanual mirror controller.
        
        Args:
            lead_port: Polymetis port for lead arm
            follower_port: Polymetis port for follower arm
            mirror_frequency: Control loop frequency in Hz
            position_threshold: Minimum joint change to trigger movement (rad)
            joint_kp_scale: Scale factor for joint stiffness (higher = stiffer)
            return_to_home: If True, return lead arm to home position on startup
        """
        print(f"Connecting to lead arm on port {lead_port}...")
        self.lead_robot = RobotInterface(ip_address='localhost', port=lead_port)
        # self.lead_gripper = GripperInterface(ip_address='localhost', port=lead_port)
        
        print(f"Connecting to follower arm on port {follower_port}...")
        self.follower_robot = RobotInterface(ip_address='localhost', port=follower_port)
        self.follower_gripper = GripperInterface(ip_address='localhost', port=follower_gripper_port)

        if return_to_home:
            print("Lead robot joint positions:", self.lead_robot.get_joint_positions().numpy())
            self.lead_robot.move_to_joint_positions(
                positions=torch.Tensor(HOME_JOINTS),
                time_to_go=2.0
            )
            self.follower_robot.move_to_joint_positions(
                positions=torch.Tensor(HOME_JOINTS),
                time_to_go=2.0
            )
        
        self.mirror_frequency = mirror_frequency
        self.position_threshold = position_threshold
        self.joint_kp_scale = joint_kp_scale
        
        self.mirroring_active = False
        self.mirror_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.iteration_count = 0
        self.movement_count = 0
        
        print(f"Lead arm EE pose (RPY): {self._format_ee_pose(self.lead_robot.get_ee_pose())}")
        print(f"Lead arm EE pose (Quat): {self._format_ee_pose_quat(self.lead_robot.get_ee_pose())}")
        print(f"Follower arm EE pose (RPY): {self._format_ee_pose(self.follower_robot.get_ee_pose())}")
        print(f"Follower arm EE pose (Quat): {self._format_ee_pose_quat(self.follower_robot.get_ee_pose())}")
    
    def _format_ee_pose(self, pose_data):
        """Convert polymetis pose to [x, y, z, rx, ry, rz] format."""
        pos = pose_data[0].numpy()
        quat_xyzw = pose_data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec])
    
    def _format_ee_pose_quat(self, pose_data):
        """Convert polymetis pose to [x, y, z, qx, qy, qz, qw] format."""
        pos = pose_data[0].numpy()
        quat_xyzw = pose_data[1].numpy()
        return np.concatenate([pos, quat_xyzw])
    
    # ========= State query methods (for ZeroRPC) =========
    
    def get_lead_ee_pose(self):
        """Get lead arm end-effector pose [x, y, z, rx, ry, rz]."""
        return self._format_ee_pose_quat(self.lead_robot.get_ee_pose()).tolist()
    
    def get_follower_ee_pose(self):
        """Get follower arm end-effector pose [x, y, z, rx, ry, rz]."""
        return self._format_ee_pose_quat(self.follower_robot.get_ee_pose()).tolist()
    
    def get_lead_joint_positions(self):
        """Get lead arm joint positions."""
        return self.lead_robot.get_joint_positions().numpy().tolist()
    
    def get_follower_joint_positions(self):
        """Get follower arm joint positions."""
        return self.follower_robot.get_joint_positions().numpy().tolist()
    
    def get_lead_joint_velocities(self):
        """Get lead arm joint velocities."""
        return self.lead_robot.get_joint_velocities().numpy().tolist()
    
    def get_follower_joint_velocities(self):
        """Get follower arm joint velocities."""
        return self.follower_robot.get_joint_velocities().numpy().tolist()
    
    def get_both_ee_poses(self):
        """Get both arms' end-effector poses."""
        return {
            'lead': self.get_lead_ee_pose(),
            'follower': self.get_follower_ee_pose()
        }
    
    def get_both_joint_positions(self):
        """Get both arms' joint positions."""
        return {
            'lead': self.get_lead_joint_positions(),
            'follower': self.get_follower_joint_positions()
        }
    
    def get_both_joint_velocities(self):
        """Get both arms' joint velocities."""
        return {
            'lead': self.get_lead_joint_velocities(),
            'follower': self.get_follower_joint_velocities()
        }
    
    def get_mirroring_status(self):
        """Get mirroring status and statistics."""
        return {
            'active': self.mirroring_active,
            'iteration_count': self.iteration_count,
            'movement_count': self.movement_count,
            'frequency': self.mirror_frequency
        }
    
    # ========= Gripper query methods =========
    
    # def get_lead_gripper_width(self):
    #     """Get lead gripper opening width in meters."""
    #     return self.lead_gripper.get_state().width
    
    def get_follower_gripper_width(self):
        """Get follower gripper opening width in meters."""
        return self.follower_gripper.get_state().width
    
    # def get_both_gripper_widths(self):
    #     """Get both grippers' opening widths."""
    #     return {
    #         'lead': self.get_lead_gripper_width(),
    #         'follower': self.get_follower_gripper_width()
    #     }
    
    # ========= Gripper control methods =========
    
    # def set_lead_gripper_width(self, width, speed=0.1, force=20.0):
    #     """
    #     Set lead gripper to target width.
        
    #     Args:
    #         width: Target width in meters (0.0 = closed, ~0.08 = open)
    #         speed: Gripper speed in m/s
    #         force: Gripper force in Newtons
    #     """
    #     width = float(np.clip(width, 0.0, 0.08))
    #     self.lead_gripper.goto(width=width, speed=speed, force=force, blocking=False)
    #     return f"Lead gripper moving to {width:.4f}m"
    
    def set_follower_gripper_width(self, width, speed=0.1, force=20.0):
        """
        Set follower gripper to target width.
        
        Args:
            width: Target width in meters (0.0 = closed, ~0.08 = open)
            speed: Gripper speed in m/s
            force: Gripper force in Newtons
        """
        width = float(np.clip(width, 0.0, 0.08))
        self.follower_gripper.goto(width=width, speed=speed, force=force, blocking=False)
        return f"Follower gripper moving to {width:.4f}m"
    
    # def open_lead_gripper(self, width=0.08, speed=0.1, force=20.0):
    #     """Open lead gripper."""
    #     self.lead_gripper.goto(width=width, speed=speed, force=force, blocking=False)
    #     return f"Opening lead gripper to {width:.4f}m"
    
    def close_lead_gripper(self, speed=0.1, force=20.0):
        """Close/grasp with lead gripper."""
        self.lead_gripper.grasp(speed=speed, force=force, blocking=False)
        return "Closing lead gripper"
    
    def open_follower_gripper(self, width=0.08, speed=0.1, force=20.0):
        """Open follower gripper."""
        self.follower_gripper.goto(width=width, speed=speed, force=force, blocking=False)
        return f"Opening follower gripper to {width:.4f}m"
    
    def close_follower_gripper(self, speed=0.1, force=20.0):
        """Close/grasp with follower gripper."""
        self.follower_gripper.grasp(speed=speed, force=force, blocking=False)
        return "Closing follower gripper"
    
    # ========= Interface for client =========

    def set_gripper_position(self, width):
        """Set gripper position for follower gripper."""
        return self.set_follower_gripper_width(width)
    
    def get_gripper_position(self):
        """Get gripper position for follower gripper."""
        return self.get_follower_gripper_width()
    
    def get_ee_pose(self):
        """Get end-effector pose for follower arm [x, y, z, rx, ry, rz]."""
        return self.get_follower_ee_pose()
    
    def get_joint_positions(self):
        """Get current joint positions for follower arm (7 DOF)."""
        return self.get_follower_joint_positions()
    
    def get_joint_velocities(self):
        """Get current joint velocities for follower arm (7 DOF)."""
        return self.get_follower_joint_velocities()
    
    def start_cartesian_impedance(self, Kx, Kxd):
        pass

    def start_joint_impedance(self, Kq, Kqd):
        pass

    def update_desired_ee_pose(self, pose):
        pass

    def update_desired_joint_pos(self, positions):
        pass

    def move_to_joint_positions(self, positions, time_to_go):
        pass

    def control_gripper(self, gripper_action: bool):
        # open or close follower gripper
        if gripper_action:
            return self.close_follower_gripper()
        else:
            return self.open_follower_gripper()
    
    def terminate_current_policy(self):
        """Terminate current robot policy for follower arm."""
        self.follower_robot.terminate_current_policy()

    # ========= Mirroring control methods =========
    
    def start_mirroring(self):
        """Start mirroring lead arm to follower arm."""
        if self.mirroring_active:
            print("[WARNING] Mirroring already active")
            return "Mirroring already active"
        
        self._stop_event.clear()
        self.mirror_thread = threading.Thread(target=self._mirror_loop, daemon=True)
        self.mirror_thread.start()
        return "Mirroring started"
    
    def stop_mirroring(self):
        """Stop mirroring."""
        if not self.mirroring_active:
            print("[WARNING] Mirroring not active")
            return "Mirroring not active"
        
        self._stop_event.set()
        if self.mirror_thread:
            self.mirror_thread.join(timeout=2.0)
        return "Mirroring stopped"
    
    def _mirror_loop(self):
        """Main mirroring control loop (runs in separate thread)."""
        self.mirroring_active = True
        self.iteration_count = 0
        self.movement_count = 0
        
        print(f"\n{'='*60}")
        print(f"Starting mirroring at {self.mirror_frequency} Hz")
        print(f"Position threshold: {self.position_threshold} rad")
        print(f"{'='*60}\n")
        
        dt = 1.0 / self.mirror_frequency
        prev_lead_positions = self.lead_robot.get_joint_positions().numpy()
        
        # Initial synchronization
        print("Synchronizing follower to lead initial position...")
        self.follower_robot.move_to_joint_positions(
            positions=torch.Tensor(prev_lead_positions),
            time_to_go=2.0
        )
        time.sleep(2.5)
        
        # Start joint impedance controller for high-frequency updates
        print("Starting joint impedance controller on follower...")
        # Use default impedance gains from Polymetis (scaled by user parameter)
        Kq = torch.Tensor([75.0, 75.0, 75.0, 75.0, 75.0, 50.0, 50.0]) * self.joint_kp_scale
        Kqd = 2 * torch.sqrt(Kq)  # Critical damping
        self.follower_robot.start_joint_impedance(Kq=Kq, Kqd=Kqd)
        print("Synchronization complete. Starting mirroring...\n")
        
        start_time = time.time()
        last_print_time = start_time
        
        try:
            while not self._stop_event.is_set():
                loop_start = time.time()
                
                # Get current lead arm joint positions
                current_lead_positions = self.lead_robot.get_joint_positions().numpy()
                
                # Calculate position difference
                position_diff = np.abs(current_lead_positions - prev_lead_positions)
                max_diff = np.max(position_diff)
                
                # Update desired joint positions (non-blocking with impedance controller)
                self.follower_robot.update_desired_joint_positions(
                    torch.Tensor(current_lead_positions)
                )
                
                # Count as movement if above threshold
                if max_diff > self.position_threshold:
                    self.movement_count += 1
                    
                prev_lead_positions = current_lead_positions.copy()
                
                self.iteration_count += 1
                
                # Print status every second
                current_time = time.time()
                if current_time - last_print_time >= 1.0:
                    actual_freq = self.iteration_count / (current_time - start_time)
                    print(f"[{self.iteration_count:08d}] "
                          f"Movements: {self.movement_count:06d} | "
                          f"Freq: {actual_freq:.1f} Hz | "
                          f"Max diff: {max_diff:.4f} rad")
                    last_print_time = current_time
                
                # Maintain frequency
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif self.iteration_count % 100 == 0:
                    print(f"[WARNING] Loop slower than target: {elapsed*1000:.1f}ms vs {dt*1000:.1f}ms target")
        
        except Exception as e:
            print(f"\n[ERROR] Exception in mirror loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Terminate impedance controller
            try:
                print("\nTerminating follower impedance controller...")
                self.follower_robot.terminate_current_policy()
            except Exception as e:
                print(f"[WARNING] Error terminating policy: {e}")
            
            self.mirroring_active = False
            elapsed_total = time.time() - start_time
            avg_freq = self.iteration_count / elapsed_total if elapsed_total > 0 else 0
            print(f"\nMirroring stopped.")
            print(f"Total iterations: {self.iteration_count}")
            print(f"Total movements: {self.movement_count}")
            print(f"Average frequency: {avg_freq:.1f} Hz")
    
    def shutdown(self):
        """Shutdown controller and cleanup."""
        if self.mirroring_active:
            self.stop_mirroring()
        print("Controller shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description='Bimanual mirror server with state query interface',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--lead-port', type=int, default=50054,
                        help='Polymetis port for lead arm')
    parser.add_argument('--follower-port', type=int, default=50051,
                        help='Polymetis port for follower arm')
    parser.add_argument('--zerorpc-port', type=int, default=4242,
                        help='ZeroRPC server port for state queries')
    parser.add_argument('--frequency', type=float, default=200.0,
                        help='Mirroring control loop frequency in Hz')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Minimum joint position change to trigger movement (rad)')
    parser.add_argument('--joint-kp-scale', type=float, default=1.0,
                        help='Scale factor for joint stiffness (higher = stiffer, faster response)')
    parser.add_argument('--no-auto-start', action='store_true',
                        help='Do not automatically start mirroring (wait for RPC command)')
    parser.add_argument('--return-to-home', action='store_true',
                        help='Return lead arm to home position on startup')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BIMANUAL MIRROR SERVER")
    print("=" * 60)
    print(f"Lead arm port:     {args.lead_port}")
    print(f"Follower arm port: {args.follower_port}")
    print(f"ZeroRPC port:      {args.zerorpc_port}")
    print(f"Mirror frequency:  {args.frequency} Hz")
    print(f"Position threshold: {args.threshold} rad")
    print(f"Joint Kp scale:    {args.joint_kp_scale}")
    print(f"Auto-start:        {not args.no_auto_start}")
    print("=" * 60)
    
    # Create controller
    controller = BimanualMirrorController(
        lead_port=args.lead_port,
        follower_port=args.follower_port,
        mirror_frequency=args.frequency,
        position_threshold=args.threshold,
        joint_kp_scale=args.joint_kp_scale,
        return_to_home=args.return_to_home,
    )
    
    # Start mirroring if auto-start enabled
    if not args.no_auto_start:
        controller.start_mirroring()
    
    # Start ZeroRPC server
    print(f"\nStarting ZeroRPC server on port {args.zerorpc_port}...")
    print("Available RPC methods:")
    print("  State Queries:")
    print("    - get_lead_ee_pose()")
    print("    - get_follower_ee_pose()")
    print("    - get_lead_joint_positions()")
    print("    - get_follower_joint_positions()")
    print("    - get_lead_joint_velocities()")
    print("    - get_follower_joint_velocities()")
    print("    - get_both_ee_poses()")
    print("    - get_both_joint_positions()")
    print("    - get_both_joint_velocities()")
    print("    - get_mirroring_status()")
    print("  Gripper Queries:")
    print("    - get_lead_gripper_width()")
    print("    - get_follower_gripper_width()")
    print("    - get_both_gripper_widths()")
    print("  Gripper Control:")
    print("    - set_lead_gripper_width(width, speed=0.1, force=20.0)")
    print("    - set_follower_gripper_width(width, speed=0.1, force=20.0)")
    print("    - open_lead_gripper(width=0.08, speed=0.1, force=20.0)")
    print("    - close_lead_gripper(speed=0.1, force=20.0)")
    print("    - open_follower_gripper(width=0.08, speed=0.1, force=20.0)")
    print("    - close_follower_gripper(speed=0.1, force=20.0)")
    print("  Mirroring Control:")
    print("    - start_mirroring()")
    print("    - stop_mirroring()")
    print("\nPress Ctrl+C to stop server\n")
    
    try:
        server = zerorpc.Server(controller, heartbeat=20)
        server.bind(f"tcp://0.0.0.0:{args.zerorpc_port}")
        server.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        controller.shutdown()
        print("Server stopped")


if __name__ == '__main__':
    main()
