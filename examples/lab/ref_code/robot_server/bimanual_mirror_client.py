#!/usr/bin/env python3
"""
Example client for the bimanual mirror server.
Demonstrates how to query arm states via ZeroRPC.
"""

import zerorpc
import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Client to query bimanual mirror server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--ip', type=str, default='localhost',
                        help='IP address of mirror server')
    parser.add_argument('--port', type=int, default=4244,
                        help='ZeroRPC port of mirror server')
    parser.add_argument('--command', type=str, default='status',
                        choices=['status', 'positions', 'poses', 'velocities', 
                                'start', 'stop', 'monitor', 'gripper_status',
                                'open_lead', 'close_lead', 'open_follower', 'close_follower'],
                        help='Command to execute')
    parser.add_argument('--rate', type=float, default=10.0,
                        help='Monitoring rate in Hz (for monitor command)')
    parser.add_argument('--gripper-width', type=float, default=0.08,
                        help='Gripper width in meters for open commands (default: 0.08m = 80mm)')
    
    args = parser.parse_args()
    
    # Connect to server
    print(f"Connecting to {args.ip}:{args.port}...")
    client = zerorpc.Client(heartbeat=20, timeout=30)
    client.connect(f"tcp://{args.ip}:{args.port}")
    print("Connected!\n")
    
    try:
        if args.command == 'status':
            # Get mirroring status
            status = client.get_mirroring_status()
            print("Mirroring Status:")
            print(f"  Active:         {status['active']}")
            print(f"  Iterations:     {status['iteration_count']}")
            print(f"  Movements:      {status['movement_count']}")
            print(f"  Frequency:      {status['frequency']} Hz")
        
        elif args.command == 'positions':
            # Get joint positions
            positions = client.get_both_joint_positions()
            print("Joint Positions:")
            print(f"  Lead:     {np.array2string(np.array(positions['lead']), precision=4, suppress_small=True)}")
            print(f"  Follower: {np.array2string(np.array(positions['follower']), precision=4, suppress_small=True)}")
        
        elif args.command == 'poses':
            # Get end-effector poses
            poses = client.get_both_ee_poses()
            print("End-Effector Poses [x, y, z, rx, ry, rz]:")
            print(f"  Lead:     {np.array2string(np.array(poses['lead']), precision=4, suppress_small=True)}")
            print(f"  Follower: {np.array2string(np.array(poses['follower']), precision=4, suppress_small=True)}")
        
        elif args.command == 'velocities':
            # Get joint velocities
            velocities = client.get_both_joint_velocities()
            print("Joint Velocities:")
            print(f"  Lead:     {np.array2string(np.array(velocities['lead']), precision=4, suppress_small=True)}")
            print(f"  Follower: {np.array2string(np.array(velocities['follower']), precision=4, suppress_small=True)}")
        
        elif args.command == 'start':
            # Start mirroring
            result = client.start_mirroring()
            print(f"Start mirroring: {result}")
        
        elif args.command == 'stop':
            # Stop mirroring
            result = client.stop_mirroring()
            print(f"Stop mirroring: {result}")
        
        elif args.command == 'gripper_status':
            # Get gripper widths
            widths = client.get_both_gripper_widths()
            print("Gripper Widths (meters):")
            print(f"  Lead:     {widths['lead']:.4f}m ({widths['lead']*1000:.1f}mm)")
            print(f"  Follower: {widths['follower']:.4f}m ({widths['follower']*1000:.1f}mm)")
        
        elif args.command == 'open_lead':
            # Open lead gripper
            result = client.open_lead_gripper(width=args.gripper_width)
            print(result)
        
        elif args.command == 'close_lead':
            # Close lead gripper
            result = client.close_lead_gripper()
            print(result)
        
        elif args.command == 'open_follower':
            # Open follower gripper
            result = client.open_follower_gripper(width=args.gripper_width)
            print(result)
        
        elif args.command == 'close_follower':
            # Close follower gripper
            result = client.close_follower_gripper()
            print(result)
        
        elif args.command == 'monitor':
            # Continuous monitoring
            print(f"Monitoring at {args.rate} Hz. Press Ctrl+C to stop.\n")
            dt = 1.0 / args.rate
            
            try:
                while True:
                    loop_start = time.time()
                    
                    # Get data
                    status = client.get_mirroring_status()
                    positions = client.get_both_joint_positions()
                    
                    # Calculate difference
                    lead_pos = np.array(positions['lead'])
                    follower_pos = np.array(positions['follower'])
                    diff = np.abs(lead_pos - follower_pos)
                    max_diff = np.max(diff)
                    
                    # Print
                    print(f"[{status['iteration_count']:08d}] "
                          f"Active: {status['active']} | "
                          f"Movements: {status['movement_count']:06d} | "
                          f"Max diff: {max_diff:.4f} rad")
                    
                    # Sleep to maintain rate
                    elapsed = time.time() - loop_start
                    sleep_time = dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
    
    finally:
        client.close()
        print("\nDisconnected")


if __name__ == '__main__':
    main()
