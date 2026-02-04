# Bimanual Mirror Server

A high-frequency bimanual mirroring system that directly controls two Franka arms via Polymetis RobotInterface and provides a ZeroRPC interface for state queries.

## Overview

This system consists of:
1. **Server** (`bimanual_mirror_server.py`): Directly controls both arms and hosts a ZeroRPC API
2. **Client** (`bimanual_mirror_client.py`): Example client for querying arm states

## Key Features

- **Direct Control**: Uses Polymetis RobotInterface directly (no intermediate servers needed)
- **High Frequency**: Supports 200+ Hz control loop for responsive mirroring
- **State Query Interface**: ZeroRPC server allows external clients to query:
  - Joint positions and velocities
  - End-effector poses
  - Mirroring status and statistics
- **Thread-Safe**: Mirroring runs in background thread while serving RPC requests

## Setup

Make sure both robot controllers are running:
```bash
# Terminal 1: Start lead arm controller (port 50054)
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=<lead_ip>

# Terminal 2: Start follower arm controller (port 50051)
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=<follower_ip>
```

## Usage

### Start the Server

Basic usage (auto-starts mirroring):
```bash
python scripts_real/bimanual_mirror_server.py
```

Custom configuration:
```bash
python scripts_real/bimanual_mirror_server.py \
    --lead-port 50054 \
    --follower-port 50051 \
    --zerorpc-port 4244 \
    --frequency 200.0 \
    --threshold 0.001 \
    --joint-kp-scale 1.0
```

For stiffer/faster response (follower tracks more aggressively):
```bash
python scripts_real/bimanual_mirror_server.py --joint-kp-scale 1.5
```

For softer/smoother response (more compliant):
```bash
python scripts_real/bimanual_mirror_server.py --joint-kp-scale 0.7
```

Start without auto-mirroring (wait for RPC command):
```bash
python scripts_real/bimanual_mirror_server.py --no-auto-start
```

### Query States with Client

Get mirroring status:
```bash
python scripts_real/bimanual_mirror_client.py --command status
```

Get joint positions:
```bash
python scripts_real/bimanual_mirror_client.py --command positions
```

Get end-effector poses:
```bash
python scripts_real/bimanual_mirror_client.py --command poses
```

Get gripper status:
```bash
python scripts_real/bimanual_mirror_client.py --command gripper_status
```

Control grippers:
```bash
# Open lead gripper to 80mm (default)
python scripts_real/bimanual_mirror_client.py --command open_lead

# Open follower gripper to specific width (e.g., 50mm = 0.05m)
python scripts_real/bimanual_mirror_client.py --command open_follower --gripper-width 0.05

# Close grippers
python scripts_real/bimanual_mirror_client.py --command close_lead
python scripts_real/bimanual_mirror_client.py --command close_follower
```

Continuous monitoring:
```bash
python scripts_real/bimanual_mirror_client.py --command monitor --rate 10.0
```

Start/stop mirroring remotely:
```bash
python scripts_real/bimanual_mirror_client.py --command start
python scripts_real/bimanual_mirror_client.py --command stop
```

### Use in Your Own Code

```python
import zerorpc
import numpy as np

# Connect to server
client = zerorpc.Client()
client.connect("tcp://localhost:4244")

# Get both arms' joint positions
positions = client.get_both_joint_positions()
lead_pos = np.array(positions['lead'])
follower_pos = np.array(positions['follower'])

# Get end-effector poses
poses = client.get_both_ee_poses()
lead_ee = np.array(poses['lead'])  # [x, y, z, rx, ry, rz]
follower_ee = np.array(poses['follower'])

# Get gripper widths
gripper_widths = client.get_both_gripper_widths()
lead_gripper = gripper_widths['lead']  # in meters
follower_gripper = gripper_widths['follower']

# Control grippers
client.set_lead_gripper_width(0.08)  # Open to 80mm
client.close_follower_gripper()  # Close/grasp

# Check mirroring status
status = client.get_mirroring_status()
print(f"Mirroring active: {status['active']}")
print(f"Frequency: {status['frequency']} Hz")

# Control mirroring
client.start_mirroring()
client.stop_mirroring()
```

## Available RPC Methods

### State Query Methods
- `get_lead_ee_pose()` → List[float] - Lead arm EE pose [x, y, z, rx, ry, rz]
- `get_follower_ee_pose()` → List[float] - Follower arm EE pose
- `get_lead_joint_positions()` → List[float] - Lead arm joint positions (7 DOF)
- `get_follower_joint_positions()` → List[float] - Follower arm joint positions
- `get_lead_joint_velocities()` → List[float] - Lead arm joint velocities
- `get_follower_joint_velocities()` → List[float] - Follower arm joint velocities
- `get_both_ee_poses()` → Dict - Both arms' EE poses
- `get_both_joint_positions()` → Dict - Both arms' joint positions
- `get_both_joint_velocities()` → Dict - Both arms' joint velocities
- `get_mirroring_status()` → Dict - Mirroring status and statistics

### Gripper Query Methods
- `get_lead_gripper_width()` → float - Lead gripper width in meters
- `get_follower_gripper_width()` → float - Follower gripper width in meters
- `get_both_gripper_widths()` → Dict - Both grippers' widths

### Gripper Control Methods
- `set_lead_gripper_width(width, speed=0.1, force=20.0)` → str - Set lead gripper to width (m)
- `set_follower_gripper_width(width, speed=0.1, force=20.0)` → str - Set follower gripper to width (m)
- `open_lead_gripper(width=0.08, speed=0.1, force=20.0)` → str - Open lead gripper
- `close_lead_gripper(speed=0.1, force=20.0)` → str - Close/grasp with lead gripper
- `open_follower_gripper(width=0.08, speed=0.1, force=20.0)` → str - Open follower gripper
- `close_follower_gripper(speed=0.1, force=20.0)` → str - Close/grasp with follower gripper

### Control Methods
- `start_mirroring()` → str - Start mirroring lead to follower
- `stop_mirroring()` → str - Stop mirroring

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lead-port` | 50054 | Polymetis port for lead arm |
| `--follower-port` | 50051 | Polymetis port for follower arm |
| `--zerorpc-port` | 4244 | ZeroRPC server port |
| `--frequency` | 200.0 | Mirroring control frequency (Hz) |
| `--threshold` | 0.001 | Min joint change to count as movement (rad) |
| `--joint-kp-scale` | 1.0 | Joint stiffness scale (higher = faster/stiffer) |
| `--no-auto-start` | False | Don't auto-start mirroring |

## Performance

- **Control Frequency**: 200 Hz (5ms per iteration)
- **Control Method**: Joint impedance controller for non-blocking, high-frequency updates
- **Latency**: <10ms follower lag
- **Accuracy**: Sub-millimeter positioning

**Note**: The system uses Polymetis's `start_joint_impedance()` and `update_desired_joint_positions()` for non-blocking control. This allows the control loop to run at 200 Hz without waiting for motion completion, unlike `move_to_joint_positions()` which blocks until the movement finishes.

## Comparison with Previous Approach

### Previous (with intermediate servers):
```
Client → ZeroRPC → Server1 → Polymetis → Robot1
                → Server2 → Polymetis → Robot2
```
- Extra network hops add latency
- Two separate server processes to manage
- More complex deployment

### Current (direct control):
```
Server → Polymetis → Robot1 (lead)
      → Polymetis → Robot2 (follower)
Client → ZeroRPC → Server (for queries only)
```
- Direct robot control for minimal latency
- Single server process
- ZeroRPC only for state queries (not control path)
- Higher achievable frequencies (200+ Hz)

## Troubleshooting

**Issue**: "Connection refused" error
- **Solution**: Make sure both Polymetis robot controllers are running

**Issue**: Low actual frequency (e.g., 3 Hz instead of 200 Hz)
- **Root cause**: The old code used `move_to_joint_positions()` which blocks until motion completes
- **Solution**: The updated code uses `start_joint_impedance()` + `update_desired_joint_positions()` for non-blocking control
- **Check**: Make sure you're using the latest version of the script with impedance control

**Issue**: Follower moves too slowly or lags significantly
- **Solution**: Increase `--joint-kp-scale` (try 1.5 or 2.0) for higher stiffness and faster response

**Issue**: Follower is too jerky or oscillates
- **Solution**: Decrease `--joint-kp-scale` (try 0.7 or 0.5) for softer, more compliant motion

**Issue**: Follower doesn't move
- **Solution**: Check that mirroring is active with `get_mirroring_status()` and ensure lead arm is being moved
