# Franka Robot ZeroRPC Servers

This directory contains ZeroRPC server scripts to run on the Franka host PC (NUC) for remote robot control. These servers expose a Polymetis-based control interface via ZeroRPC that is compatible with `franka_interface.py` clients running on remote workstations.

## Overview

Two server configurations are provided:

1. **`single_arm_server.py`** - Controls a single Franka arm
2. **`dual_arm_server.py`** - Controls two Franka arms for bimanual tasks

Both servers implement the same API as `franka_interface.py` expects, allowing seamless remote control from policy inference workstations.

## Architecture

```
Workstation (GPU + Policy)          NUC (Franka Host PC)
┌──────────────────────┐            ┌────────────────────┐
│  main.py             │            │  single_arm_server │
│  - Policy inference  │  ZeroRPC   │  - Polymetis       │
│  - franka_interface  │◄──────────►│  - RobotInterface  │
│    (client)          │  Network   │  - GripperInterface│
└──────────────────────┘            └────────────────────┘
                                              │
                                              ▼
                                    ┌────────────────────┐
                                    │  Franka Robot      │
                                    │  - 7-DOF arm       │
                                    │  - Gripper         │
                                    └────────────────────┘
```

## Prerequisites

On the Franka host PC (NUC), you need:

- **Polymetis** installed and configured
- **Python 3.8+**
- **ZeroRPC**: `pip install zerorpc`
- **PyTorch**: `pip install torch`
- **scipy**: `pip install scipy`

## Single Arm Server

### Basic Usage

```bash
# On the NUC, run:
python single_arm_server.py --robot-port 50051 --gripper-port 50052 --zerorpc-port 4242
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--robot-port` | 50051 | Polymetis port for robot arm |
| `--gripper-port` | 50052 | Polymetis port for gripper |
| `--zerorpc-port` | 4242 | Port for ZeroRPC server |
| `--ip` | 0.0.0.0 | Bind address (0.0.0.0 = all interfaces) |

### Client Connection

From your workstation:

```python
from franka_interface import FrankaInterface

# Connect to single arm server
robot = FrankaInterface(ip='192.168.1.143', port=4242)

# Use the robot
joint_pos = robot.get_joint_positions()
robot.move_to_joint_positions(joint_pos, time_to_go=2.0)
```

## Dual Arm Server

### Basic Usage

```bash
# On the NUC, run:
python dual_arm_server.py \
  --left-robot-port 50051 \
  --left-gripper-port 50052 \
  --right-robot-port 50054 \
  --right-gripper-port 50055 \
  --left-zerorpc-port 4242 \
  --right-zerorpc-port 4243 \
  --dual-zerorpc-port 4244
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--left-robot-port` | 50051 | Polymetis port for left robot |
| `--left-gripper-port` | 50052 | Polymetis port for left gripper |
| `--right-robot-port` | 50054 | Polymetis port for right robot |
| `--right-gripper-port` | 50055 | Polymetis port for right gripper |
| `--left-zerorpc-port` | 4242 | ZeroRPC port for left arm |
| `--right-zerorpc-port` | 4243 | ZeroRPC port for right arm |
| `--dual-zerorpc-port` | 4244 | ZeroRPC port for bimanual methods |
| `--ip` | 0.0.0.0 | Bind address |

### Client Connection - Individual Arms

The dual arm server exposes **three separate ZeroRPC servers**:

1. **Left arm** (port 4242) - Standard FrankaInterface API
2. **Right arm** (port 4243) - Standard FrankaInterface API  
3. **Dual arm** (port 4244) - Bimanual methods

```python
from franka_interface import FrankaInterface

# Control left arm
left_robot = FrankaInterface(ip='192.168.1.143', port=4242)
left_pos = left_robot.get_joint_positions()

# Control right arm
right_robot = FrankaInterface(ip='192.168.1.143', port=4243)
right_pos = right_robot.get_joint_positions()

# Control both independently
left_robot.move_to_joint_positions(left_pos, time_to_go=2.0)
right_robot.move_to_joint_positions(right_pos, time_to_go=2.0)
```

### Client Connection - Bimanual Methods

```python
import zerorpc

# Connect to dual arm interface
dual = zerorpc.Client()
dual.connect('tcp://192.168.1.143:4244')

# Get both arms' states simultaneously
both_poses = dual.get_both_ee_poses()
# Returns: {'left': [x,y,z,rx,ry,rz], 'right': [x,y,z,rx,ry,rz]}

both_joints = dual.get_both_joint_positions()
# Returns: {'left': [q1,...,q7], 'right': [q1,...,q7]}

# Individual arm control via prefixed methods
dual.left_move_to_joint_positions([0,0,0,0,0,0,0], 2.0)
dual.right_set_gripper_position(0.04)

# Terminate all policies
dual.terminate_all_policies()
```

## API Reference

### State Queries

All methods compatible with `franka_interface.py`:

| Method | Returns | Description |
|--------|---------|-------------|
| `get_ee_pose()` | `[x,y,z,rx,ry,rz]` | End-effector pose (6D) |
| `get_joint_positions()` | `[q1,...,q7]` | Joint positions (7D) |
| `get_joint_velocities()` | `[dq1,...,dq7]` | Joint velocities (7D) |
| `get_force_torque()` | `[fx,fy,fz,tx,ty,tz]` | Force/torque (6D, zeros for now) |

### Gripper Control

| Method | Args | Description |
|--------|------|-------------|
| `get_gripper_position()` | - | Returns `[width]` in meters |
| `set_gripper_position(pos)` | `pos`: float (0.0-0.08) | Set gripper width |
| `control_gripper(action)` | `action`: bool | True=close, False=open |

### Motion Control

| Method | Args | Description |
|--------|------|-------------|
| `move_to_joint_positions(positions, time_to_go)` | `positions`: 7D list<br>`time_to_go`: float | Move to joint config |
| `start_cartesian_impedance(Kx, Kxd)` | `Kx`: 6D stiffness<br>`Kxd`: 6D damping | Start Cartesian controller |
| `start_joint_impedance(Kq, Kqd)` | `Kq`: 7D stiffness<br>`Kqd`: 7D damping | Start joint controller |
| `update_desired_ee_pose(pose)` | `pose`: 6D list | Update Cartesian target |
| `update_desired_joint_pos(pos)` | `pos`: 7D list | Update joint target |
| `terminate_current_policy()` | - | Stop active controller |

## Typical Workflow

### 1. Start Server on NUC

```bash
# SSH into NUC
ssh franka@192.168.1.143

# Single arm
python single_arm_server.py

# Or dual arm
python dual_arm_server.py
```

### 2. Test Connection from Workstation

```bash
# On workstation
cd examples/franka_pi05
python test_robot.py --ip 192.168.1.143 --port 4242
```

### 3. Run Policy

```bash
# On workstation
python main.py \
  --external-camera 327122079691 \
  --wrist-camera 218622273043 \
  --nuc-ip 192.168.1.143 \
  --nuc-port 4242
```

## Differences from bimanual_mirror_server.py

The example `bimanual_mirror_server.py` provided:
- **Automatic mirroring**: Continuously copies lead arm to follower arm
- **Teleoperation focus**: Designed for manual teaching/demonstration
- **High frequency control loop**: 200 Hz mirroring in separate thread

The new servers (`single_arm_server.py`, `dual_arm_server.py`):
- **Policy execution focus**: Designed for remote policy control
- **Explicit commands**: Client sends specific joint/EE targets
- **Standard API**: Compatible with existing `franka_interface.py` client
- **Dual independent control**: Each arm controlled separately via different ports

## Troubleshooting

### Connection Refused

```python
# Error: "Connection refused" when connecting
```

**Solutions:**
1. Check server is running on NUC: `ps aux | grep arm_server`
2. Check firewall allows port: `sudo ufw allow 4242`
3. Verify NUC IP: `ip addr show`

### Import Error: polymetis

```python
# Error: "No module named 'polymetis'"
```

**Solution:** These scripts run on the NUC with Polymetis installed, not on the workstation. Copy them to the NUC:

```bash
# On workstation
scp single_arm_server.py franka@192.168.1.143:~/
scp dual_arm_server.py franka@192.168.1.143:~/
```

### Slow Response / Timeouts

**Solutions:**
1. Increase ZeroRPC heartbeat: Change `heartbeat=20` to `heartbeat=60` in both server and client
2. Check network latency: `ping 192.168.1.143`
3. Use wired Ethernet connection (not WiFi)

### Robot Not Moving

**Solutions:**
1. Check robot is not in E-Stop
2. Check Polymetis is running: `systemctl status polymetis`
3. Test with direct Polymetis commands first
4. Check impedance controller is active: Call `start_joint_impedance()` before `update_desired_joint_pos()`

## Performance Notes

- **Network latency**: ~1-5ms on local network (wired)
- **ZeroRPC overhead**: ~1-2ms per call
- **Polymetis control**: 1kHz internal loop (1ms)
- **Policy frequency**: 15Hz recommended (66ms period)
- **Gripper response**: ~50-200ms depending on distance

The servers are designed for policy execution at 10-15 Hz, not real-time teleoperation at 200+ Hz.

## See Also

- `franka_interface.py` - Client interface
- `test_robot.py` - Test script for connection
- `main.py` - Full policy execution pipeline
- Polymetis documentation: https://facebookresearch.github.io/fairo/polymetis/
