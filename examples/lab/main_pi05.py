"""Main script for running pi05_droid policy on Franka robot.

This script runs on your workstation with the 5090 GPU and:
1. Loads the pi05_droid policy locally (on GPU)
2. Captures images from RealSense cameras
3. Connects to the Franka NUC via ZeroRPC
4. Runs policy inference and executes actions

Usage:
    # With real hardware:
    python main.py

    # With mock cameras (testing):
    python main.py --use-mock-cameras

    # Custom config:
    python main.py --nuc-ip 192.168.1.143 --external-camera <serial>
"""

SINGLE_TASK="Place the orange cube onto the green coaster."

import contextlib
import datetime
import signal
import time
import threading
import multiprocessing
from multiprocessing import Process, Queue
from typing import Optional
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
from traitlets import observe
import tyro
from typing import Literal
import json
from moviepy import ImageSequenceClip

# Import openpi modules
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from PIL import Image

# Import local modules
import openpi.lab_utils.camera_utils as camera_utils
import openpi.lab_utils.config as cfg
from franka_interface import FrankaInterface, MockRobot
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

GRIPPER_CLOSE_THRESHOLD = 0.07  # meters (70mm)


def display_process_func(frame_queue: Queue, quit_event):
    """
    Separate process for displaying camera feeds.
    This runs in its own process to avoid blocking the main thread.
    """
    cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Cameras", 960, 240)
    
    while not quit_event.is_set():
        try:
            # Get frame from queue with timeout
            if not frame_queue.empty():
                frame_data = frame_queue.get_nowait()
                if frame_data is None:  # Poison pill to stop
                    break
                
                combined_view, t_step = frame_data
                
                # Add step info
                cv2.putText(combined_view, f"Step: {t_step}", (10, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Robot Cameras", combined_view)
            
            # Process GUI events
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                quit_event.set()
                break
                
        except Exception as e:
            pass  # Ignore errors and continue
    
    cv2.destroyAllWindows()


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until after policy inference completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        quat: [qx, qy, qz, qw] (Franka format)

    Returns:
        [roll, pitch, yaw] in radians
    """
    rot = R.from_quat(quat)  # scipy expects [qx, qy, qz, qw]
    euler = rot.as_euler("xyz", degrees=False)
    return euler


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to quaternion.

    Args:
        euler: [roll, pitch, yaw] in radians

    Returns:
        [qx, qy, qz, qw] quaternion (Franka format)
    """
    rot = R.from_euler("xyz", euler, degrees=False)
    quat = rot.as_quat()  # Returns [qx, qy, qz, qw]
    return quat


def apply_euler_delta(current_euler: np.ndarray, delta_euler: np.ndarray, mode="add") -> np.ndarray:
    """
    Apply Euler angle delta using rotation composition.
    This is more robust than simple addition for large rotations.

    Args:
        current_euler: [roll, pitch, yaw] current orientation
        delta_euler: [Δroll, Δpitch, Δyaw] delta orientation
        mode: "add" to add delta, "multiply" to compose rotations

    Returns:
        [roll, pitch, yaw] new orientation
    """

    if mode == "add":
        # Simple addition (not recommended for large angles)
        new_euler = current_euler + delta_euler
    elif mode == "multiply":
        # Convert to rotation matrices
        R_current = R.from_euler("xyz", current_euler, degrees=False)
        R_delta = R.from_euler("xyz", delta_euler, degrees=False)
        # Compose: R_new = R_delta * R_current
        R_new = R_delta * R_current
        new_euler = R_new.as_euler("xyz", degrees=False)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert back to Euler
    return new_euler


class FrankaPolicyRunner:
    """Manages policy inference and robot control."""

    def __init__(self, config: cfg.Config):
        self.config = config
        self.robot: Optional[FrankaInterface] = None
        self.external_cam = None
        self.wrist_cam = None
        self.left_cam = None
        self.policy = None

        # Safety limits
        self.max_position_delta = 0.05  # 5cm per step
        self.max_rotation_delta = 0.2  # ~11 degrees per step

        # use normal
        self.use_normalize = False

        # Gripper state management
        self.last_gripper_state = None  # Track last gripper command (True=closed, False=open)
        self.gripper_cooldown_steps = 10  # Number of steps to wait before allowing another gripper command
        self.steps_since_last_gripper_cmd = 0  # Counter for cooldown

        # Trajectory tracking for prediction vs actual comparison
        self.chunk_predicted_actions = []  # Store predicted action chunk
        self.chunk_actual_states = []  # Store actual robot states during chunk execution
        self.chunk_initial_state = None  # Initial state at start of chunk

        # Camera display thread
        self._display_thread = None
        self._display_lock = threading.Lock()
        self._current_frame = None
        self._display_running = False
        self._user_quit = False
        
        # Display via shared file
        self._frame_dir = None
        self._viewer_proc = None

    def setup(self):
        """Initialize all components."""
        print("=" * 70)
        print("Setting up Franka Pi05 Policy Runner")
        print("=" * 70)

        # 1. Setup cameras
        self._setup_cameras()

        # 2. Connect to robot
        self._connect_robot()

        # 3. Load policy model (locally on this GPU workstation)
        self._load_policy()
        self._bind_action_executor() ## choose how execute action based on control mode.
        
        # 4. Start camera display if enabled
        if self.config.show_cameras:
            self._start_display_thread()

        print("\n" + "=" * 70)
        print("Setup complete! Ready to run.")
        print("=" * 70 + "\n")

    def _start_display_thread(self):
        """Start the camera display by creating shared frame directory and launching viewer."""
        import os
        import subprocess
        
        # Only start once
        if self._viewer_proc is not None:
            return
        
        self._frame_dir = os.path.expanduser("~/tmp/robot_camera_frames")
        os.makedirs(self._frame_dir, exist_ok=True)
        
        # Launch the camera viewer in a separate process
        viewer_script = os.path.join(os.path.dirname(__file__), "camera_viewer.py")
        if os.path.exists(viewer_script):
            try:
                self._viewer_proc = subprocess.Popen(
                    ["python3", viewer_script],
                    start_new_session=True
                )
                print(f"  ✓ Camera viewer launched (PID: {self._viewer_proc.pid})")
            except Exception as e:
                print(f"  [WARNING] Failed to launch camera viewer: {e}")
                self._viewer_proc = None
        else:
            print(f"  [WARNING] Camera viewer script not found: {viewer_script}")
        
        print(f"  ✓ Camera frames saved to: {self._frame_dir}")

    def _stop_display_thread(self):
        """Stop the camera display."""
        import os
        
        # Terminate viewer process if running
        if hasattr(self, '_viewer_proc') and self._viewer_proc is not None:
            try:
                self._viewer_proc.terminate()
                self._viewer_proc.wait(timeout=2)
            except:
                try:
                    self._viewer_proc.kill()
                except:
                    pass
            self._viewer_proc = None
        
        # Clean up shared files
        try:
            if self._frame_dir:
                frame_file = os.path.join(self._frame_dir, "combined_frame.npy")
                step_file = os.path.join(self._frame_dir, "step.txt")
                if os.path.exists(frame_file):
                    os.remove(frame_file)
                if os.path.exists(step_file):
                    os.remove(step_file)
        except:
            pass
        print("  ✓ Camera display stopped")

    def _display_loop(self):
        """Background thread for displaying camera feeds."""
        # Not used anymore
        pass

    def _update_display(self, external_img_rgb, wrist_img_rgb, left_img_rgb, t_step, resolution=224):
        """Update the display frame by saving to shared file.
        
        Args:
            external_img_rgb: External camera image (RGB)
            wrist_img_rgb: Wrist camera image (RGB)
            left_img_rgb: Left camera image (RGB)
            t_step: Current step number
            resolution: Image resolution (default 224 for model input)
        """
        import os
        
        if self._frame_dir is None:
            return False
        
        try:
            # Process images using resize_with_pad (same as model input)
            # display_external = camera_utils.resize_with_pad(external_img_rgb, resolution, resolution)
            # display_wrist = camera_utils.resize_with_pad(wrist_img_rgb, resolution, resolution)
            # display_left = camera_utils.resize_with_pad(left_img_rgb, resolution, resolution)
            display_external = camera_utils.resize_with_pad(external_img_rgb, 320, 180)
            display_wrist = camera_utils.resize_with_pad(wrist_img_rgb, 320, 180)
            display_left = camera_utils.resize_with_pad(left_img_rgb, 320, 180)
            
            # Convert RGB to BGR for OpenCV display
            display_external = cv2.cvtColor(display_external, cv2.COLOR_RGB2BGR)
            display_wrist = cv2.cvtColor(display_wrist, cv2.COLOR_RGB2BGR)
            display_left = cv2.cvtColor(display_left, cv2.COLOR_RGB2BGR)
            
            # Stack cameras horizontally
            combined_view = np.hstack([display_external, display_wrist, display_left])
            
            # Add labels
            cv2.putText(combined_view, "External", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_view, "Wrist", (resolution + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined_view, "Left", (resolution * 2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Ensure the array is contiguous and uint8
            combined_view = np.ascontiguousarray(combined_view, dtype=np.uint8)
            
            # Save to shared file (atomic write using temp file)
            frame_file = os.path.join(self._frame_dir, "combined_frame.npy")
            temp_file = frame_file + ".tmp.npy"
            np.save(temp_file, combined_view)
            os.replace(temp_file, frame_file)  # Atomic rename
            
            # Save step number
            step_file = os.path.join(self._frame_dir, "step.txt")
            with open(step_file, 'w') as f:
                f.write(str(t_step))
            
            # Debug: print on first frame
            if t_step == 0:
                print(f"    [Display] Saved frame shape: {combined_view.shape}, dtype: {combined_view.dtype}")
                print(f"    [Display] File: {frame_file}")
            
            return True
        except Exception as e:
            print(f"  [WARNING] Failed to update display: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _check_user_quit(self):
        """Check if user requested quit via display window."""
        # With file-based approach, we don't have a direct quit signal
        # User can use Ctrl+C to stop the main program
        return False

    def _setup_cameras(self):
        """Initialize RealSense cameras."""
        print("\n[1/3] Setting up cameras...")

        if self.config.camera.use_mock_cameras:
            print("  Using MOCK cameras (no real hardware)")
            self.external_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.wrist_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
            self.left_cam = camera_utils.MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )

        else:
            # List available cameras
            devices = camera_utils.list_realsense_devices()
            print(f"  Found {len(devices)} RealSense device(s)")

            if not devices:
                raise RuntimeError(
                    "No RealSense cameras found! "
                    "Run 'python camera_utils.py' to list devices, "
                    "or use --use-mock-cameras for testing."
                )

            for i, dev in enumerate(devices):
                print(f"    [{i}] {dev['name']} (Serial: {dev['serial_number']})")

            # Initialize cameras
            print(f"  Initializing external camera (serial: {self.config.camera.external_camera_serial})...")
            self.external_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.external_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )
            print(f"  Initializing wrist camera (serial: {self.config.camera.wrist_camera_serial})...")
            self.wrist_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.wrist_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
                powerline_frequency='60Hz',
            )
            print(f"  Initializing left camera (serial: {self.config.camera.left_camera_serial})...")
            self.left_cam = camera_utils.RealSenseCamera(
                serial_number=self.config.camera.left_camera_serial,
                width=self.config.camera.width,
                height=self.config.camera.height,
                fps=self.config.camera.fps,
            )

        print("  ✓ Cameras ready")

    def _connect_robot(self):
        """Connect to Franka robot via ZeroRPC."""
        if self.config.robot.use_mock_robot:
            print(f"\n[2/3] Using MOCK robot (no real hardware)...")
            self.robot = MockRobot(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
        else:
            print(f"\n[2/3] Connecting to Franka NUC at {self.config.robot.nuc_ip}:{self.config.robot.nuc_port}...")

            try:
                self.robot = FrankaInterface(ip=self.config.robot.nuc_ip, port=self.config.robot.nuc_port)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to robot: {e}")

        # Test connection by getting robot state
        joint_pos = self.robot.get_joint_positions()
        gripper_pos = self.robot.get_gripper_position()

        print(f"  ✓ Connected to robot")
        print(f"    Joint positions: {joint_pos}")
        print(f"    Gripper position: {gripper_pos}")

        # Initialize robot controller based on control mode
        if self.config.robot.control_mode == "joint":
            print("  Starting joint impedance controller...")
            self.robot.start_joint_impedance(Kq=None, Kqd=None)
        elif self.config.robot.control_mode == "eef":
            print("  Starting Cartesian impedance controller...")
            # Stiffness and damping for position (x,y,z) and orientation (rx,ry,rz)
            Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
            Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
            self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)
            # Note: End-effector poses use quaternion format: [x, y, z, qw, qx, qy, qz]
        elif self.config.robot.control_mode == "joint_vel":
            print("  Starting velocity controller...")
            self.robot.start_joint_velocity_control(np.zeros(7, dtype=np.float32))

        print("  ✓ Robot controller started")

    def _load_policy(self):
        """Load pi05_droid policy model on local GPU."""
        print(f"\n[3/3] Loading policy model '{self.config.policy.checkpoint_name}'...")
        print(f"  Checkpoint: {self.config.policy.checkpoint_path}")

        # Get policy config
        print(f"Trying to load policy config... {self.config.policy.checkpoint_name}")
        policy_cfg = _config.get_config(self.config.policy.checkpoint_name)
        print(f"Loaded policy is configured for task: {policy_cfg}")

        # Download checkpoint if needed
        print("  Downloading checkpoint (if not cached)...")
        checkpoint_dir = download.maybe_download(self.config.policy.checkpoint_path)
        print(f"  Using checkpoint from: {checkpoint_dir}")

        # Create policy (loads model weights onto GPU)
        print("  Loading model weights onto GPU...")
        self.policy = policy_config.create_trained_policy(policy_cfg, checkpoint_dir)

        print("  ✓ Policy loaded and ready")

    def _get_robot_joint_state(self) -> np.ndarray:
        """
        Get current robot joint state.

        Returns:
            joint_state (7D): [q1, q2, q3, q4, q5, q6, q7]
        """
        joint_positions = self.robot.get_joint_positions()  # 7D
        return joint_positions.astype(np.float32)

    def _get_gripper_position(self) -> np.ndarray:
        """
        Get current gripper position.

        Returns:
            gripper_width (1D): [width]
        """
        gripper_pos = self.robot.get_gripper_position()[0:1]  # Assuming get_gripper_position returns [width, ...]
        return gripper_pos.astype(np.float32)
    
    def _get_gripper_position_normalised(self) -> np.ndarray:
        """
        Get current gripper position normalized to [0, 1].

        Returns:
            gripper_width_normalized (1D): [width_normalized]
        """
        gripper_pos = self._get_gripper_position()
        # Assuming max gripper width is around 0.08m (80mm), normalize accordingly
        gripper_width_normalized = np.clip((0.08 - gripper_pos) / 0.08, 0.0, 1.0) ## 1.0 means fully closed, 0.0 means fully open. 
        return gripper_width_normalized.astype(np.float32)


    def _get_robot_state(self, target_pose) -> np.ndarray:
        """
        Get current robot state in model format.

        Returns:
            state (8D): [x, y, z, roll, pitch, yaw, gripper_1, gripper_2]
        """
        # Get end-effector pose: [x, y, z, qx, qy, qz, qw]
        if target_pose is not None:
            ee_pose = target_pose

        else:
            ee_pose = self.robot.get_ee_pose()

        # Extract position
        position = ee_pose[:3]

        # Convert quaternion to Euler
        quat = ee_pose[3:]  # [qx, qy, qz, qw] - Franka format
        euler = quaternion_to_euler(quat)  # [roll, pitch, yaw]

        # Get gripper width
        gripper_width = self.robot.get_gripper_position()[0]

        # Convert to symmetric format (matches training data)
        gripper_1 = gripper_width / 2.0
        gripper_2 = -gripper_width / 2.0

        # [x, y, z]  # [roll, pitch, yaw]  # symmetric gripper
        state = np.concatenate([position, euler, [gripper_1, gripper_2]]).astype(np.float32)

        return state

    ## setting the action executor based on control mode
    def _bind_action_executor(self):
        mode = self.config.robot.control_mode
        if mode == "eef":
            self._execute_action = self._execute_action_eef
        elif mode == "joint_vel":
            self._execute_action = self._execute_action_joint_vel
        elif mode == "joint":
            raise NotImplementedError("Joint position control not implemented yet.")
        else:
            raise ValueError(f"Unknown control_mode: {mode}")


    ## method to match original pi0.5 velocity control regression.
    ## main reason is for smoothness.
    def _execute_action_joint_vel(self, pred_action: np.ndarray, current_state: np.ndarray, dt: float):
        """
        Execute ONE action on robot in joint velocity mode.

        Args:
            pred_action (8D): [qdot_1..qdot_7, gripper_binary]
            current_state: Current robot state (raw)
            dt: Control timestep (unused for direct velocity setpoint, kept for symmetry)
        """
        pred_action = np.asarray(pred_action).reshape(-1)
        assert pred_action.shape[0] >= 8, f"Expected 8D action [7 qdot + gripper], got {pred_action.shape}"

        # Parse action
        qdot = pred_action[:7]
        gripper_cmd = pred_action[7]

        # Safety clipping (analogous to delta_pos/delta_rot clipping)
        vmax = getattr(self, "max_joint_velocity", None)
        if vmax is None:
            vmax = getattr(self.config.robot, "max_joint_velocity", 0.5)  # rad/s conservative default

        qdot_clipped = np.clip(qdot, -vmax, vmax)

        if not np.allclose(qdot, qdot_clipped):
            print(f"  [WARNING] Joint velocity clipped for safety!")
            print(f"    qdot: {qdot} → {qdot_clipped}")

        # Send to robot (velocity control setpoint)
        self.robot.update_desired_joint_velocities(qdot_clipped.astype(np.float32))

        # --- Gripper control with state tracking and cooldown (copied structure from your eef method) ---
        # gripper_open = bool(gripper_cmd > 0.045)  # 1.0 means open
        gripper_open = bool(gripper_cmd < 0.5)  # 1.0 means CLOSE, 0.0 means open. following original droid dataset convention. 

        gripper_close = not gripper_open        # Invert for robot API (True=close, False=open)

        # Increment cooldown counter
        self.steps_since_last_gripper_cmd += 1

        # Initialize gripper state on first call
        if self.last_gripper_state is None:
            current_gripper_width = current_state[6:8].mean() * 2  # Convert symmetric format back to width
            self.last_gripper_state = current_gripper_width < GRIPPER_CLOSE_THRESHOLD
            print(
                f"  [GRIPPER] Initialized state: {'CLOSED' if self.last_gripper_state else 'OPEN'} "
                f"(width: {current_gripper_width*1000:.1f}mm)"
            )

        # Only send control command if:
        # 1. State has changed from last commanded state
        # 2. Cooldown period has passed
        # 3. gripper prev cmd success
        if (
            gripper_close != self.last_gripper_state
            and self.steps_since_last_gripper_cmd >= self.gripper_cooldown_steps
            and self.robot.get_gripper_prev_cmd_success()
        ):
            print(
                f"  [GRIPPER] State change detected: "
                f"{'CLOSED' if self.last_gripper_state else 'OPEN'} → "
                f"{'CLOSED' if gripper_close else 'OPEN'}"
            )
            self.robot.control_gripper(gripper_close)
            self.last_gripper_state = gripper_close
            self.steps_since_last_gripper_cmd = 0
        elif gripper_close != self.last_gripper_state:
            print(
                f"  [GRIPPER] State change requested but in cooldown "
                f"({self.steps_since_last_gripper_cmd}/{self.gripper_cooldown_steps} steps)"
            )

        # Return the command we actually sent (analogous to returning target_ee_pose)
        return qdot_clipped.astype(np.float32)



    def _execute_action_eef(self, pred_action: np.ndarray, current_state: np.ndarray, dt: float):
        """
        Execute action on robot.

        Args:
            raw_action (7D): [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper_binary]
            current_state (8D): Current robot state (raw)
            dt: Control timestep
        """
        # Parse action
        delta_pos = pred_action[:3]
        delta_rot = pred_action[3:6]
        gripper_cmd = pred_action[6]

        # Safety clipping
        delta_pos_clipped = np.clip(delta_pos, -self.max_position_delta, self.max_position_delta)
        delta_rot_clipped = np.clip(delta_rot, -self.max_rotation_delta, self.max_rotation_delta)

        if not np.allclose(delta_pos, delta_pos_clipped) or not np.allclose(delta_rot, delta_rot_clipped):
            print(f"  [WARNING] Action clipped for safety!")
            print(f"    Δpos: {delta_pos} → {delta_pos_clipped}")
            print(f"    Δrot: {delta_rot} → {delta_rot_clipped}")

        # Current state
        current_pos = current_state[:3]
        current_euler = current_state[3:6]

        # Compute target
        target_pos = current_pos + delta_pos_clipped
        target_euler = apply_euler_delta(current_euler, delta_rot_clipped, mode="add")
        target_quat = euler_to_quaternion(target_euler)

        # Ensure z-axis is vertical by zeroing out roll and pitch
        # target_euler_vertical = np.array([0.0, 0.0, target_euler[2]])
        # target_quat = euler_to_quaternion(target_euler_vertical)

        # Build target pose
        target_ee_pose = np.concatenate([target_pos, target_quat]).astype(np.float32)

        # Send to robot
        self.robot.update_desired_ee_pose(target_ee_pose)

        # Control gripper with state tracking and cooldown
        gripper_open = bool(gripper_cmd > 0.5)  # 1.0 means open
        gripper_close = not gripper_open  # Invert for robot API (True=close, False=open)

        # Increment cooldown counter
        self.steps_since_last_gripper_cmd += 1

        # Initialize gripper state on first call
        if self.last_gripper_state is None:
            # Get current gripper state from observation
            current_gripper_width = current_state[6:8].mean() * 2  # Convert symmetric format back to width
            self.last_gripper_state = current_gripper_width < GRIPPER_CLOSE_THRESHOLD
            print(
                f"  [GRIPPER] Initialized state: {'CLOSED' if self.last_gripper_state else 'OPEN'} (width: {current_gripper_width*1000:.1f}mm)"
            )

        # Only send control command if:
        # 1. State has changed from last commanded state
        # 2. Cooldown period has passed
        # 3. gripper prev cmd success
        if (
            gripper_close != self.last_gripper_state
            and self.steps_since_last_gripper_cmd >= self.gripper_cooldown_steps
            and self.robot.get_gripper_prev_cmd_success()
        ):
            print(
                f"  [GRIPPER] State change detected: {'CLOSED' if self.last_gripper_state else 'OPEN'} → {'CLOSED' if gripper_close else 'OPEN'}"
            )
            self.robot.control_gripper(gripper_close)
            self.last_gripper_state = gripper_close
            self.steps_since_last_gripper_cmd = 0
        elif gripper_close != self.last_gripper_state:
            # State change requested but cooldown not elapsed
            print(
                f"  [GRIPPER] State change requested but in cooldown ({self.steps_since_last_gripper_cmd}/{self.gripper_cooldown_steps} steps)"
            )

        return target_ee_pose

    def _compare_predicted_vs_actual(self):
        return
        """
        Compare predicted actions vs actual robot states after chunk completion.
        Prints two types of comparisons:
        1. Accumulated poses: Compare target pose from accumulating actions vs actual trajectory
        2. Delta actions: Compare predicted delta actions vs actual deltas in robot state
        """
        if len(self.chunk_predicted_actions) == 0 or len(self.chunk_actual_states) < 2:
            print("  [COMPARISON] Insufficient data for comparison")
            return

        print(f"\n{'=' * 70}")
        print("PREDICTED vs ACTUAL COMPARISON")
        print(f"{'=' * 70}")

        # Method 1: Compare accumulated poses (absolute comparison)
        print("\n[Method 1] Accumulated Pose Comparison:")
        print("  Ground Truth: Target poses from accumulating predicted actions")
        print("  Executed: Actual robot state trajectory\n")

        # Start from initial state and accumulate predicted actions
        accumulated_state = self.chunk_initial_state.copy()
        
        print(f"  {'Step':<6} {'GT Pos (m)':<30} {'Actual Pos (m)':<30} {'Pos Error (mm)':<15}")
        print(f"  {'':6} {'GT Rot (deg)':<30} {'Actual Rot (deg)':<30} {'Rot Error (deg)':<15}")
        print("  " + "-" * 90)

        print(f"len(predicted_actions): {len(self.chunk_predicted_actions)}")
        print(f"len(actual_states): {len(self.chunk_actual_states)}")

        for i in range(len(self.chunk_predicted_actions)):
            # Accumulate predicted action onto accumulated state
            pred_action = self.chunk_predicted_actions[i]
            accumulated_state[:3] += pred_action[:3]  # Position
            accumulated_state[3:6] = apply_euler_delta(accumulated_state[3:6], pred_action[3:6], mode="add")  # Rotation

            # Get actual state
            actual_state = self.chunk_actual_states[i]

            # Compute errors
            pos_error = np.linalg.norm(accumulated_state[:3] - actual_state[:3]) * 1000  # Convert to mm
            rot_error_rad = np.abs(accumulated_state[3:6] - actual_state[3:6])
            rot_error_deg = np.rad2deg(rot_error_rad)

            # Print comparison
            gt_pos_str = f"[{accumulated_state[0]:.4f}, {accumulated_state[1]:.4f}, {accumulated_state[2]:.4f}]"
            actual_pos_str = f"[{actual_state[0]:.4f}, {actual_state[1]:.4f}, {actual_state[2]:.4f}]"
            gt_rot_str = f"[{np.rad2deg(accumulated_state[3]):.1f}, {np.rad2deg(accumulated_state[4]):.1f}, {np.rad2deg(accumulated_state[5]):.1f}]"
            actual_rot_str = f"[{np.rad2deg(actual_state[3]):.1f}, {np.rad2deg(actual_state[4]):.1f}, {np.rad2deg(actual_state[5]):.1f}]"

            print(f"  {i:<6} {gt_pos_str:<30} {actual_pos_str:<30} {pos_error:.2f}")
            print(f"  {'':6} {gt_rot_str:<30} {actual_rot_str:<30} [{rot_error_deg[0]:.2f}, {rot_error_deg[1]:.2f}, {rot_error_deg[2]:.2f}]")

        # Summary statistics for Method 1
        accumulated_states = [self.chunk_initial_state.copy()]
        for pred_action in self.chunk_predicted_actions:
            prev_state = accumulated_states[-1].copy()
            prev_state[:3] += pred_action[:3]
            prev_state[3:6] = apply_euler_delta(prev_state[3:6], pred_action[3:6], mode="add")
            accumulated_states.append(prev_state)
        
        accumulated_states = np.array(accumulated_states[1:])  # Remove initial state, keep predictions
        actual_states = np.array(self.chunk_actual_states)
        
        pos_errors = np.linalg.norm(accumulated_states[:, :3] - actual_states[:, :3], axis=1) * 1000
        rot_errors = np.rad2deg(np.abs(accumulated_states[:, 3:6] - actual_states[:, 3:6]))

        print("\n  Summary Statistics:")
        print(f"    Position Error: Mean={np.mean(pos_errors):.2f}mm, Max={np.max(pos_errors):.2f}mm")
        print(f"    Rotation Error: Mean={np.mean(rot_errors, axis=0)} deg, Max={np.max(rot_errors, axis=0)} deg")

        # Method 2: Compare delta actions (relative comparison)
        print("\n[Method 2] Delta Action Comparison:")
        print("  Ground Truth: Predicted delta actions")
        print("  Executed: Actual deltas in robot state\n")

        print(f"  {'Step':<6} {'Pred ΔPos (mm)':<30} {'Actual ΔPos (mm)':<30} {'Error (mm)':<15}")
        print(f"  {'':6} {'Pred ΔRot (deg)':<30} {'Actual ΔRot (deg)':<30} {'Error (deg)':<15}")
        print("  " + "-" * 90)

        # Prepend initial state to actual states for delta calculation
        all_actual_states = [self.chunk_initial_state] + self.chunk_actual_states

        for i in range(len(self.chunk_predicted_actions)):
            pred_action = self.chunk_predicted_actions[i]
            
            # Calculate actual delta
            prev_actual_state = all_actual_states[i]
            curr_actual_state = all_actual_states[i + 1]
            actual_delta_pos = curr_actual_state[:3] - prev_actual_state[:3]
            actual_delta_rot = curr_actual_state[3:6] - prev_actual_state[3:6]

            # Compute errors
            delta_pos_error = np.linalg.norm(pred_action[:3] - actual_delta_pos) * 1000
            delta_rot_error = np.rad2deg(np.abs(pred_action[3:6] - actual_delta_rot))

            # Print comparison
            pred_pos_str = f"[{pred_action[0]*1000:.2f}, {pred_action[1]*1000:.2f}, {pred_action[2]*1000:.2f}]"
            actual_pos_str = f"[{actual_delta_pos[0]*1000:.2f}, {actual_delta_pos[1]*1000:.2f}, {actual_delta_pos[2]*1000:.2f}]"
            pred_rot_str = f"[{np.rad2deg(pred_action[3]):.2f}, {np.rad2deg(pred_action[4]):.2f}, {np.rad2deg(pred_action[5]):.2f}]"
            actual_rot_str = f"[{np.rad2deg(actual_delta_rot[0]):.2f}, {np.rad2deg(actual_delta_rot[1]):.2f}, {np.rad2deg(actual_delta_rot[2]):.2f}]"

            print(f"  {i:<6} {pred_pos_str:<30} {actual_pos_str:<30} {delta_pos_error:.2f}")
            print(f"  {'':6} {pred_rot_str:<30} {actual_rot_str:<30} [{delta_rot_error[0]:.2f}, {delta_rot_error[1]:.2f}, {delta_rot_error[2]:.2f}]")

        # Summary statistics for Method 2
        actual_deltas_pos = []
        actual_deltas_rot = []
        for i in range(len(self.chunk_predicted_actions)):
            prev_state = all_actual_states[i]
            curr_state = all_actual_states[i + 1]
            actual_deltas_pos.append(curr_state[:3] - prev_state[:3])
            actual_deltas_rot.append(curr_state[3:6] - prev_state[3:6])
        
        actual_deltas_pos = np.array(actual_deltas_pos)
        actual_deltas_rot = np.array(actual_deltas_rot)
        pred_deltas_pos = np.array([a[:3] for a in self.chunk_predicted_actions])
        pred_deltas_rot = np.array([a[3:6] for a in self.chunk_predicted_actions])

        delta_pos_errors = np.linalg.norm(pred_deltas_pos - actual_deltas_pos, axis=1) * 1000
        delta_rot_errors = np.rad2deg(np.abs(pred_deltas_rot - actual_deltas_rot))

        print("\n  Summary Statistics:")
        print(f"    Delta Position Error: Mean={np.mean(delta_pos_errors):.2f}mm, Max={np.max(delta_pos_errors):.2f}mm")
        print(f"    Delta Rotation Error: Mean={np.mean(delta_rot_errors, axis=0)} deg, Max={np.max(delta_rot_errors, axis=0)} deg")

        print(f"\n{'=' * 70}\n")

    def run_rollout(self, instruction: str, max_timesteps: Optional[int] = None):
        """Execute rollout with proper normalization."""
        if max_timesteps is None:
            max_timesteps = self.config.max_timesteps

        print(f"\n{'=' * 70}")
        print(f"TASK: {instruction}")
        print(f"Max timesteps: {max_timesteps}")
        print(f"{'=' * 70}")

        # Reset gripper state tracking for new rollout
        self.last_gripper_state = None
        self.steps_since_last_gripper_cmd = 0

        actions_from_chunk_completed = 0
        pred_action_chunk = None

        video_frames = []
        dt = 1.0 / self.config.robot.control_frequency
        inference_times = []

        print("Running rollout... (Press Ctrl+C to stop)")

        target_pose = None
        try:
            for t_step in range(max_timesteps):
                step_start_time = time.time()

                # 1. Get raw robot state
                # raw_state = self._get_robot_state(target_pose)
                raw_joint_state = self._get_robot_joint_state() ## 7D
                gripper_state = self._get_gripper_position_normalised() ## 1D normalized

                ## compute raw_state, will be concatenation of xyz row pitch yaw and yanzhe's gripper definition. to get the gripper timeout to work. 
                raw_state = self._get_robot_state(target_pose)

                # 2. Capture images
                ret_ext, external_img, _ = self.external_cam.read()
                ret_wrist, wrist_img, _ = self.wrist_cam.read()
                ret_left, left_img, _ = self.left_cam.read()

                if not ret_ext or not ret_wrist or not ret_left:
                    print(f"  [Step {t_step}] Failed to capture images!")
                    break

                external_img_rgb = cv2.cvtColor(external_img, cv2.COLOR_BGR2RGB)
                wrist_img_rgb = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
                left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)

                if self.config.save_video:
                    video_frames.append(external_img_rgb.copy())

                # Display camera feeds if enabled (runs in separate process, started in setup())
                if self.config.show_cameras:
                    # Send RGB frames to display process (same as model input)
                    self._update_display(external_img_rgb, wrist_img_rgb, left_img_rgb, t_step)
                    
                    # Check if user quit via display window
                    if self._check_user_quit():
                        print("\n  User requested quit via display window")
                        break

                # 3. Check against actual chunk length, not config
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.config.policy.open_loop_horizon:
                    # Before starting new chunk, analyze the previous chunk if it exists
                    if actions_from_chunk_completed >= self.config.policy.open_loop_horizon and len(self.chunk_predicted_actions) > 0:
                        self._compare_predicted_vs_actual()
                    
                    # Reset for new chunk
                    actions_from_chunk_completed = 0
                    self.chunk_predicted_actions = []
                    self.chunk_actual_states = []
                    self.chunk_initial_state = raw_joint_state.copy()

                    # RESOLUTION = 256
                    # CROP_CONFIG = {"front": (180, 180, 300), "left": (350, 120, 250)}

                    # x_f, y_f, cs_f = CROP_CONFIG["front"]
                    # front_cropped = external_img_rgb[y_f : y_f + cs_f, x_f : x_f + cs_f]

                    # front_final = camera_utils.resize_with_pad(front_cropped, RESOLUTION, RESOLUTION)
                    # x_l, y_l, cs_l = CROP_CONFIG["left"]
                    # left_cropped = left_img_rgb[y_l : y_l + cs_l, x_l : x_l + cs_l]
                    # left_final = camera_utils.resize_with_pad(left_cropped, RESOLUTION, RESOLUTION)
                    # wrist_final = camera_utils.resize_with_pad(wrist_img_rgb, RESOLUTION, RESOLUTION)

                    # # --- 3. 构建最终的 obs ---
                    # obs = {
                    #     "image": front_final,
                    #     "wrist_image": wrist_final,
                    #     "left_image": left_final,
                    #     "state": raw_state,
                    #     "prompt": instruction,
                    # }

                    # if actions_from_chunk_completed == 0:
                    #     Image.fromarray(front_final).save("./debug_front.png")
                    #     Image.fromarray(left_final).save("./debug_left.png")

                    # TODO: 注意是否 crop & mask
                    RESOLUTION = 224
                    ## it is width and height for camera_utils.resize_with_pad 
                    # obs = {
                    #     "image": camera_utils.resize_with_pad(external_img_rgb, RESOLUTION, RESOLUTION),
                    #     "wrist_image": camera_utils.resize_with_pad(wrist_img_rgb, RESOLUTION, RESOLUTION),
                    #     "left_image": camera_utils.resize_with_pad(left_img_rgb, RESOLUTION, RESOLUTION),
                    #     "state": raw_state,
                    #     "prompt": instruction,
                    # }

                    ## TODO check if this is the correct name mapping for all inputs.
                    obs = {
                            "observation/exterior_image_2_left": camera_utils.resize_with_pad(external_img_rgb, 320, 180),
                            "observation/exterior_image_1_left": camera_utils.resize_with_pad(left_img_rgb, 320, 180),
                            "observation/wrist_image_left": camera_utils.resize_with_pad(wrist_img_rgb, 320, 180),
                            "observation/joint_position": raw_joint_state,
                            "observation/gripper_position": gripper_state,
                            "prompt": SINGLE_TASK,
                    }

                    # Run inference
                    inference_start = time.time()

                    result = self.policy.infer(obs)
                    inference_time = (time.time() - inference_start) * 1000
                    inference_times.append(inference_time)

                    pred_action_chunk = result["actions"]
                    
                    # Store predicted actions for this chunk
                    self.chunk_predicted_actions = pred_action_chunk.copy()

                    print(f"\n  [Step {t_step:3d}] NEW ACTION CHUNK")
                    print(f"    Inference: {inference_time:.1f}ms")
                    print(f"    Chunk size: {len(pred_action_chunk)} actions")
                    print(f"    First action: {pred_action_chunk[0]}")
                    # print(f"      Δpos: {pred_action_chunk[0, :3]}")
                    # print(f"      Δrot (deg): {np.rad2deg(pred_action_chunk[0, 3:6])}")
                    # print(f"      Gripper: {pred_action_chunk[0, 6]:.3f}")

                # 4. Execute denormalized action
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1  # Small delay to ensure timing accuracy

                # Print periodic updates
                if t_step % 10 == 0 or actions_from_chunk_completed == 1:
                    print(f"  [Step {t_step:3d}] Action {actions_from_chunk_completed}/{len(pred_action_chunk)}")
                    print(
                        f"      Gripper cmd: {pred_action_chunk[0, 7]:.3f} -> {'OPEN' if pred_action_chunk[0, 7] > 0.5 else 'CLOSE'}"
                    )

                target_pose = self._execute_action(action, raw_state, dt)

                # 5. Regulate timing
                elapsed = time.time() - step_start_time
                # print(f"    Step time: {elapsed*1000:.1f}ms (target: {dt*1000:.1f}ms)")
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                elif t_step % 10 == 0:
                    print(f"  [WARNING] Step {t_step} overran: {elapsed*1000:.1f}ms (target: {dt*1000:.1f}ms)")

                # Store actual state after action execution
                # Get the actual state after execution (with small delay to let robot settle)
                # time.sleep(0.01)  # Small delay to ensure robot state is updated
                actual_state_after = self._get_robot_state(target_pose)
                self.chunk_actual_states.append(actual_state_after.copy())
                print(f"Current chunk actual states stored: {len(self.chunk_actual_states)}")

        except KeyboardInterrupt:
            print("\n\n  Rollout interrupted by user")

        finally:
            # Print comparison for the last chunk if it wasn't completed
            if len(self.chunk_predicted_actions) > 0 and len(self.chunk_actual_states) > 0:
                print("\n[Final Chunk Comparison]")
                self._compare_predicted_vs_actual()
            
            print(f"\n{'=' * 70}")
            print("ROLLOUT COMPLETE")
            print(f"{'=' * 70}")
            print(f"  Steps: {len(video_frames)}")
            if inference_times:
                print(f"  Inference: {np.mean(inference_times):.1f}ms ± {np.std(inference_times):.1f}ms")
                print(f"    Min: {np.min(inference_times):.1f}ms, Max: {np.max(inference_times):.1f}ms")

            if self.config.save_video and video_frames:
                self._save_video(video_frames, instruction)

            if self.config.show_cameras:
                self._stop_display_thread()

    def _save_video(self, frames: list[np.ndarray], instruction: str):
        """Save rollout video."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_instruction = "".join(c if c.isalnum() else "_" for c in instruction[:30])
        filename = f"rollout_{timestamp}_{safe_instruction}.mp4"

        print(f"\nSaving video to {filename}...")
        try:
            clip = ImageSequenceClip(frames, fps=self.config.video_fps)
            clip.write_videofile(filename, codec="libx264", verbose=False, logger=None)
            print(f"  ✓ Video saved: {filename}")
        except Exception as e:
            print(f"  ✗ Failed to save video: {e}")

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        if self.robot is not None:
            try:
                self.robot.terminate_current_policy()
                self.robot.close()
                print("  ✓ Robot disconnected")
            except Exception as e:
                print(f"  ✗ Error disconnecting robot: {e}")

        if self.external_cam is not None:
            self.external_cam.release()

        if self.wrist_cam is not None:
            self.wrist_cam.release()

        if self.left_cam is not None:
            self.left_cam.release()

        # cv2.destroyAllWindows()
        print("  ✓ Cleanup complete")


def main(
    # Camera settings
    external_camera: Optional[str] = None,
    wrist_camera: Optional[str] = None,
    left_camera: Optional[str] = None,
    use_mock_cameras: bool = False,
    # Robot settings
    nuc_ip: str = "192.168.1.143",
    nuc_port: int = 4242,
    control_mode: Literal["joint", "eef", "joint_vel"] = "joint",
    use_mock_robot: bool = False,
    # Policy settings
    checkpoint_name: str = "pi05_droid",
    checkpoint_path: str = "gs://openpi-assets/checkpoints/pi05_droid",
    # Rollout settings
    max_timesteps: int = 600,
    show_cameras: bool = True,
    save_video: bool = True,
):
    """
    Run pi05_droid policy on Franka robot.

    Args:
        external_camera: Serial number of external RealSense camera
        wrist_camera: Serial number of wrist RealSense camera
        left_camera: Serial number of left RealSense camera
        use_mock_cameras: Use mock cameras for testing (no real hardware)
        nuc_ip: IP address of Franka NUC running Polymetis
        nuc_port: Port of ZeroRPC server on NUC
        control_mode: Control mode ('joint' or 'eef')
        use_mock_robot: Use mock robot for testing (no real robot hardware)
        checkpoint_name: Name of policy checkpoint
        checkpoint_path: Path to policy checkpoint
        max_timesteps: Maximum steps per rollout
        show_cameras: Display camera feeds during rollout
        save_video: Save rollout video
    """

    # Build configuration
    config = cfg.Config(
        camera=cfg.CameraConfig(
            external_camera_serial=external_camera,
            wrist_camera_serial=wrist_camera,
            left_camera_serial=left_camera,
            use_mock_cameras=use_mock_cameras,
        ),
        robot=cfg.RobotConfig(
            nuc_ip=nuc_ip,
            nuc_port=nuc_port,
            control_mode=control_mode,
            use_mock_robot=use_mock_robot,
        ),
        policy=cfg.PolicyConfig(
            checkpoint_name=checkpoint_name,
            checkpoint_path=checkpoint_path,
        ),
        max_timesteps=max_timesteps,
        show_cameras=show_cameras,
        save_video=save_video,
    )

    # Create runner
    runner = FrankaPolicyRunner(config)

    try:
        # Setup all components
        runner.setup()

        # Main loop: ask for instructions and run rollouts
        while True:
            print("\n" + "=" * 70)
            # instruction = input("Enter task instruction (or 'quit' to exit): ").strip()

            # if instruction.lower() in ["quit", "exit", "q"]:
            #     break

            # if not instruction:
            #     print("Please enter a valid instruction.")
            #     continue
            instruction = "Place the orange block on top of the white block"

            # Run rollout
            runner.run_rollout(instruction)

            # Ask if user wants to continue
            continue_prompt = input("\nRun another task? (y/n): ").strip().lower()
            if continue_prompt != "y":
                break

    except KeyboardInterrupt:
        print("\n\nShutdown requested (Ctrl+C)")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        runner.cleanup()
        print("\nGoodbye!")


if __name__ == "__main__":
    tyro.cli(main)