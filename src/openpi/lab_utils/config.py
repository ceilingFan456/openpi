"""Configuration for Franka robot with pi05 policy."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class CameraConfig:
    """Configuration for cameras."""

    # Camera serial numbers - find these by running: python camera_utils.py
    external_camera_serial: Optional[str] = None
    wrist_camera_serial: Optional[str] = None
    left_camera_serial: Optional[str] = None

    # Camera settings
    width: int = 640
    height: int = 480
    fps: int = 30

    # Use mock cameras for testing without hardware
    use_mock_cameras: bool = False


@dataclasses.dataclass
class RobotConfig:
    """Configuration for robot connection."""

    # NUC connection (your Polymetis ZeroRPC server)
    nuc_ip: str = "192.168.1.143"
    nuc_port: int = 4242

    # Control mode: 'joint' for joint control, 'eef' for end-effector control
    control_mode: str = "eef"  # We uses end-effector control

    # Control frequency (Hz) - We uses 10 Hz
    control_frequency: int = 10

    # Safety limits
    max_joint_velocity: float = 1.0  # rad/s
    max_gripper_width: float = 0.085  # meters (Franka max)

    # Use mock robot for testing without hardware
    use_mock_robot: bool = False


@dataclasses.dataclass
class PolicyConfig:
    """Configuration for policy inference."""

    checkpoint_name: str = "pi0_build_left150_hor16_4_step20k_b16"
    checkpoint_path: str = "/home/showlab/VLASafety/openpi-main/examples/ckpt/pi0_build_left150_hor16_4_step20k_b16/20000"

    # TODO: Action execution
    action_horizon: int = 4
    open_loop_horizon: int = 4

    # Default prompt if none provided
    default_prompt: Optional[str] = None


@dataclasses.dataclass
class Config:
    """Main configuration."""

    camera: CameraConfig = dataclasses.field(default_factory=CameraConfig)
    robot: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    policy: PolicyConfig = dataclasses.field(default_factory=PolicyConfig)

    # TODO: Rollout settings
    max_timesteps: int = 600

    # Visualization
    show_cameras: bool = True
    save_video: bool = True
    video_fps: int = 10


# Example configurations


def get_default_config() -> Config:
    """Get default configuration - UPDATE WITH YOUR CAMERA SERIALS."""
    return Config(
        camera=CameraConfig(
            external_camera_serial=None,  # Set your camera serial here
            wrist_camera_serial=None,  # Set your camera serial here
            use_mock_cameras=False,  # Set to True for testing without cameras
        ),
        robot=RobotConfig(
            nuc_ip="192.168.1.143",
            nuc_port=4242,
        ),
    )


def get_test_config() -> Config:
    """Get test configuration with mock hardware."""
    return Config(
        camera=CameraConfig(
            use_mock_cameras=True,
        ),
        robot=RobotConfig(
            nuc_ip="192.168.1.143",
            nuc_port=4242,
        ),
        max_timesteps=100,
    )
