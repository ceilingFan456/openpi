"""Utilities for managing RealSense cameras."""

import time
from typing import Optional
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("WARNING: pyrealsense2 not installed. Install with: pip install pyrealsense2")


class RealSenseCamera:
    """Wrapper for Intel RealSense camera with retry logic."""
    
    def __init__(
        self, 
        serial_number: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_depth: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        powerline_frequency: str = "50Hz",
    ):
        """
        Initialize RealSense camera with automatic retry on failure.
        
        Args:
            serial_number: Camera serial number (None for any camera)
            width: Image width
            height: Image height
            fps: Frames per second
            enable_depth: Whether to enable depth stream
            max_retries: Maximum number of initialization attempts
            retry_delay: Delay between retries in seconds
        """
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 is required. Install with: pip install pyrealsense2")
        
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_depth = enable_depth
        self.powerline_frequency = powerline_frequency
        self.pipeline = None
        self.config = None
        
        # Try to initialize camera with retries
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Starting RealSense camera {serial_number or 'default'} (attempt {attempt}/{max_retries})...")
                
                # Reset pipeline and config for each attempt
                if self.pipeline is not None:
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                
                self.pipeline = rs.pipeline()
                self.config = rs.config()
                
                if serial_number:
                    self.config.enable_device(serial_number)
                
                self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                if enable_depth:
                    self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                
                # Start pipeline
                self.pipeline.start(self.config)
                
                # Warm up camera - try to read a few frames
                print(f"  Warming up camera...")
                for i in range(3):
                    self.pipeline.wait_for_frames(timeout_ms=1000)
                
                print(f"✓ RealSense camera {serial_number or 'default'} ready!")
                return  # Success!
                
            except Exception as e:
                print(f"✗ Attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"\n{'='*60}")
                    print(f"CAMERA INITIALIZATION FAILED")
                    print(f"{'='*60}")
                    print(f"Camera: {serial_number or 'default'}")
                    print(f"All {max_retries} attempts failed.")
                    print(f"\nTroubleshooting steps:")
                    print(f"1. Unplug and replug the camera USB cable")
                    print(f"2. Try a different USB 3.0 port (preferably directly on motherboard)")
                    print(f"3. Check USB cable quality (use USB 3.0 cables)")
                    print(f"4. Reset USB bus: sudo usbreset")
                    print(f"5. Check camera power: lsusb | grep Intel")
                    print(f"6. See CAMERA_TROUBLESHOOTING.md for permanent solutions")
                    print(f"{'='*60}\n")
                    raise RuntimeError(f"Failed to initialize RealSense camera after {max_retries} attempts")
    
    def read(self) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frame from camera.
        
        Returns:
            (success, color_image, depth_image)
            depth_image is None if depth is not enabled
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return False, None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            
            depth_image = None
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
            
            return True, color_image, depth_image
            
        except Exception as e:
            print(f"Error reading from camera: {e}")
            return False, None, None
    
    def release(self):
        """Stop the camera pipeline."""
        try:
            self.pipeline.stop()
            print(f"RealSense camera {self.serial_number or 'default'} stopped.")
        except Exception as e:
            print(f"Error stopping camera: {e}")


class MockCamera:
    """Mock camera for testing without real hardware."""
    
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        **kwargs
    ):
        self.width = width
        self.height = height
        print(f"Using mock camera ({width}x{height})")
    
    def read(self) -> tuple[bool, np.ndarray, None]:
        """Generate random test image."""
        # Create a colorful test pattern
        img = np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8)
        time.sleep(0.03)  # Simulate 30 FPS
        return True, img, None
    
    def release(self):
        """No-op for mock camera."""
        print("Mock camera released.")


def list_realsense_devices() -> list[dict]:
    """
    List all connected RealSense devices.
    
    Returns:
        List of dicts with keys: 'serial_number', 'name', 'firmware_version'
    """
    if not REALSENSE_AVAILABLE:
        print("pyrealsense2 not available")
        return []
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    device_list = []
    for dev in devices:
        ## wrist camera needs to be reset before using. 
        if dev.get_info(rs.camera_info.serial_number) == '218622273043':
            dev.hardware_reset()
            print("Waiting 5 seconds to reset wrist camera and wait for it to reboot...")
            time.sleep(5)

        info = {
            'serial_number': dev.get_info(rs.camera_info.serial_number),
            'name': dev.get_info(rs.camera_info.name),
            'firmware_version': dev.get_info(rs.camera_info.firmware_version),
        }
        device_list.append(info)
    
    return device_list


def resize_with_pad(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Resize image with padding to maintain aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized and padded image
    """
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


if __name__ == "__main__":
    # Test script
    print("Listing RealSense devices:")
    devices = list_realsense_devices()
    
    if not devices:
        print("No RealSense devices found!")
    else:
        for i, dev in enumerate(devices):
            print(f"\nDevice {i}:")
            print(f"  Serial: {dev['serial_number']}")
            print(f"  Name: {dev['name']}")
            print(f"  Firmware: {dev['firmware_version']}")
