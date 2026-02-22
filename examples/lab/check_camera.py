import pyrealsense2 as rs

def check_camera_speeds():
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if not devices:
        print("No devices found.")
        return

    print(f"{'Serial Number':<15} {'Name':<20} {'USB Type':<10}")
    print("-" * 50)
    
    for dev in devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        # Check if the device is connected via USB 3 or 2
        usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
        
        status = "OK" if "3." in usb_type else "WARNING: SLOW (USB 2.0)"
        print(f"{sn:<15} {name:<20} {usb_type:<10} {status}")

import pyrealsense2 as rs
import time

def test_specific_camera(target_serial):
    print(f"\n--- Testing D405 (Serial: {target_serial}) ---")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(target_serial)
    
    # D405 specific: Default to a stable resolution
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        print("Starting pipeline...")
        pipeline.start(config)
        print("Pipeline started. Waiting for frames...")
        
        # Try to fetch 10 frames to confirm stability
        for i in range(10):
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            if frames:
                print(f"Frame {i+1} received!")
            time.sleep(0.1)
            
        print("SUCCESS: Camera is streaming correctly.")
        
    except Exception as e:
        print(f"FAILURE: Could not get frames. Error: {e}")
        
    finally:
        pipeline.stop()
        print("Pipeline closed.")

def reset_and_test_d405(target_serial):
    ctx = rs.context()
    devices = ctx.query_devices()
    d405_dev = None

    for dev in devices:
        if dev.get_info(rs.camera_info.serial_number) == target_serial:
            d405_dev = dev
            break

    if d405_dev:
        print(f"Found D405. Sending Hardware Reset...")
        d405_dev.hardware_reset()
        print("Waiting 5 seconds for camera to reboot...")
        time.sleep(5) 
    else:
        print("D405 not found to reset.")
        return

    # Now try to start the pipeline again
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(target_serial)
    # D405 specific: use basic settings
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
        print("Pipeline started successfully after reset!")
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        print("Success! Received frames.")
    except Exception as e:
        print(f"Still failing: {e}")
    finally:
        try: pipeline.stop()
        except: pass

if __name__ == "__main__":
    check_camera_speeds()
    
    # Call the test
    # test_specific_camera('218622273043')
    # test_specific_camera('317222075319')
    # test_specific_camera('327122079691')
    test_specific_camera('336222073740')
    # reset_and_test_d405('218622273043')