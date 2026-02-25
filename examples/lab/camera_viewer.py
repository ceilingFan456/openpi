#!/usr/bin/env python3
"""
Standalone camera viewer that reads frames from shared memory.
Run this script separately to view camera feeds without blocking the main program.

Usage:
    python camera_viewer.py
"""

import cv2
import numpy as np
import os
import time
import sys

FRAME_DIR = os.path.expanduser("~/tmp/robot_camera_frames")
FRAME_FILE = os.path.join(FRAME_DIR, "combined_frame.npy")
STEP_FILE = os.path.join(FRAME_DIR, "step.txt")

def main():
    print("=" * 50)
    print("Robot Camera Viewer")
    print("=" * 50)
    print(f"Watching: {FRAME_FILE}")
    print("Press 'q' to quit")
    print("=" * 50)
    
    cv2.namedWindow("Robot Cameras", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot Cameras", 224 * 3, 224)  # 3 cameras at 224x224 each
    
    last_mtime = 0
    frame_count = 0
    
    # Show a placeholder while waiting for frames
    placeholder = np.zeros((224, 224 * 3, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for frames...", (200, 112), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Robot Cameras", placeholder)
    cv2.waitKey(100)
    
    while True:
        try:
            # Check if frame file exists and has been updated
            if os.path.exists(FRAME_FILE):
                try:
                    mtime = os.path.getmtime(FRAME_FILE)
                except:
                    mtime = 0
                
                if mtime > last_mtime:
                    last_mtime = mtime
                    
                    # Load the frame with error handling
                    try:
                        frame = np.load(FRAME_FILE, allow_pickle=False)
                        
                        if frame is not None and frame.size > 0:
                            # Read step number if available
                            step = "?"
                            if os.path.exists(STEP_FILE):
                                try:
                                    with open(STEP_FILE, 'r') as f:
                                        step = f.read().strip()
                                except:
                                    pass
                            
                            # Add step info
                            cv2.putText(frame, f"Step: {step}", (10, frame.shape[0] - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            cv2.imshow("Robot Cameras", frame)
                            frame_count += 1
                            
                            if frame_count % 100 == 0:
                                print(f"  Displayed {frame_count} frames")
                    except Exception as e:
                        print(f"  Error loading frame: {e}")
            
            # Process GUI events (non-blocking)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("\nUser pressed 'q', exiting...")
                break
                
        except KeyboardInterrupt:
            print("\nInterrupted, exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
    
    cv2.destroyAllWindows()
    print("Camera viewer closed.")

if __name__ == "__main__":
    main()
