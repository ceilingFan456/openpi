import h5py
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def convert_sim_to_hdf5(npz_path, video_path, output_hdf5, fps=15.0):
    """
    Conversion logic for a single episode: 
    Adapts single camera view and complements with dummy views.
    """
    # 1. Load NPZ numerical data
    data = np.load(npz_path)
    ee_pose = data['robot_eef']          # Shape: (N, 7) -> [x, y, z, qx, qy, qz, qw]
    gripper = data['robot_gripper']      # Shape: (N,)
    
    # Combine into 8D format required by conversion script: [pos, quat, gripper]
    combined_ee_pose = np.hstack([ee_pose, gripper.reshape(-1, 1)]).astype(np.float32)
    N = len(combined_ee_pose)
    
    # 2. Extract image frames from the single video (mapped to camera_left_color)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file missing: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    left_frames = []
    while len(left_frames) < N:
        ret, frame = cap.read()
        if not ret:
            break
        # Sim video is usually BGR, convert to RGB for LeRobot convention
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        left_frames.append(frame_rgb)
    cap.release()
    
    # Ensure length alignment
    actual_frames = np.array(left_frames[:N], dtype=np.uint8)
    N = len(actual_frames) # Use actual extracted frame count for final alignment
    
    # 3. Create dummy data (black frames) to satisfy LeRobot conversion script requirements
    # Assumes same resolution as the left camera view
    H, W, C = actual_frames.shape[1:]
    dummy_frames = np.zeros((N, H, W, C), dtype=np.uint8)

    # 4. Write to HDF5
    with h5py.File(output_hdf5, 'w') as f:
        # Logical timestamps based on 15fps
        f.create_dataset("timestamp", data=np.arange(N) / fps)
        
        obs = f.create_group("observations")
        # Align trajectory length with actual image frames
        obs.create_dataset("ee_pose", data=combined_ee_pose[:N])
        
        img_group = obs.create_group("images")
        # Core view: store into camera_left_color
        img_group.create_dataset("camera_left_color", data=actual_frames)
        # Placeholder views: store black frames to prevent script errors
        img_group.create_dataset("camera_front_color", data=dummy_frames)
        img_group.create_dataset("camera_wrist_color", data=dummy_frames)
        
        # Store joint positions if available in NPZ
        if 'robot_joint_pos' in data:
            obs.create_dataset("joint_pos", data=data['robot_joint_pos'][:N].astype(np.float32))

def main():
    parser = argparse.ArgumentParser(description="Batch convert Sim NPZ and Single Video to HDF5")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="./rendered_videos_and_actions",
        help="Root directory containing numbered episode folders"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Target directory to save generated episode_X.hdf5 files"
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Simulation frame rate")
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Find and sort all numeric subdirectories
    sub_dirs = [d for d in os.listdir(args.input_dir) if d.isdigit()]
    sub_dirs = sorted(sub_dirs, key=int)

    if not sub_dirs:
        print(f"No numeric folders found in {args.input_dir}")
        return

    print(f"Found {len(sub_dirs)} episodes. Starting conversion...")

    for folder_name in tqdm(sub_dirs, desc="Processing Episodes"):
        try:
            episode_path = os.path.join(args.input_dir, folder_name)
            
            # Define file paths within the episode folder
            npz_file = os.path.join(episode_path, "robot_traj_Panda_single_arm.npz")
            video_file = os.path.join(episode_path, "video_overlay_Panda_single_arm.mp4")
            
            # Set output filename
            output_hdf5 = os.path.join(args.output_dir, f"episode_{folder_name}.hdf5")

            convert_sim_to_hdf5(npz_file, video_file, output_hdf5, fps=args.fps)

        except Exception as e:
            print(f"\n❌ Error in folder {folder_name}: {e}")
            continue

    print(f"\n✅ Batch conversion complete. Files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

# python convert_masquerade_npz_to_hdf5.py \
#   --input_dir ./rendered_videos_and_actions \
#   --output_dir /path/to/hdf5_output \
#   --fps 15.0