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
    joint_pos = data['robot_joint_pos']  # Shape: (N, 7)

    # print(f"loaded npz data has keys {data.keys()}") ## keys are {robot_eef, robot_gripper, robot_joint_pos}
    # print(f"gripper data is {gripper}") ## everything is 0.08, doesnt have gripper closing signal. 
 
    print(f"ee_pose shape: {ee_pose.shape}, gripper shape: {gripper.shape}, joint_pos shape: {joint_pos.shape}")
    
    # Combine into 8D format required by conversion script: [pos, quat, gripper]
    combined_ee_pose = np.hstack([ee_pose, gripper.reshape(-1, 1)]).astype(np.float32)
    N = len(combined_ee_pose)

    ## Combine into 9D format to be [7 joint pose, gripper left and right]
    gripper_left = gripper.copy()
    gripper_right = gripper.copy()
    combined_joint_pos = np.hstack([joint_pos, gripper_left.reshape(-1, 1), gripper_right.reshape(-1, 1)]).astype(np.float32)

    ## joint action is different from gripper state
    joint_actions = gripper.copy() if "joint_action" not in data else data["joint_action"]

    ## compute estimated joint velocities by finite differencing the joint positions
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / (1.0 / fps)  # Assuming constant time step based on fps
    ## set the last timestamp to be 0 
    joint_vel[-1] = 0.0
    
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
        f.create_dataset("joint_action", data=joint_actions[:N])
        
        obs = f.create_group("observations")
        # Align trajectory length with actual image frames
        obs.create_dataset("ee_pose", data=combined_ee_pose[:N])
        obs.create_dataset("full_joint_pos", data=combined_joint_pos[:N])
        obs.create_dataset("joint_pos", data=combined_joint_pos[:N])
        obs.create_dataset("joint_vel", data=joint_vel[:N])
        # obs.create_dataset("gripper", data=gripper[:N])
        
        
        img_group = obs.create_group("images")
        # Core view: store into camera_left_color
        img_group.create_dataset("camera_left_color", data=actual_frames)
        # Placeholder views: store black frames to prevent script errors
        img_group.create_dataset("camera_front_color", data=actual_frames)
        img_group.create_dataset("camera_wrist_color", data=actual_frames)
        
        

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
#   --input_dir /home/t-qimhuang/disk/datasets/rendered_videos_and_actions_02_09 \
#   --output_dir /home/t-qimhuang/disk/datasets/rendered_videos_and_actions_02_09_hdf5 \
#   --fps 15.0