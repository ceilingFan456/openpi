"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

from pickle import LIST
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import os
import h5py
import numpy as np
from PIL import Image

REPO_NAME = "ceilingfan456/lab_data_test"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_DIR_PATH = "/home/t-qimhuang/disk2/labdata_test"
LIST_OF_TASK_DESCRIPTIONS = [
    "Place the orange test tube into the cup",
]

## copied from convert_droid_data_to_lerobot.py
def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    print(f"Converting raw data from {data_dir} to LeRobot dataset at {output_path}")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    # dataset = LeRobotDataset.create(
    #     repo_id=REPO_NAME,
    #     robot_type="panda",
    #     fps=10,
    #     features={
    #         "image": {
    #             "dtype": "image",
    #             "shape": (256, 256, 3),
    #             "names": ["height", "width", "channel"],
    #         },
    #         "wrist_image": {
    #             "dtype": "image",
    #             "shape": (256, 256, 3),
    #             "names": ["height", "width", "channel"],
    #         },
    #         "state": {
    #             "dtype": "float32",
    #             "shape": (8,),
    #             "names": ["state"],
    #         },
    #         "actions": {
    #             "dtype": "float32",
    #             "shape": (7,),
    #             "names": ["actions"],
    #         },
    #     },
    #     image_writer_threads=10,
    #     image_writer_processes=5,
    # )

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    ## code for the original Libero dataset loading
    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    # for raw_dataset_name in RAW_DATASET_NAMES:
    #     raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
    #     for episode in raw_dataset:
    #         for step in episode["steps"].as_numpy_iterator():
    #             dataset.add_frame(
    #                 {
    #                     "image": step["observation"]["image"],
    #                     "wrist_image": step["observation"]["wrist_image"],
    #                     "state": step["observation"]["state"],
    #                     "actions": step["action"],
    #                     "task": step["language_instruction"].decode(),
    #                 }
    #             )
    #         dataset.save_episode()
    
    ## find all the task directories inside the RAW_DATASET_DIR_PATH
    
    task_dir_paths = [
        os.path.join(RAW_DATASET_DIR_PATH, name)
        for name in os.listdir(RAW_DATASET_DIR_PATH)
        if os.path.isdir(os.path.join(RAW_DATASET_DIR_PATH, name))
    ]

    assert len(task_dir_paths) == len(LIST_OF_TASK_DESCRIPTIONS), "Number of task directories does not match number of task descriptions"
    
    for task_idx, task_dir_path in enumerate(task_dir_paths):
        ## the task directory has all the videos stored in the videos. subdirectory
        ## and motions stored in all the .hdf5 files 
        video_dir_path = os.path.join(task_dir_path, "videos")
        ## count the number of HDF5 files in the task directory
        hdf5_file_paths = [
            os.path.join(task_dir_path, name)
            for name in os.listdir(task_dir_path)
            if name.endswith(".hdf5")
        ]
        
        for episode_idx, hdf5_file_path in enumerate(hdf5_file_paths):
            ## read the motion data from the hdf5 file
            ## timestamps inside [DATASET] timestamp
            with h5py.File(hdf5_file_path, "r") as f:
                timestamps = f["timestamp"] ## shape (T,)
                joint_poss = f["observations/joint_pos"][:, :7] ## shape (T, 7), after cutting out from (T, 9) ##  [j0, j1, j2, j3, j4, j4, j5, gripper L, gripper R]
                gripper_poss = f["observations/joint_pos"][:, 7:8] ## taking the gripper L or gripper R. ## shape (T, 1)
                joint_vels = f["observations/joint_vel"][:, :8] ## shape (T, 8). original is (T, 9) we just remove the R gripper   
                wrist_images = f["observations/images/camera_wrist_color"] ## (287, 480, 640, 3)
                front_images = f["observations/images/camera_front_color"] ## (287, 480, 640, 3)
                left_images = f["observations/images/camera_left_color"] ## (287, 480, 640, 3)
                
                num_frames = timestamps.shape[0]

                for frame_idx in range(num_frames):
                    # Read images
                    exterior_image_1 = front_images[frame_idx]  # (480, 640, 3)
                    exterior_image_2 = left_images[frame_idx]  # (480, 640, 3)
                    wrist_image = wrist_images[frame_idx]  # (480, 640, 3)

                    # Resize images to (180, 320, 3)
                    import cv2 ## cv2 uses cartesian order for width and height, which is x and y or width and height.

                    exterior_image_1_resized = cv2.resize(exterior_image_1, (320, 180))
                    exterior_image_2_resized = cv2.resize(exterior_image_2, (320, 180))
                    wrist_image_resized = cv2.resize(wrist_image, (320, 180))

                    # Read proprio and actions
                    joint_position = joint_poss[frame_idx]  # (7,)
                    gripper_position = gripper_poss[frame_idx]  # (1,)
                    actions = joint_vels[frame_idx]  # (8,)
                    
                    # print("joint_position", type(joint_poss[frame_idx]), np.shape(joint_poss[frame_idx]))
                    # print("gripper_position", type(gripper_poss[frame_idx]), np.shape(gripper_poss[frame_idx]))
                    # print("actions", type(joint_vels[frame_idx]), np.shape(joint_vels[frame_idx]))
                    # print("exterior_image_1_resized", type(exterior_image_1_resized), np.shape(exterior_image_1_resized))
                    # print("exterior_image_2_resized", type(exterior_image_2_resized), np.shape(exterior_image_2_resized))
                    # print("wrist_image_resized", type(wrist_image_resized), np.shape(wrist_image_resized))
                    # print("task description", LIST_OF_TASK_DESCRIPTIONS[task_idx])

                    dataset.add_frame(
                        {
                            "exterior_image_1_left": exterior_image_1_resized,
                            "exterior_image_2_left": exterior_image_2_resized,
                            "wrist_image_left": wrist_image_resized,
                            "joint_position": joint_position,
                            "gripper_position": gripper_position,
                            "actions": actions,
                            "task": LIST_OF_TASK_DESCRIPTIONS[task_idx],
                        }
                    )
                
                dataset.save_episode()
                print(f"Added episode {episode_idx} from task {task_idx} with {num_frames} frames.")

            ## end of processing for each episode
        ## end of processing for each task.
                    
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["panda"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
