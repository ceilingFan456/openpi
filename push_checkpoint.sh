#!/bin/bash

sudo apt-get install git-lfs -y  # or brew install git-lfs
git lfs install

git clone https://huggingface.co/ceilingfan456/lab_training_orange_cube_single_point_wrong_gripper_state
cp -r /home/t-qimhuang/code/openpi/checkpoints/pi05_lab_finetune_orange_cube_single_point/* lab_training_orange_cube_single_point/
cd lab_training_orange_cube_single_point
git add .
git commit -m "upload model"
git push


## or use hf upload
hf upload ceilingfan456/lab_training_orange_cube_single_point_wrong_gripper_state .