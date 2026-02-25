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

## use below to download checkpoints
git clone https://huggingface.co/ceilingfan456/lab_training_orange_cube_single_point/


## upload trained model 
## remember to delete train_states/, weights is ~12GB.
## will upload everything inside <exp_name>, meaning will see 
## huggingface_model_path/1999
## so it might be better to save the config name inside.
hf upload ceilingfan456/lab_training_orange_cube_single_point <exp_name>

## download model from hugging face, need git lfs
git clone https://huggingface.co/ceilingfan456/lab_training_orange_cube/