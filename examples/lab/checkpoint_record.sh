#!/bin/bash

echo "checkpoints folder is structured as /config_name/experiment_name/checkpoint_files."

## pi05_lab_finetune_orange_cube_single_point_single_base_view
## uses base view/left camera as the only visual input. 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_single_base_view --exp-name=pi05_lab_finetune_orange_cube_single_point_single_base_view --overwrite


## pi05_lab_finetune_orange_cube_single_point_3_views is the first attempt to check three views
## uses left view as the base view and front camera as the right wrist camera. 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point --exp-name=pi05_lab_finetune_orange_cube_single_point_3_views --overwrite

## old_pi05_lab_finetune
## is the one that works somewhat the best with testtube dataset

## pi05_lab_finetune_orange_cube_single_point
## is the base line for danze orange cube 25. has been uploaded to huggingface.
## but this one only make use of the left and wrist camera. left camera is the one near base of our robot. 