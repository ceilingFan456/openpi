#!/bin/bash

echo "checkpoints folder is structured as /config_name/experiment_name/checkpoint_files."

## pi05_lab_finetune_orange_cube_single_point_3_views is the first attempt to check three views
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point --exp-name=pi05_lab_finetune_orange_cube_single_point_3_views --overwrite

## old_pi05_lab_finetune
## is the one that works somewhat the best with testtube dataset

## pi05_lab_finetune_orange_cube_single_point
## is the base line for danze orange cube 25. has been uploaded to huggingface.