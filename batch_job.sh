#!/bin/bash

## Run all experiments one by one

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_three_views \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_three_views --overwrite

# rm -rf ./checkpoints/pi05_lab_finetune_orange_cube_single_point_three_views/pi05_lab_finetune_orange_cube_single_point_three_views/1999/train_state 
    


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_15 \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_15 --overwrite

# rm -rf ./checkpoints/pi05_lab_finetune_orange_cube_single_point_15/pi05_lab_finetune_orange_cube_single_point_15/1999/train_state 



# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_10 \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_10 --overwrite

# rm -rf ./checkpoints/pi05_lab_finetune_orange_cube_single_point_10/pi05_lab_finetune_orange_cube_single_point_10/1999/train_state 



# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_single_base_view \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_single_base_view --overwrite

# rm -rf ./checkpoints/pi05_lab_finetune_orange_cube_single_point_single_base_view/pi05_lab_finetune_orange_cube_single_point_single_base_view/1999/train_state 



# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_dual_external_views \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_dual_external_views --overwrite

# rm -rf ./checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views/pi05_lab_finetune_orange_cube_single_point_dual_external_views/1999/train_state 


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_three_views_10k_steps \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_three_views_10k_steps --overwrite

# find ./checkpoints/pi05_lab_finetune_orange_cube_single_point_three_views_10k_steps/ -type d -name "train_state" -exec rm -rf {} +


# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps --overwrite

# find ./checkpoints/pi05_lab_finetune_orange_cube_single_point_dual_external_views_15k_steps/ -type d -name "train_state" -exec rm -rf {} +

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps \
#     --exp-name=pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps --overwrite

# find ./checkpoints/pi05_lab_finetune_orange_cube_single_point_single_base_view_15k_steps/ -type d -name "train_state" -exec rm -rf {} +



# XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_aux2d_human \
#     --exp-name=pi05_aux2d_human --overwrite

# find ./checkpoints/pi05_aux2d_human/ -type d -name "train_state" -exec rm -rf {} +


XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train_eval.py pi05_aux2d_co_training \
  --val-interval 5 \
  --log-interval 5 \
  --save-interval 1000 \
  --exp-name=pi05_aux2d_co_training --overwrite

find ./checkpoints/pi05_aux2d_co_training/ -type d -name "train_state" -exec rm -rf {} +
