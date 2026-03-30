#!/bin/bash
# ============================================================
#  Quick reference: submitting openpi training jobs on AMLT
# ============================================================

# --- Prerequisites ---
# 1. Activate the amlt conda env
conda activate amlt

# 2. cd into submit_jobs/
cd ~/code/openpi/submit_jobs

# --- Submit a training job ---
amlt run -y -d "<description>" train_job.yaml

# --- What to change in train_job.yaml each time ---
# 1. Training command (last line under `command:`):
#      - Config name:   pick_and_place_new_132_three_views_30k_steps
#      - Exp name:      --exp-name=<your_experiment_name>
#      - Batch size:    --batch-size=96  (must be divisible by num GPUs)
#      - Overwrite:     --overwrite  OR  --resume
# 2. SKU (GPU count):
#      141G1-H200           = 1x H200
#      141G8-H200-NvLink    = 8x H200 (current)
# 3. max_run_duration_seconds: 86400 = 24h

# --- Example: full training command ---
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
#   pick_and_place_new_132_three_views_30k_steps \
#   --exp-name=pick_and_place_new_132_three_views_30k_steps \
#   --checkpoint-base-dir=/mnt/default_storage/qiming/openpi/checkpoints \
#   --batch-size=96 \
#   --overwrite

# --- Checkpoints saved to ---
# /mnt/default_storage/qiming/openpi/checkpoints/<config_name>/<exp_name>/
# (shared storage, persists after job ends)

# --- Monitor jobs ---
amlt status <experiment_name>                     # check status
amlt logs <experiment_name> :<job_name>           # stream logs
amlt ssh <experiment_name> :<job_name>            # SSH into running job
amlt cancel <experiment_name>                     # cancel a job

# --- Debug shell (interactive) ---
amlt debug -y job.yaml                            # launches interactive node
# then SSH in:  amlt ssh debug-- :shell

# --- Batch size rule ---
# batch_size must be divisible by number of GPUs
# 1 GPU:  --batch-size=12
# 8 GPUs: --batch-size=96  (12 per GPU)
