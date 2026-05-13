#!/usr/bin/env bash
set -euo pipefail

cd "${OPENPI_ROOT:-/home/t-qimhuang/code/openpi}"

TARGET="${1:-all}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-/home/t-qimhuang/data/openpi_checkpoints}"
PYTHON="${PYTHON:-.venv/bin/python}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

run_train() {
  local config_name="$1"
  local exp_name="$2"
  local dataset_root="$3"

  echo "=== Training ${config_name} (${exp_name}) ==="
  LIBERO_DATASET_ROOT="${dataset_root}" "${PYTHON}" scripts/train.py "${config_name}" \
    --exp-name="${exp_name}" \
    --checkpoint-base-dir="${CHECKPOINT_BASE_DIR}" \
    --batch-size="${BATCH_SIZE}" \
    --num-train-steps="${NUM_TRAIN_STEPS}" \
    --save-interval="${SAVE_INTERVAL}" \
    --log-interval="${LOG_INTERVAL}" \
    --overwrite
}

case "${TARGET}" in
  e1|original)
    run_train "pi05_libero_5task" "e1_libero_5task_original" "/home/t-qimhuang/data/libero_5task"
    ;;
  e2|speed|speed-varied)
    run_train "pi05_libero_5task_speed_varied" "e2_libero_5task_speed_varied" "/home/t-qimhuang/data/libero_5task_speed_varied/mixed"
    ;;
  all)
    run_train "pi05_libero_5task" "e1_libero_5task_original" "/home/t-qimhuang/data/libero_5task"
    run_train "pi05_libero_5task_speed_varied" "e2_libero_5task_speed_varied" "/home/t-qimhuang/data/libero_5task_speed_varied/mixed"
    ;;
  *)
    echo "Usage: $0 [e1|e2|all]" >&2
    exit 2
    ;;
esac
