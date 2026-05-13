#!/usr/bin/env bash
# Repair missing assets / _CHECKPOINT_METADATA / params for ED-Pos checkpoints.
# Usage: bash scripts/repair_missing_pieces.sh <ds>   e.g. 002, 001, 004
set -uo pipefail

DS=${1:?ds required}
SUFFIX="0511"
[ "$DS" != "002" ] && SUFFIX="0513"

SRC_BASE="https://singaporeteamstorage.blob.core.windows.net/shared/qiming/openpi/checkpoints"
DEST_5TASK=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task
DEST_VAR=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task_speed_varied

export AZCOPY_AUTO_LOGIN_TYPE=AZCLI

declare -A SUITE_DIR
SUITE_DIR[original]="pi05_libero_5task"
SUITE_DIR[speed_varied]="pi05_libero_5task_speed_varied"
SUITE_DIR[speed_quarter]="pi05_libero_5task_speed_varied"
declare -A DEST_FOR
DEST_FOR[original]="${DEST_5TASK}"
DEST_FOR[speed_varied]="${DEST_VAR}"
DEST_FOR[speed_quarter]="${DEST_VAR}"

fetch() {
  local src=$1
  local dst=$2
  local label=$3
  for i in 1 2 3 4 5; do
    if azcopy copy "$src" "$dst" --recursive=true 2>&1 | tail -2 | grep -q "Completed"; then
      echo "  [$label] OK on try $i"
      return 0
    fi
    echo "  [$label] retry $i ..."
    sleep $((i * 5))
  done
  echo "  [$label] FAILED after 5 retries"
  return 1
}

for dataset in original speed_varied speed_quarter; do
  EXP="edpos_libero_3task_${dataset}_ds${DS}_${SUFFIX}"
  SUITE="${SUITE_DIR[$dataset]}"
  DEST_BASE="${DEST_FOR[$dataset]}/${EXP}"
  for step in 2499 4999; do
    STEP_DIR="${DEST_BASE}/${step}"
    [ ! -d "${STEP_DIR}" ] && continue  # base dir not even created — sync wasn't run
    if [ ! -d "${STEP_DIR}/assets" ]; then
      fetch "${SRC_BASE}/${SUITE}/${EXP}/${step}/assets" "${STEP_DIR}/" "${dataset}/${step}/assets"
    fi
    if [ ! -f "${STEP_DIR}/_CHECKPOINT_METADATA" ]; then
      fetch "${SRC_BASE}/${SUITE}/${EXP}/${step}/_CHECKPOINT_METADATA" "${STEP_DIR}/" "${dataset}/${step}/metadata"
    fi
    if [ ! -d "${STEP_DIR}/params/d" ]; then
      fetch "${SRC_BASE}/${SUITE}/${EXP}/${step}/params" "${STEP_DIR}/" "${dataset}/${step}/params"
    fi
  done
done
