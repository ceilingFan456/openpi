#!/usr/bin/env bash
# Sync ED-Pos ds=0.002 checkpoints (params + assets + metadata only, NO train_state).
# Skips any (dataset, step) that already has params + assets + _CHECKPOINT_METADATA locally.

set -uo pipefail

SRC_BASE="https://singaporeteamstorage.blob.core.windows.net/shared/qiming/openpi/checkpoints"
DEST_5TASK=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task
DEST_VAR=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task_speed_varied

export AZCOPY_AUTO_LOGIN_TYPE=AZCLI

declare -A CONFIG_DIR
CONFIG_DIR[original]="${DEST_5TASK}"
CONFIG_DIR[speed_varied]="${DEST_VAR}"
CONFIG_DIR[speed_quarter]="${DEST_VAR}"

declare -A SUITE_DIR
SUITE_DIR[original]="pi05_libero_5task"
SUITE_DIR[speed_varied]="pi05_libero_5task_speed_varied"
SUITE_DIR[speed_quarter]="pi05_libero_5task_speed_varied"

DS="${DS:-002}"
DATASETS="${DATASETS:-original speed_varied speed_quarter}"
STEPS="${STEPS:-2499 4999}"

for dataset in $DATASETS; do
  EXP_NAME="edpos_libero_3task_${dataset}_ds${DS}_0511"
  if [ "$DS" != "002" ]; then
    EXP_NAME="edpos_libero_3task_${dataset}_ds${DS}_0513"
  fi
  DEST="${CONFIG_DIR[$dataset]}/${EXP_NAME}"
  SUITE="${SUITE_DIR[$dataset]}"
  for step in $STEPS; do
    STEP_DIR="${DEST}/${step}"
    if [ -d "${STEP_DIR}/params/d" ] && [ -d "${STEP_DIR}/assets" ] && [ -f "${STEP_DIR}/_CHECKPOINT_METADATA" ]; then
      echo "$(date -u +%FT%TZ) [SKIP] ${dataset} step ${step} already complete (size: $(du -sh ${STEP_DIR} | cut -f1))"
      continue
    fi
    mkdir -p "${STEP_DIR}"
    SRC="${SRC_BASE}/${SUITE}/${EXP_NAME}/${step}"
    echo "$(date -u +%FT%TZ) Syncing ${dataset} step ${step} (params)..."
    azcopy copy "${SRC}/params" "${STEP_DIR}/" --recursive=true 2>&1 | tail -3
    echo "$(date -u +%FT%TZ) Syncing ${dataset} step ${step} (assets)..."
    azcopy copy "${SRC}/assets" "${STEP_DIR}/" --recursive=true 2>&1 | tail -3
    echo "$(date -u +%FT%TZ) Syncing ${dataset} step ${step} (metadata)..."
    azcopy copy "${SRC}/_CHECKPOINT_METADATA" "${STEP_DIR}/" 2>&1 | tail -3
    echo "$(date -u +%FT%TZ) Done ${dataset} step ${step} (total: $(du -sh ${STEP_DIR} | cut -f1))"
  done
done
echo "$(date -u +%FT%TZ) === ALL DONE ==="
