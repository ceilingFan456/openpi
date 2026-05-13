#!/usr/bin/env bash
# Drive all ED-Pos eval runs back-to-back. Each eval takes ~1-2 hours.
#
# 6 cells: 3 datasets × 2 step counts. Each emits a summary.json under
# data/libero/3task_eval/edpos_ds002_<dataset>_step<step>/.
#
# This script polls the local checkpoint dir once per minute; as soon as a
# (dataset, step) becomes ready (params + assets + _CHECKPOINT_METADATA), it
# evaluates it. Skips already-evaluated cells.

set -uo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/t-qimhuang/code/openpi}"
DEST_5TASK=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task
DEST_VAR=/home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task_speed_varied

cd "${OPENPI_ROOT}"

DS="${DS:-002}"
PORT="${PORT:-8013}"

declare -A CONFIG_MAP
CONFIG_MAP[original]="pi05_libero_5task"
CONFIG_MAP[speed_varied]="pi05_libero_5task_speed_varied"
CONFIG_MAP[speed_quarter]="pi05_libero_5task_speed_varied"

declare -A ROOT_MAP
ROOT_MAP[original]="${DEST_5TASK}"
ROOT_MAP[speed_varied]="${DEST_VAR}"
ROOT_MAP[speed_quarter]="${DEST_VAR}"

DATASETS="original speed_varied speed_quarter"
STEPS="2499 4999"

DATE_TAG="$(date -u +%FT%TZ)"
echo "${DATE_TAG} Starting ED-Pos ds=${DS} eval driver"

# We only allow one policy server at a time. Cells go in a fixed order.
for dataset in $DATASETS; do
  EXP_NAME="edpos_libero_3task_${dataset}_ds${DS}_0511"
  if [ "$DS" != "002" ]; then
    EXP_NAME="edpos_libero_3task_${dataset}_ds${DS}_0513"
  fi
  ROOT="${ROOT_MAP[$dataset]}"
  CONFIG="${CONFIG_MAP[$dataset]}"
  for step in $STEPS; do
    RUN_NAME="edpos_ds${DS}_${dataset}_step${step}"
    OUT_DIR="data/libero/3task_eval/${RUN_NAME}"
    SUMMARY="${OUT_DIR}/summary.json"

    if [ -f "${SUMMARY}" ]; then
      echo "$(date -u +%FT%TZ) [SKIP] ${RUN_NAME} already evaluated"
      continue
    fi

    CKPT_DIR="${ROOT}/${EXP_NAME}/${step}"

    # Wait for checkpoint to be ready (max 4h)
    READY_FOR=0
    for i in $(seq 1 240); do
      if [ -d "${CKPT_DIR}/params/d" ] && [ -d "${CKPT_DIR}/assets" ] && [ -f "${CKPT_DIR}/_CHECKPOINT_METADATA" ]; then
        READY_FOR=1
        break
      fi
      sleep 60
    done
    if [ "${READY_FOR}" != "1" ]; then
      echo "$(date -u +%FT%TZ) [TIMEOUT] ${CKPT_DIR} not ready after 4 hours, skipping"
      continue
    fi

    echo "$(date -u +%FT%TZ) [START] ${RUN_NAME}  ckpt=${CKPT_DIR}"
    bash scripts/eval_edpos_checkpoint.sh "${RUN_NAME}" "${CONFIG}" "${CKPT_DIR}" "${PORT}" \
      > "logs/eval_${RUN_NAME}.log" 2>&1 || true
    if [ -f "${SUMMARY}" ]; then
      python3 -c "
import json
with open('${SUMMARY}') as f:
    s = json.load(f)
print(f'   {s[\"total_successes\"]}/{s[\"total_episodes\"]} = {s[\"total_success_rate\"]*100:.1f}%')
"
    else
      echo "$(date -u +%FT%TZ) [FAIL] no summary.json produced for ${RUN_NAME}"
      tail -20 "logs/eval_${RUN_NAME}.log" 2>/dev/null | sed 's/^/   /'
    fi
  done
done
echo "$(date -u +%FT%TZ) === ALL EVALS DONE ==="
