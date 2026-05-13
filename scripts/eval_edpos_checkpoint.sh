#!/usr/bin/env bash
# Evaluate one ED-Pos pi05 checkpoint on the LIBERO 3-task benchmark (60 trials).
#
# Usage:
#   scripts/eval_edpos_checkpoint.sh \
#     <run_name> <config_name> <checkpoint_dir> [port]
#
# Examples (run from /home/t-qimhuang/code/openpi):
#   scripts/eval_edpos_checkpoint.sh \
#     edpos_ds002_original_step2499 pi05_libero_5task \
#     /home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task/edpos_libero_3task_original_ds002_0511/2499
#
#   scripts/eval_edpos_checkpoint.sh \
#     edpos_ds002_speed_quarter_step4999 pi05_libero_5task_speed_varied \
#     /home/t-qimhuang/data/openpi_checkpoints/pi05_libero_5task_speed_varied/edpos_libero_3task_speed_quarter_ds002_0511/4999
#
# Output:
#   data/libero/3task_eval/<run_name>/{summary.json, results/, videos/, server.log, eval.log}
set -euo pipefail

RUN_NAME="${1:?run name required}"
CONFIG_NAME="${2:?config name required (pi05_libero_5task or pi05_libero_5task_speed_varied)}"
CHECKPOINT_DIR="${3:?checkpoint dir required}"
PORT="${4:-8012}"

OPENPI_ROOT="${OPENPI_ROOT:-/home/t-qimhuang/code/openpi}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}"
SEED="${SEED:-7}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${OPENPI_ROOT}"

OUT_DIR="data/libero/3task_eval/${RUN_NAME}"
mkdir -p "${OUT_DIR}"
SERVER_LOG="${OUT_DIR}/server.log"
EVAL_LOG="${OUT_DIR}/eval.log"

if [ ! -d "${CHECKPOINT_DIR}/params" ]; then
  echo "ERROR: ${CHECKPOINT_DIR}/params does not exist" >&2
  exit 1
fi

# Kill any orphan process holding our target port.
PORT_PIDS=$(lsof -t -i ":${PORT}" 2>/dev/null || true)
if [ -n "${PORT_PIDS}" ]; then
  echo "$(date -u +%FT%TZ) Killing orphan process(es) on port ${PORT}: ${PORT_PIDS}"
  for p in ${PORT_PIDS}; do kill -KILL "$p" 2>/dev/null || true; done
  sleep 5
fi

echo "$(date -u +%FT%TZ) Starting policy server (config=${CONFIG_NAME}, port=${PORT}, ckpt=${CHECKPOINT_DIR})"
# Keep JAX memory footprint small to coexist with other GPU users on shared box.
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.20}" \
  .venv/bin/python scripts/serve_policy.py \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${CHECKPOINT_DIR}" \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "Server PID: ${SERVER_PID}"

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) Stopping server PID ${SERVER_PID}"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Wait for server to bind to its port (up to 10 minutes for cold-start JAX).
for i in $(seq 1 120); do
  if (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
    echo "$(date -u +%FT%TZ) Server is ready on port ${PORT} (attempt ${i})"
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server died before becoming ready" >&2
    tail -20 "${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 5
done

if ! (echo > "/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
  echo "ERROR: server failed to start within 10 minutes" >&2
  tail -50 "${SERVER_LOG}" >&2
  exit 1
fi

echo "$(date -u +%FT%TZ) Running 3-task benchmark (${NUM_TRIALS_PER_TASK} trials/task, seed=${SEED})"
PORT="${PORT}" HOST="127.0.0.1" \
  NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
  SEED="${SEED}" \
  OUT_DIR="${OUT_DIR}" \
  MUJOCO_GL="${MUJOCO_GL}" \
  examples/libero/run_3task_benchmark.sh "${RUN_NAME}" \
  > "${EVAL_LOG}" 2>&1

echo "$(date -u +%FT%TZ) Eval complete; summary:"
python3 -c "
import json, sys
with open('${OUT_DIR}/summary.json') as f:
    summary = json.load(f)
print(f'Total: {summary[\"total_successes\"]}/{summary[\"total_episodes\"]} = {summary[\"total_success_rate\"]*100:.1f}%')
for suite in summary['suites']:
    for task in suite['per_task']:
        print(f'  {suite[\"task_suite_name\"]}/task{task[\"task_id\"]:02d}: {task[\"successes\"]}/{task[\"episodes\"]} = {task[\"success_rate\"]*100:.1f}% ({task[\"task_description\"]})')
"
