#!/usr/bin/env bash
set -euo pipefail

cd "${OPENPI_ROOT:-/home/t-qimhuang/code/openpi}"

RUN_NAME="${1:-base}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-20}"
SEED="${SEED:-7}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_GL

OUT_DIR="${OUT_DIR:-data/libero/5task_eval/${RUN_NAME}}"
export RUN_NAME OUT_DIR
mkdir -p "${OUT_DIR}/videos" "${OUT_DIR}/results"

run_suite() {
  local suite="$1"
  shift
  examples/libero/.venv/bin/python examples/libero/main.py \
    --args.host "${HOST}" \
    --args.port "${PORT}" \
    --args.task-suite-name "${suite}" \
    --args.task-ids "$@" \
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
    --args.seed "${SEED}" \
    --args.video-out-path "${OUT_DIR}/videos/${suite}" \
    --args.results-out-path "${OUT_DIR}/results/${suite}.json"
}

run_suite libero_goal 7 8
run_suite libero_spatial 2
run_suite libero_object 8
run_suite libero_10 2

examples/libero/.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", ""))
if not str(out_dir):
    out_dir = Path("data/libero/5task_eval") / os.environ.get("RUN_NAME", "base")
results = []
for path in sorted((out_dir / "results").glob("*.json")):
    with open(path) as file:
        results.append(json.load(file))
episodes = sum(item["total_episodes"] for item in results)
successes = sum(item["total_successes"] for item in results)
summary = {
    "total_episodes": episodes,
    "total_successes": successes,
    "total_success_rate": successes / episodes if episodes else 0.0,
    "suites": results,
}
summary_path = out_dir / "summary.json"
with open(summary_path, "w") as file:
    json.dump(summary, file, indent=2)
print(json.dumps(summary, indent=2))
PY