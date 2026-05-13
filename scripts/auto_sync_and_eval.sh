#!/usr/bin/env bash
# Watch AMLT for ds=001 and ds=004 to complete, then sync + eval.
#
# Polls amlt status every 5 minutes. Once all 3 jobs of an experiment are PASS,
# launches the sync script for that DS, then the eval driver. Each call is
# idempotent so re-running is safe.
set -uo pipefail
cd /home/t-qimhuang/code/openpi

LOG=logs/auto_sync_and_eval.log
mkdir -p logs
exec > >(tee -a "$LOG") 2>&1

ts() { date -u +%FT%TZ; }

is_done() {
  # Returns 0 (true) if all jobs of the given experiment are PASS or COMPLETED
  local exp=$1
  AMLT_JSON_TABLES=1 amlt status "$exp" 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
rows = data[0]['rows']
n_done = sum(1 for r in rows if r['status'].lower() in ('completed', 'pass', 'passed'))
n_failed = sum(1 for r in rows if r['status'].lower() in ('failed', 'fail', 'cancelled'))
n_total = len(rows)
sys.exit(0 if n_done == n_total else (2 if n_failed > 0 else 1))
" 2>/dev/null
}

wait_for_amlt() {
  local exp=$1
  echo "$(ts) Waiting for $exp to complete..."
  while true; do
    if is_done "$exp"; then
      echo "$(ts) $exp complete!"
      return 0
    fi
    rc=$?
    if [ "$rc" = "2" ]; then
      echo "$(ts) WARN: some jobs in $exp failed. Will still try to sync what's available."
      sleep 60
      return 1
    fi
    # Print short status every loop so the log shows progress
    AMLT_JSON_TABLES=1 amlt status "$exp" 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
rows = data[0]['rows']
parts = []
for r in rows:
    parts.append(f\"{r['job_name'].lstrip(':')}={r['status']}({r['duration']})\")
print('  ' + ', '.join(parts))
" 2>/dev/null || echo "  (status parse failed)"
    sleep 300  # 5 minutes
  done
}

handle() {
  local ds=$1
  local exp="edpos-3task-ds${ds}-0513"
  wait_for_amlt "$exp" || true

  echo "$(ts) Starting sync for ds=${ds}..."
  DS=$ds bash scripts/sync_edpos_checkpoints.sh 2>&1 | tail -50
  # Repair pass: any missing assets/metadata fixes
  echo "$(ts) Repair pass for ds=${ds}..."
  bash scripts/repair_missing_pieces.sh "$ds" 2>&1 | tail -20

  echo "$(ts) Starting eval driver for ds=${ds}..."
  DS=$ds DATASETS="original speed_varied speed_quarter" \
    bash scripts/drive_edpos_evals.sh 2>&1 | tail -200
  echo "$(ts) ds=${ds} done."
}

# Process both in series so they don't fight for the GPU
handle 001
handle 004

echo "$(ts) === AUTO-SYNC-AND-EVAL ALL DONE ==="
