#!/usr/bin/env python3
"""Monitor LIBERO 5-task training jobs and run local benchmarks when ready.

Example:
    python3 scripts/monitor_libero_5task_jobs.py --poll-seconds 600 --num-trials-per-task 20

Outputs:
    data/libero/5task_eval/finetuned_monitor/monitor_state.json
    data/libero/5task_eval/finetuned_monitor/report.md
    data/libero/5task_eval/{e1_original_2gpu_wandb,e2_speed_varied_2gpu_wandb}/summary.json
    data/libero/5task_eval/{run_name}/visual_samples/{success,failure}/*.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any


OPENPI_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "speed-var-5task-2gpu-0428b"
CHECKPOINT_BASE_DIR = Path("/mnt/default_storage/qiming/openpi/checkpoints")
OUTPUT_ROOT = OPENPI_ROOT / "data/libero/5task_eval"
MONITOR_DIR = OUTPUT_ROOT / "finetuned_monitor"
SUCCESS_STATUSES = {"completed", "succeeded"}
FAILURE_STATUSES = {"failed", "canceled", "cancelled", "killed", "lost", "stopped"}


@dataclasses.dataclass(frozen=True)
class JobSpec:
    job_name: str
    config_name: str
    exp_name: str
    run_name: str
    port: int
    final_step: int = 9999

    @property
    def checkpoint_root(self) -> Path:
        return CHECKPOINT_BASE_DIR / self.config_name / self.exp_name


JOBS = [
    JobSpec(
        job_name=":e1-libero-5task-original",
        config_name="pi05_libero_5task",
        exp_name="e1_libero_5task_original_2gpu_wandb",
        run_name="e1_original_2gpu_wandb",
        port=8011,
    ),
    JobSpec(
        job_name=":e2-libero-5task-speed-varied",
        config_name="pi05_libero_5task_speed_varied",
        exp_name="e2_libero_5task_speed_varied_2gpu_wandb",
        run_name="e2_speed_varied_2gpu_wandb",
        port=8012,
    ),
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if stdout_path:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with stdout_path.open("a", encoding="utf-8") as stdout_file:
            stdout_file.write(f"\n\n===== {utc_now()} command: {' '.join(command)} =====\n")
            stdout_file.flush()
            result = subprocess.run(
                command,
                cwd=OPENPI_ROOT,
                env=merged_env,
                text=True,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
    else:
        result = subprocess.run(
            command,
            cwd=OPENPI_ROOT,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )

    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result


def parse_amlt_json_tables(stdout: str) -> list[dict[str, Any]]:
    json_start = stdout.find("[")
    if json_start < 0:
        raise ValueError(f"Could not find JSON table in amlt output:\n{stdout}")
    return json.loads(stdout[json_start:])


def get_job_statuses() -> dict[str, dict[str, Any]]:
    result = run_command(["amlt", "status", EXPERIMENT_NAME], env={"AMLT_JSON_TABLES": "1"})
    tables = parse_amlt_json_tables(result.stdout)
    rows = tables[0].get("rows", []) if tables else []
    return {row["job_name"]: row for row in rows}


def find_latest_checkpoint(spec: JobSpec) -> tuple[int, Path] | None:
    if not spec.checkpoint_root.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for path in spec.checkpoint_root.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        if (path / "params").exists() and (path / "assets").exists():
            candidates.append((int(path.name), path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])


def write_state(state: dict[str, Any]) -> None:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    with (MONITOR_DIR / "monitor_state.json").open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


def wait_for_tcp_port(port: int, server_process: subprocess.Popen[str], timeout_seconds: int = 1800) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server_process.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {server_process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(5)
    raise TimeoutError(f"Policy server on port {port} did not become ready within {timeout_seconds}s")


def start_policy_server(spec: JobSpec, checkpoint_dir: Path, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    command = [
        ".venv/bin/python",
        "scripts/serve_policy.py",
        "--port",
        str(spec.port),
        "policy:checkpoint",
        "--policy.config",
        spec.config_name,
        "--policy.dir",
        str(checkpoint_dir),
    ]
    log_file.write(f"\n\n===== {utc_now()} command: {' '.join(command)} =====\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=OPENPI_ROOT,
        env={**os.environ, "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85"},
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    wait_for_tcp_port(spec.port, process)
    return process


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def run_benchmark(spec: JobSpec, checkpoint_dir: Path, num_trials_per_task: int) -> dict[str, Any]:
    run_dir = OUTPUT_ROOT / spec.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    log_dir = MONITOR_DIR / "logs"
    server_log = log_dir / f"{spec.run_name}_server.log"
    eval_log = log_dir / f"{spec.run_name}_eval.log"
    server_process = start_policy_server(spec, checkpoint_dir, server_log)
    try:
        run_command(
            ["examples/libero/run_5task_benchmark.sh", spec.run_name],
            env={
                "PORT": str(spec.port),
                "HOST": "0.0.0.0",
                "NUM_TRIALS_PER_TASK": str(num_trials_per_task),
                "OUT_DIR": str(run_dir.relative_to(OPENPI_ROOT)),
                "MUJOCO_GL": "egl",
            },
            stdout_path=eval_log,
        )
    finally:
        stop_process(server_process)

    summary_path = run_dir / "summary.json"
    with summary_path.open(encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    summary["checkpoint_dir"] = str(checkpoint_dir)
    summary["run_dir"] = str(run_dir)
    summary["visual_samples"] = sample_visualizations(run_dir)
    return summary


def sample_visualizations(run_dir: Path, max_per_outcome: int = 3) -> dict[str, list[str]]:
    samples_root = run_dir / "visual_samples"
    if samples_root.exists():
        shutil.rmtree(samples_root)
    samples: dict[str, list[str]] = {"success": [], "failure": []}
    for outcome in samples:
        destination = samples_root / outcome
        destination.mkdir(parents=True, exist_ok=True)
        videos = sorted((run_dir / "videos").glob(f"**/*_{outcome}.mp4"))[:max_per_outcome]
        for video in videos:
            copied_path = destination / video.name
            shutil.copy2(video, copied_path)
            samples[outcome].append(str(copied_path))
    with (samples_root / "manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(samples, manifest_file, indent=2)
    return samples


def write_report(results: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# LIBERO 5-task finetuned benchmark report",
        "",
        f"Generated: {utc_now()}",
        f"AMLT experiment: `{EXPERIMENT_NAME}`",
        "",
        "| Run | Episodes | Successes | Success rate | Checkpoint |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for run_name, summary in results.items():
        lines.append(
            f"| {run_name} | {summary['total_episodes']} | {summary['total_successes']} | "
            f"{summary['total_success_rate']:.4f} | `{summary['checkpoint_dir']}` |"
        )
    lines.extend(["", "## Per-task results", ""])
    for run_name, summary in results.items():
        lines.extend([
            f"### {run_name}",
            "",
            "| Suite | Task ID | Task | Episodes | Successes | Rate |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ])
        for suite in summary["suites"]:
            for task in suite["per_task"]:
                task_description = task["task_description"].replace("|", " ")
                lines.append(
                    f"| {task['task_suite_name']} | {task['task_id']} | {task_description} | "
                    f"{task['episodes']} | {task['successes']} | {task['success_rate']:.4f} |"
                )
        lines.extend(["", "Visual samples:"])
        for outcome, paths in summary["visual_samples"].items():
            if paths:
                joined_paths = ", ".join(f"`{path}`" for path in paths)
                lines.append(f"- {outcome}: {joined_paths}")
            else:
                lines.append(f"- {outcome}: none sampled")
        lines.append("")

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    (MONITOR_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    with (MONITOR_DIR / "results.json").open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor AMLT training jobs and run local LIBERO 5-task benchmarks.")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--num-trials-per-task", type=int, default=20)
    args = parser.parse_args()

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{utc_now()}] Monitoring {EXPERIMENT_NAME}", flush=True)
    print(f"[{utc_now()}] State: {MONITOR_DIR / 'monitor_state.json'}", flush=True)
    print(f"[{utc_now()}] Report: {MONITOR_DIR / 'report.md'}", flush=True)

    completed_jobs: dict[str, Path] = {}
    while len(completed_jobs) < len(JOBS):
        statuses = get_job_statuses()
        state: dict[str, Any] = {"updated_at": utc_now(), "experiment": EXPERIMENT_NAME, "jobs": {}}
        for spec in JOBS:
            row = statuses.get(spec.job_name, {})
            status = str(row.get("status", "unknown")).lower()
            latest_checkpoint = find_latest_checkpoint(spec)
            state["jobs"][spec.job_name] = {
                "status": status,
                "wandb_url": row.get("wandb_url", ""),
                "checkpoint_root": str(spec.checkpoint_root),
                "latest_checkpoint": str(latest_checkpoint[1]) if latest_checkpoint else "",
                "latest_step": latest_checkpoint[0] if latest_checkpoint else None,
            }

            if status in FAILURE_STATUSES:
                write_state(state)
                raise RuntimeError(f"{spec.job_name} ended with status {status}")
            if status in SUCCESS_STATUSES and latest_checkpoint and latest_checkpoint[0] >= spec.final_step:
                completed_jobs[spec.job_name] = latest_checkpoint[1]

        write_state(state)
        if len(completed_jobs) < len(JOBS):
            printable = {name: data["status"] for name, data in state["jobs"].items()}
            print(f"[{utc_now()}] Waiting for jobs/checkpoints: {printable}", flush=True)
            time.sleep(args.poll_seconds)

    results: dict[str, dict[str, Any]] = {}
    for spec in JOBS:
        checkpoint_dir = completed_jobs[spec.job_name]
        print(f"[{utc_now()}] Running benchmark for {spec.run_name}: {checkpoint_dir}", flush=True)
        results[spec.run_name] = run_benchmark(spec, checkpoint_dir, args.num_trials_per_task)
        write_report(results)

    write_report(results)
    print(f"[{utc_now()}] Done. Report: {MONITOR_DIR / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())