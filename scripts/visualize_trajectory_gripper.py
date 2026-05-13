"""
Generate trajectory + gripper plots for rollout episodes.

Creates a 2-panel figure per episode:
  Left: 3D EE trajectory with action chunk boundaries
  Right: Gripper action (dim 6) vs time, with chunk boundaries

Usage:
    python scripts/visualize_trajectory_gripper.py \
        --eval-dir data/libero/3task_eval/e7_ah40_step4999 \
        --task-filter task08
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trajectory_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def extract_executed_gripper(traj_data):
    """Extract the gripper action that was actually executed at each timestep."""
    boundaries = traj_data["chunk_boundaries"]
    chunks = traj_data["chunk_full_actions"]
    replan = traj_data["replan_steps"]
    n_steps = traj_data["num_steps"]

    gripper = np.zeros(n_steps)
    for i, (start, chunk) in enumerate(zip(boundaries, chunks)):
        chunk = np.array(chunk)  # (action_horizon, 7)
        for j in range(replan):
            t = start + j
            if t < n_steps and j < len(chunk):
                gripper[t] = chunk[j, 6]  # dim 6 = gripper
    return gripper


def extract_full_chunk_gripper(traj_data):
    """Extract the full predicted gripper trajectory for each chunk."""
    boundaries = traj_data["chunk_boundaries"]
    chunks = traj_data["chunk_full_actions"]
    n_steps = traj_data["num_steps"]

    result = []
    for start, chunk in zip(boundaries, chunks):
        chunk = np.array(chunk)
        times = np.arange(start, min(start + len(chunk), n_steps))
        values = chunk[:len(times), 6]
        result.append((times, values))
    return result


def plot_episode(traj_data, output_path, show_all_chunks=True):
    """Plot trajectory + gripper for one episode."""
    ee = np.array(traj_data["ee_positions"])
    boundaries = traj_data["chunk_boundaries"]
    replan = traj_data["replan_steps"]
    horizon = traj_data["action_horizon"]
    n_steps = traj_data["num_steps"]
    success = traj_data["success"]
    task_desc = traj_data["task_description"]
    ep_idx = traj_data["episode_idx"]

    gripper_executed = extract_executed_gripper(traj_data)
    chunk_grippers = extract_full_chunk_gripper(traj_data)

    fig = plt.figure(figsize=(16, 6))

    # --- Left: 3D trajectory ---
    ax1 = fig.add_subplot(121, projection="3d")

    # Full trajectory
    ax1.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="gray", linewidth=1.0, alpha=0.4, label="Full path")

    # Color by time
    n = len(ee)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax1.plot(ee[i:i+2, 0], ee[i:i+2, 1], ee[i:i+2, 2],
                 color=colors[i], linewidth=2.0)

    # Chunk boundaries
    for i, start in enumerate(boundaries):
        if start < n:
            ax1.scatter([ee[start, 0]], [ee[start, 1]], [ee[start, 2]],
                        color="#2171b5", s=15, marker="D", zorder=8, depthshade=False, alpha=0.5)

    # Start / end markers
    ax1.scatter(*ee[0], color="green", s=100, marker="o", zorder=10, depthshade=False, label="Start")
    ax1.scatter(*ee[-1], color="red", s=100, marker="*", zorder=10, depthshade=False, label="End")

    ax1.set_xlabel("X", fontsize=9)
    ax1.set_ylabel("Y", fontsize=9)
    ax1.set_zlabel("Z", fontsize=9)
    ax1.tick_params(labelsize=7)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("EE Trajectory (color = time)", fontsize=10)

    # --- Right: Gripper action vs time ---
    ax2 = fig.add_subplot(122)

    # Show full predicted chunks as faint lines
    if show_all_chunks:
        for times, values in chunk_grippers:
            ax2.plot(times, values, color="#e6550d", linewidth=0.5, alpha=0.2)

    # Executed gripper
    timesteps = np.arange(n_steps)
    ax2.plot(timesteps, gripper_executed, color="#2171b5", linewidth=1.5, label="Executed gripper", zorder=5)

    # Chunk boundaries as vertical lines
    for start in boundaries:
        ax2.axvline(x=start, color="gray", linewidth=0.5, alpha=0.3, linestyle="--")

    ax2.set_xlabel("Timestep", fontsize=10)
    ax2.set_ylabel("Gripper Action (dim 6)", fontsize=10)
    ax2.set_title("Gripper Action vs Time", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, n_steps)
    ax2.grid(True, alpha=0.2)

    result_str = "SUCCESS" if success else "FAILURE"
    result_color = "green" if success else "red"
    fig.suptitle(
        f"Episode {ep_idx}: {task_desc} — [{result_str}]  (ah={horizon}, replan={replan})",
        fontsize=11, fontweight="bold", color=result_color,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory + gripper from eval rollouts")
    parser.add_argument("--eval-dir", type=str, required=True, help="Eval directory with videos/")
    parser.add_argument("--task-filter", type=str, default=None, help="Filter JSON files by name substring")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: eval-dir/trajectory_plots)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Max episodes to plot")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir) if args.output_dir else eval_dir / "trajectory_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all rollout JSONs
    json_files = sorted(eval_dir.rglob("*.json"))
    json_files = [f for f in json_files if f.name.startswith("rollout_")]

    if args.task_filter:
        json_files = [f for f in json_files if args.task_filter in f.name]

    if args.max_episodes:
        json_files = json_files[:args.max_episodes]

    print(f"Found {len(json_files)} rollout JSONs")

    for json_path in json_files:
        traj_data = load_trajectory_data(json_path)
        stem = json_path.stem
        out_path = output_dir / f"{stem}.png"
        plot_episode(traj_data, out_path)
        result = "OK" if traj_data["success"] else "FAIL"
        print(f"  [{result}] {out_path.name}")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
