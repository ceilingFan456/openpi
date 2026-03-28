"""
Visualize state redundancy test results.

Generates plots showing:
  1. EE position differences over the action horizon for each null-space config
  2. EE rotation differences over the action horizon
  3. Summary bar chart: joint difference vs EE trajectory divergence
  4. 3D EE trajectory plot

Usage:
    python testing_state_redundancy/visualize_results.py
    python testing_state_redundancy/visualize_results.py --results-path testing_state_redundancy/results/state_redundancy_results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_results(results_path):
    with open(results_path) as f:
        return json.load(f)


def plot_results(data, output_dir):
    results = data["results"]
    n_configs = len(results)
    dt = data.get("dt", 0.05)

    if n_configs == 0:
        print("No results to plot.")
        return

    # Color map
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_configs))

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # --- Plot 1: EE position diff over horizon ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(results):
        steps = np.arange(len(r["ee_comparison"]["pos_diff_per_step"]))
        pos_diff_mm = np.array(r["ee_comparison"]["pos_diff_per_step"]) * 1000
        label = f"Config {i+1} (Δq={r['joint_diff_rad']:.3f} rad)"
        ax1.plot(steps, pos_diff_mm, color=colors[i], linewidth=2, label=label)
    ax1.set_xlabel("Action Step")
    ax1.set_ylabel("EE Position Difference (mm)")
    ax1.set_title("EE Position Divergence Over Action Horizon")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: EE rotation diff over horizon ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(results):
        steps = np.arange(len(r["ee_comparison"]["rot_diff_per_step_deg"]))
        rot_diff = r["ee_comparison"]["rot_diff_per_step_deg"]
        label = f"Config {i+1} (Δq={r['joint_diff_rad']:.3f} rad)"
        ax2.plot(steps, rot_diff, color=colors[i], linewidth=2, label=label)
    ax2.set_xlabel("Action Step")
    ax2.set_ylabel("EE Rotation Difference (degrees)")
    ax2.set_title("EE Rotation Divergence Over Action Horizon")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Joint diff vs final EE diff ---
    ax3 = fig.add_subplot(gs[1, 0])
    joint_diffs = [r["joint_diff_rad"] for r in results]
    pos_diffs_final = [r["ee_comparison"]["pos_diff_final"] * 1000 for r in results]
    rot_diffs_final = [r["ee_comparison"]["rot_diff_final_deg"] for r in results]
    action_diffs = [r["action_chunk_diff_l2"] for r in results]

    ax3_twin = ax3.twinx()
    bars1 = ax3.bar(np.arange(n_configs) - 0.15, pos_diffs_final, 0.3,
                     color='steelblue', alpha=0.8, label='EE pos diff (mm)')
    bars2 = ax3_twin.bar(np.arange(n_configs) + 0.15, rot_diffs_final, 0.3,
                          color='coral', alpha=0.8, label='EE rot diff (deg)')

    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("Final Position Diff (mm)", color='steelblue')
    ax3_twin.set_ylabel("Final Rotation Diff (deg)", color='coral')
    ax3.set_title("Final EE Difference vs Configuration")
    ax3.set_xticks(np.arange(n_configs))
    ax3.set_xticklabels([f"Δq={d:.2f}" for d in joint_diffs], fontsize=8)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Joint diff vs action chunk diff (scatter) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(joint_diffs, action_diffs, c=colors[:n_configs], s=100, edgecolors='black', zorder=3)
    for i, (jd, ad) in enumerate(zip(joint_diffs, action_diffs)):
        ax4.annotate(f"Config {i+1}", (jd, ad), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    ax4.set_xlabel("Joint Configuration Difference (rad)")
    ax4.set_ylabel("Action Chunk Difference (L2 norm)")
    ax4.set_title("Joint Space Diff vs Action Prediction Diff")
    ax4.grid(True, alpha=0.3)

    # Add correlation if enough points
    if n_configs >= 3:
        corr = np.corrcoef(joint_diffs, action_diffs)[0, 1]
        ax4.text(0.05, 0.95, f"Correlation: {corr:.3f}",
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Title
    config_name = data.get("config_name", "unknown")
    fig.suptitle(f"State Redundancy Analysis: {config_name}", fontsize=14, fontweight='bold')

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "state_redundancy_analysis.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize state redundancy test results")
    parser.add_argument("--results-path", type=str,
                        default=str(PROJECT_ROOT / "testing_state_redundancy/results/state_redundancy_results.json"),
                        help="Path to JSON results file")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "testing_state_redundancy/results"),
                        help="Output directory for plots")

    args = parser.parse_args()

    data = load_results(args.results_path)
    print(f"Loaded results: {len(data['results'])} configurations")
    plot_results(data, args.output_dir)


if __name__ == "__main__":
    main()
