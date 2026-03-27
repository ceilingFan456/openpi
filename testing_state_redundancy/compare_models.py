"""
Compare state redundancy results between base pi05_droid and finetuned model.

Usage:
    python testing_state_redundancy/compare_models.py
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "testing_state_redundancy" / "results"


def load(name):
    with open(RESULTS_DIR / name) as f:
        return json.load(f)


def main():
    base = load("state_redundancy_results_pi05_droid_base.json")
    fine = load("state_redundancy_results_finetuned.json")

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Shared x: joint diffs (same configs for both)
    joint_diffs = [r["joint_diff_rad"] for r in base["results"]]

    # --- 1. Action chunk L2 diff comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    base_action = [r["action_chunk_diff_l2"] for r in base["results"]]
    fine_action = [r["action_chunk_diff_l2"] for r in fine["results"]]
    x = np.arange(len(joint_diffs))
    w = 0.35
    ax1.bar(x - w/2, base_action, w, color='#e74c3c', alpha=0.8, label='Base pi05_droid')
    ax1.bar(x + w/2, fine_action, w, color='#3498db', alpha=0.8, label='Finetuned')
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Action Chunk L2 Diff")
    ax1.set_title("Action Prediction Difference")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Δq={d:.2f}" for d in joint_diffs], fontsize=7, rotation=45)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- 2. Final EE position diff comparison ---
    ax2 = fig.add_subplot(gs[0, 1])
    base_pos = [r["ee_comparison"]["pos_diff_final"] * 1000 for r in base["results"]]
    fine_pos = [r["ee_comparison"]["pos_diff_final"] * 1000 for r in fine["results"]]
    ax2.bar(x - w/2, base_pos, w, color='#e74c3c', alpha=0.8, label='Base pi05_droid')
    ax2.bar(x + w/2, fine_pos, w, color='#3498db', alpha=0.8, label='Finetuned')
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Final EE Position Diff (mm)")
    ax2.set_title("Final EE Position Divergence")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Δq={d:.2f}" for d in joint_diffs], fontsize=7, rotation=45)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 3. Final EE rotation diff comparison ---
    ax3 = fig.add_subplot(gs[0, 2])
    base_rot = [r["ee_comparison"]["rot_diff_final_deg"] for r in base["results"]]
    fine_rot = [r["ee_comparison"]["rot_diff_final_deg"] for r in fine["results"]]
    ax3.bar(x - w/2, base_rot, w, color='#e74c3c', alpha=0.8, label='Base pi05_droid')
    ax3.bar(x + w/2, fine_rot, w, color='#3498db', alpha=0.8, label='Finetuned')
    ax3.set_xlabel("Configuration")
    ax3.set_ylabel("Final EE Rotation Diff (deg)")
    ax3.set_title("Final EE Rotation Divergence")
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Δq={d:.2f}" for d in joint_diffs], fontsize=7, rotation=45)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- 4. EE position trajectory over horizon (base) ---
    ax4 = fig.add_subplot(gs[1, 0])
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(base["results"])))
    for i, r in enumerate(base["results"]):
        steps = np.arange(len(r["ee_comparison"]["pos_diff_per_step"]))
        pos_mm = np.array(r["ee_comparison"]["pos_diff_per_step"]) * 1000
        ax4.plot(steps, pos_mm, color=colors[i], linewidth=1.5,
                 label=f"Δq={r['joint_diff_rad']:.2f}" if i < 5 else None)
    ax4.set_xlabel("Action Step")
    ax4.set_ylabel("EE Position Diff (mm)")
    ax4.set_title("Base pi05_droid: EE Trajectory Divergence")
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)

    # --- 5. EE position trajectory over horizon (finetuned) ---
    ax5 = fig.add_subplot(gs[1, 1])
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(fine["results"])))
    for i, r in enumerate(fine["results"]):
        steps = np.arange(len(r["ee_comparison"]["pos_diff_per_step"]))
        pos_mm = np.array(r["ee_comparison"]["pos_diff_per_step"]) * 1000
        ax5.plot(steps, pos_mm, color=colors[i], linewidth=1.5,
                 label=f"Δq={r['joint_diff_rad']:.2f}" if i < 5 else None)
    ax5.set_xlabel("Action Step")
    ax5.set_ylabel("EE Position Diff (mm)")
    ax5.set_title("Finetuned: EE Trajectory Divergence")
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # --- 6. Summary scatter ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(joint_diffs, base_pos, c='#e74c3c', s=80, edgecolors='black',
                label='Base pi05_droid', zorder=3)
    ax6.scatter(joint_diffs, fine_pos, c='#3498db', s=80, edgecolors='black',
                label='Finetuned', zorder=3, marker='s')
    ax6.set_xlabel("Joint Config Diff (rad)")
    ax6.set_ylabel("Final EE Position Diff (mm)")
    ax6.set_title("Joint Diff vs EE Divergence")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # Summary stats as text
    base_mean_pos = np.mean(base_pos)
    fine_mean_pos = np.mean(fine_pos)
    ratio = base_mean_pos / fine_mean_pos if fine_mean_pos > 0 else float('inf')
    ax6.text(0.05, 0.95,
             f"Base avg: {base_mean_pos:.1f} mm\nFinetuned avg: {fine_mean_pos:.1f} mm\nRatio: {ratio:.1f}x",
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle("State Redundancy: Base pi05_droid vs Finetuned Model", fontsize=15, fontweight='bold')

    out = RESULTS_DIR / "comparison_base_vs_finetuned.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")

    # Print text summary
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Base pi05_droid:")
    print(f"    Action L2 diffs:    {min(base_action):.2f} - {max(base_action):.2f}")
    print(f"    Final EE pos diff:  {min(base_pos):.1f} - {max(base_pos):.1f} mm (avg {base_mean_pos:.1f})")
    print(f"    Final EE rot diff:  {min(base_rot):.1f} - {max(base_rot):.1f} deg")
    print(f"\n  Finetuned:")
    print(f"    Action L2 diffs:    {min(fine_action):.2f} - {max(fine_action):.2f}")
    print(f"    Final EE pos diff:  {min(fine_pos):.1f} - {max(fine_pos):.1f} mm (avg {fine_mean_pos:.1f})")
    print(f"    Final EE rot diff:  {min(fine_rot):.1f} - {max(fine_rot):.1f} deg")
    print(f"\n  Finetuning reduced EE divergence by ~{ratio:.1f}x")
    plt.close(fig)


if __name__ == "__main__":
    main()
