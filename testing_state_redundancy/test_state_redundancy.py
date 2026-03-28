"""
Test how Franka's kinematic redundancy affects policy action predictions.

The Franka Panda has 7 DOF but only 6 DOF are needed for a full 6D end-effector pose,
giving 1 degree of redundancy (the null-space). This script tests whether different
joint configurations that share the same end-effector pose produce similar predicted
end-effector trajectories after executing the predicted action chunks.

Workflow:
  1. Start from a reference joint configuration q_ref.
  2. Use IK to find alternative joint configurations q_alt that achieve the same EE pose.
  3. Feed (same images, q_ref) and (same images, q_alt) to the policy.
  4. Get predicted action chunks (joint velocities) for both.
  5. Simulate forward: q_new = q + qdot * dt for each step in the action horizon.
  6. Compute FK on the resulting joint trajectories to get EE trajectories.
  7. Compare the EE trajectories.

Usage:
    python testing_state_redundancy/test_state_redundancy.py \
        --config-name pi05_lab_finetune_orange_cube_single_point \
        --checkpoint-path /path/to/checkpoint \
        --num-null-space-samples 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image


def load_panda():
    """Load the Franka Panda robot model."""
    return rtb.models.Panda()


def fkine_pose(panda, q):
    """Compute FK and return (position, quaternion) for a joint config."""
    T = panda.fkine(q)
    pos = T.t
    rot = R.from_matrix(T.R)
    quat = rot.as_quat()  # [qx, qy, qz, qw]
    return pos, quat


def find_null_space_configs(panda, q_ref, num_samples=5, magnitude=0.3, seed=42):
    """
    Find alternative joint configurations in the null space of q_ref.

    Uses IK with different initial guesses seeded in the null space direction.
    The Franka has a 1-DOF null space, so we perturb along the null space
    direction and re-solve IK.
    """
    T_target = panda.fkine(q_ref)
    rng = np.random.RandomState(seed)

    # Compute the null space of the Jacobian at q_ref
    J = panda.jacob0(q_ref)
    U, S, Vt = np.linalg.svd(J)
    # Null space vectors are the last rows of Vt (corresponding to near-zero singular values)
    # For 7-DOF robot with 6-DOF task, there's 1 null-space dimension
    null_space = Vt[-1, :]  # (7,)

    configs = []
    for i in range(num_samples):
        # Perturb along null space with varying magnitude
        alpha = magnitude * (i + 1) / num_samples
        # Alternate sign
        sign = 1 if i % 2 == 0 else -1
        q_init = q_ref + sign * alpha * null_space

        # Also add some random perturbation to escape local minima
        q_init += rng.randn(7) * 0.05

        sol = panda.ikine_LM(T_target, q0=q_init, ilimit=500, slimit=100)

        if sol.success:
            # Verify the solution has the same EE pose
            T_check = panda.fkine(sol.q)
            pos_err = np.linalg.norm(T_target.t - T_check.t)
            rot_err = np.linalg.norm(T_target.R - T_check.R)

            # Check joint limits (approximate Franka limits)
            q_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
            q_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            in_limits = np.all(sol.q >= q_lower) and np.all(sol.q <= q_upper)

            if pos_err < 1e-4 and rot_err < 1e-4:
                joint_diff = np.linalg.norm(sol.q - q_ref)
                configs.append({
                    "q": sol.q,
                    "pos_err": pos_err,
                    "rot_err": rot_err,
                    "joint_diff": joint_diff,
                    "in_limits": in_limits,
                })

    # Sort by joint difference (most different first)
    configs.sort(key=lambda x: -x["joint_diff"])

    # Deduplicate: remove configs that are too similar to each other
    unique_configs = []
    for cfg in configs:
        is_duplicate = False
        for uc in unique_configs:
            if np.linalg.norm(cfg["q"] - uc["q"]) < 0.05:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_configs.append(cfg)

    return unique_configs


def simulate_trajectory(q_start, action_chunk, dt=0.05):
    """
    Simulate joint trajectory by integrating joint velocities.

    Args:
        q_start: Initial joint positions (7,)
        action_chunk: (horizon, 8) action chunk [qdot(7), gripper(1)]
        dt: Integration timestep

    Returns:
        q_trajectory: (horizon+1, 7) joint positions over time (including start)
    """
    q_traj = [q_start.copy()]
    q = q_start.copy()

    for t in range(action_chunk.shape[0]):
        qdot = action_chunk[t, :7]
        q = q + qdot * dt
        q_traj.append(q.copy())

    return np.array(q_traj)


def trajectory_to_ee(panda, q_trajectory):
    """
    Convert joint trajectory to end-effector trajectory.

    Returns:
        ee_positions: (T, 3) xyz positions
        ee_orientations: (T, 4) quaternions [qx, qy, qz, qw]
    """
    positions = []
    orientations = []

    for q in q_trajectory:
        pos, quat = fkine_pose(panda, q)
        positions.append(pos)
        orientations.append(quat)

    return np.array(positions), np.array(orientations)


def compare_ee_trajectories(pos1, quat1, pos2, quat2):
    """
    Compare two EE trajectories.

    Returns dict with comparison metrics.
    """
    # Position differences
    pos_diff = np.linalg.norm(pos1 - pos2, axis=1)

    # Orientation differences (geodesic distance between quaternions)
    rot_diffs = []
    for q1, q2 in zip(quat1, quat2):
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        r_diff = r1.inv() * r2
        angle = r_diff.magnitude()
        rot_diffs.append(angle)
    rot_diffs = np.array(rot_diffs)

    return {
        "pos_diff_mean": float(np.mean(pos_diff)),
        "pos_diff_max": float(np.max(pos_diff)),
        "pos_diff_final": float(pos_diff[-1]),
        "rot_diff_mean_deg": float(np.degrees(np.mean(rot_diffs))),
        "rot_diff_max_deg": float(np.degrees(np.max(rot_diffs))),
        "rot_diff_final_deg": float(np.degrees(rot_diffs[-1])),
        "pos_diff_per_step": pos_diff.tolist(),
        "rot_diff_per_step_deg": np.degrees(rot_diffs).tolist(),
    }


def load_sample_images(sample_dir):
    """Load sample images from the sample_first_added_frame directory."""
    img_paths = {
        "exterior_image_1_left": os.path.join(sample_dir, "exterior_image_1_left.png"),
        "wrist_image_left": os.path.join(sample_dir, "wrist_image_left.png"),
        "exterior_image_2_left": os.path.join(sample_dir, "exterior_image_2_left.png"),
    }

    images = {}
    for key, path in img_paths.items():
        if os.path.exists(path):
            img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
            images[key] = img
        else:
            print(f"  [WARNING] Image not found: {path}, using zeros")
            images[key] = np.zeros((224, 224, 3), dtype=np.uint8)

    return images


def build_observation(images, joint_positions, gripper_position, prompt):
    """
    Build observation dict in the format expected by the policy.

    The policy uses DroidInputs which expects:
      - observation/exterior_image_1_left
      - observation/wrist_image_left
      - observation/exterior_image_2_left  (for double_view configs)
      - observation/joint_position (7D)
      - observation/gripper_position (1D)
      - prompt
    """
    obs = {
        "observation/exterior_image_1_left": images["exterior_image_1_left"],
        "observation/wrist_image_left": images["wrist_image_left"],
        "observation/exterior_image_2_left": images.get("exterior_image_2_left",
                                                         np.zeros((224, 224, 3), dtype=np.uint8)),
        "observation/joint_position": np.array(joint_positions, dtype=np.float32),
        "observation/gripper_position": np.array(gripper_position, dtype=np.float32),
        "prompt": prompt,
    }
    return obs


def run_test(args):
    print("=" * 80)
    print("  STATE REDUNDANCY TEST")
    print("  Testing how null-space joint configurations affect action predictions")
    print("=" * 80)

    # --- Load robot model ---
    print("\n[1/5] Loading Franka Panda model...")
    panda = load_panda()
    print(f"  Robot: {panda.name}, {panda.n} DOF")

    # --- Set up reference joint configuration ---
    print("\n[2/5] Setting up reference configuration...")

    # Load sample data for reference joint angles and images
    sample_dir = os.path.join(PROJECT_ROOT,
                              "sample_first_added_frame/ceilingfan456/lab_data_orange_cube_single_point_paired_25")
    metadata_path = os.path.join(sample_dir, "frame_metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        q_ref = np.array(metadata["joint_position"])
        gripper_pos = np.array(metadata["gripper_position"])
        prompt = metadata.get("task", "Place the orange cube onto the green coaster.")
        print(f"  Loaded reference from sample data")
    else:
        # Fallback: use a reasonable Franka home-like config
        q_ref = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
        gripper_pos = np.array([0.04])
        prompt = "Place the orange cube onto the green coaster."
        print(f"  Using default reference configuration")

    # Compute reference EE pose
    ref_pos, ref_quat = fkine_pose(panda, q_ref)
    print(f"  q_ref: {np.round(q_ref, 4)}")
    print(f"  EE position: {np.round(ref_pos, 4)}")
    print(f"  EE quaternion: {np.round(ref_quat, 4)}")

    # --- Find null-space configurations ---
    print(f"\n[3/5] Finding {args.num_null_space_samples} null-space configurations...")
    null_configs = find_null_space_configs(
        panda, q_ref,
        num_samples=args.num_null_space_samples * 3,  # oversample then pick best
        magnitude=args.null_space_magnitude,
        seed=args.seed,
    )

    if not null_configs:
        print("  [ERROR] Could not find any valid null-space configurations!")
        print("  Try increasing --null-space-magnitude or --num-null-space-samples")
        return

    # Take the top N most different configs
    null_configs = null_configs[:args.num_null_space_samples]

    print(f"  Found {len(null_configs)} unique configurations:")
    for i, cfg in enumerate(null_configs):
        print(f"    Config {i+1}: joint_diff={cfg['joint_diff']:.4f} rad, "
              f"pos_err={cfg['pos_err']:.2e} m, rot_err={cfg['rot_err']:.2e}, "
              f"in_limits={cfg['in_limits']}")
        print(f"              q={np.round(cfg['q'], 4)}")

    # --- Load policy ---
    if args.checkpoint_path is not None:
        print(f"\n[4/5] Loading policy '{args.config_name}'...")
        print(f"  Checkpoint: {args.checkpoint_path}")

        # Heavy imports only when needed
        from openpi.training import config as _config
        from openpi.policies import policy_config

        train_config = _config.get_config(args.config_name)
        policy = policy_config.create_trained_policy(train_config, args.checkpoint_path)
        print("  Policy loaded successfully")
        use_policy = True
    else:
        print(f"\n[4/5] No checkpoint specified, running kinematics-only test...")
        print("  (Use --checkpoint-path to run with actual policy inference)")
        use_policy = False

    # --- Load images ---
    print(f"\n  Loading sample images from: {sample_dir}")
    images = load_sample_images(sample_dir)
    for key, img in images.items():
        print(f"    {key}: {img.shape}, dtype={img.dtype}")

    # --- Run inference for each configuration ---
    print(f"\n[5/5] Running inference and comparing trajectories...")

    dt = args.dt
    all_results = []

    # Reference inference
    obs_ref = build_observation(images, q_ref, gripper_pos, prompt)

    if use_policy:
        result_ref = policy.infer(obs_ref)
        actions_ref = result_ref["actions"]  # (horizon, 8)
        print(f"\n  Reference actions shape: {actions_ref.shape}")
        print(f"  Reference actions (first step): {np.round(actions_ref[0], 6)}")
    else:
        # Dummy actions for kinematics-only testing
        actions_ref = np.random.randn(16, 8) * 0.01
        actions_ref[:, 7] = 0.0  # gripper
        print(f"\n  Using random dummy actions for kinematics test")

    # Simulate reference trajectory
    q_traj_ref = simulate_trajectory(q_ref, actions_ref, dt=dt)
    ee_pos_ref, ee_quat_ref = trajectory_to_ee(panda, q_traj_ref)

    print(f"  Reference EE trajectory:")
    print(f"    Start: {np.round(ee_pos_ref[0], 4)}")
    print(f"    End:   {np.round(ee_pos_ref[-1], 4)}")
    print(f"    Total displacement: {np.linalg.norm(ee_pos_ref[-1] - ee_pos_ref[0]):.6f} m")

    # Run for each null-space config
    for i, cfg in enumerate(null_configs):
        q_alt = cfg["q"]
        print(f"\n  --- Config {i+1} (joint_diff={cfg['joint_diff']:.4f} rad) ---")

        obs_alt = build_observation(images, q_alt, gripper_pos, prompt)

        if use_policy:
            result_alt = policy.infer(obs_alt)
            actions_alt = result_alt["actions"]
        else:
            actions_alt = np.random.randn(16, 8) * 0.01
            actions_alt[:, 7] = 0.0

        # Compare raw actions
        action_diff = np.linalg.norm(actions_ref - actions_alt)
        action_diff_per_step = np.linalg.norm(actions_ref[:, :7] - actions_alt[:, :7], axis=1)
        print(f"  Action chunk diff (L2 norm): {action_diff:.6f}")
        print(f"  Action diff per step (mean): {np.mean(action_diff_per_step):.6f}")
        print(f"  Actions (first step): {np.round(actions_alt[0], 6)}")

        # Simulate trajectory
        q_traj_alt = simulate_trajectory(q_alt, actions_alt, dt=dt)
        ee_pos_alt, ee_quat_alt = trajectory_to_ee(panda, q_traj_alt)

        # Compare EE trajectories
        comparison = compare_ee_trajectories(ee_pos_ref, ee_quat_ref, ee_pos_alt, ee_quat_alt)

        print(f"  EE trajectory comparison:")
        print(f"    Position diff - mean: {comparison['pos_diff_mean']*1000:.3f} mm, "
              f"max: {comparison['pos_diff_max']*1000:.3f} mm, "
              f"final: {comparison['pos_diff_final']*1000:.3f} mm")
        print(f"    Rotation diff - mean: {comparison['rot_diff_mean_deg']:.3f} deg, "
              f"max: {comparison['rot_diff_max_deg']:.3f} deg, "
              f"final: {comparison['rot_diff_final_deg']:.3f} deg")

        result_entry = {
            "config_idx": i,
            "q_ref": q_ref.tolist(),
            "q_alt": q_alt.tolist(),
            "joint_diff_rad": float(cfg["joint_diff"]),
            "in_limits": bool(cfg["in_limits"]),
            "action_chunk_diff_l2": float(action_diff),
            "action_diff_per_step_mean": float(np.mean(action_diff_per_step)),
            "ee_comparison": comparison,
            "actions_ref_first": actions_ref[0].tolist(),
            "actions_alt_first": actions_alt[0].tolist(),
            "ee_traj_ref_start": ee_pos_ref[0].tolist(),
            "ee_traj_ref_end": ee_pos_ref[-1].tolist(),
            "ee_traj_alt_start": ee_pos_alt[0].tolist(),
            "ee_traj_alt_end": ee_pos_alt[-1].tolist(),
        }
        all_results.append(result_entry)

    # --- Summary ---
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    if all_results:
        pos_diffs_final = [r["ee_comparison"]["pos_diff_final"] for r in all_results]
        rot_diffs_final = [r["ee_comparison"]["rot_diff_final_deg"] for r in all_results]
        action_diffs = [r["action_chunk_diff_l2"] for r in all_results]
        joint_diffs = [r["joint_diff_rad"] for r in all_results]

        print(f"\n  Across {len(all_results)} null-space configurations:")
        print(f"  Joint angle differences:    {np.min(joint_diffs):.4f} - {np.max(joint_diffs):.4f} rad")
        print(f"  Action chunk L2 diffs:      {np.min(action_diffs):.6f} - {np.max(action_diffs):.6f}")
        print(f"  Final EE position diffs:    {np.min(pos_diffs_final)*1000:.3f} - {np.max(pos_diffs_final)*1000:.3f} mm")
        print(f"  Final EE rotation diffs:    {np.min(rot_diffs_final):.3f} - {np.max(rot_diffs_final):.3f} deg")

        # Interpretation
        max_pos_mm = np.max(pos_diffs_final) * 1000
        max_rot_deg = np.max(rot_diffs_final)

        print(f"\n  Interpretation:")
        if max_pos_mm < 1.0 and max_rot_deg < 1.0:
            print(f"    The model is ROBUST to null-space variations.")
            print(f"    Different joint configs with same EE pose produce nearly identical EE trajectories.")
        elif max_pos_mm < 5.0 and max_rot_deg < 5.0:
            print(f"    The model shows MODERATE sensitivity to null-space variations.")
            print(f"    EE trajectories differ somewhat but remain close.")
        else:
            print(f"    The model is SENSITIVE to null-space variations.")
            print(f"    Different joint configs (same EE) produce significantly different EE trajectories.")
            print(f"    This suggests the model has learned joint-space-specific behavior.")

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, "testing_state_redundancy", "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "state_redundancy_results.json")

    save_data = {
        "config_name": args.config_name,
        "checkpoint_path": args.checkpoint_path,
        "dt": dt,
        "num_configs": len(all_results),
        "prompt": prompt,
        "reference_ee_position": ref_pos.tolist(),
        "reference_ee_quaternion": ref_quat.tolist(),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test state redundancy effects on policy predictions")
    parser.add_argument("--config-name", type=str, default="pi05_lab_finetune_orange_cube_single_point",
                        help="Training config name (must match a registered config)")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to model checkpoint. If not provided, runs kinematics-only test.")
    parser.add_argument("--num-null-space-samples", type=int, default=5,
                        help="Number of null-space configurations to test")
    parser.add_argument("--null-space-magnitude", type=float, default=0.5,
                        help="Magnitude of null-space perturbation (radians)")
    parser.add_argument("--dt", type=float, default=0.05,
                        help="Integration timestep for simulating joint velocities")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
