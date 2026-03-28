# State Redundancy Testing

Tests how the Franka Panda's 1-DOF kinematic redundancy (null space) affects policy action predictions.

## Concept

The Franka Panda has 7 joints but only needs 6 DOF for a full end-effector pose, leaving 1 degree of redundancy. This means infinitely many joint configurations can achieve the exact same end-effector position and orientation. A robust policy should predict actions that produce similar end-effector trajectories regardless of which null-space configuration the robot is in.

## Files

- `test_state_redundancy.py` — Main test script. Finds null-space joint configurations via IK, runs policy inference on each, and compares resulting EE trajectories.
- `visualize_results.py` — Generates plots from the JSON results.
- `results/` — Output directory for results and plots.

## Usage

### Kinematics-only test (no model checkpoint required)
```bash
python testing_state_redundancy/test_state_redundancy.py --num-null-space-samples 5
```

### Full test with policy inference
```bash
python testing_state_redundancy/test_state_redundancy.py \
    --config-name pi05_lab_finetune_orange_cube_single_point \
    --checkpoint-path /path/to/checkpoint \
    --num-null-space-samples 5
```

### Visualize results
```bash
python testing_state_redundancy/visualize_results.py
```

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--config-name` | `pi05_lab_finetune_orange_cube_single_point` | Training config name |
| `--checkpoint-path` | None | Path to model checkpoint (omit for kinematics-only) |
| `--num-null-space-samples` | 5 | Number of null-space configurations to test |
| `--null-space-magnitude` | 0.5 | How far to perturb in null space (radians) |
| `--dt` | 0.05 | Integration timestep for joint velocity simulation |
| `--seed` | 42 | Random seed |

## How it works

1. Load a reference joint configuration (from sample data or default).
2. Compute the Jacobian null space at that configuration.
3. Use IK to find alternative joint configs along the null space that achieve the same EE pose.
4. For each config, run `policy.infer()` with identical images but different joint angles.
5. Simulate forward kinematics: integrate predicted joint velocities over the action horizon.
6. Compare resulting EE trajectories (position and orientation).
