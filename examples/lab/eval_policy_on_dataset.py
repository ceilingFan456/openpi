#!/usr/bin/env python3
"""Offline validation for OpenPI policies on a LeRobot dataset.

Runs policy inference on dataset samples and computes absolute action error (MAE)
against ground-truth actions. Supports one checkpoint or all checkpoints in a
training run directory to track progress over training steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import re
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# Allow running this script directly from the repo root without installing the package.
# REPO_ROOT = Path(__file__).resolve().parents[2]
# SRC_DIR = REPO_ROOT / "src"
# if str(SRC_DIR) not in sys.path:
#     sys.path.insert(0, str(SRC_DIR))

from openpi.policies import policy_config
from openpi.shared import download
from openpi.models import model as model_lib
from openpi.training import config as train_config_lib
from openpi.training import data_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy MAE on a LeRobot dataset.")
    parser.add_argument("--config-name", type=str, required=True, help="Training config name, e.g. pi05_lab_finetune_...")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Single checkpoint path (local or gs://...).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing per-step checkpoint folders (e.g. .../<exp_name>/).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Optional dataset repo override. If omitted, use repo from config.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Max number of samples to evaluate (0 means all).")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset.")
    parser.add_argument("--stride", type=int, default=1, help="Sample stride in dataset.")
    parser.add_argument(
        "--require-use-policy",
        action="store_true",
        help="If set, only evaluate samples where use_policy is true (if present in data).",
    )
    parser.add_argument(
        "--pytorch-device",
        type=str,
        default=None,
        help='Device for torch checkpoints (e.g. "cuda", "cuda:0", "cpu").',
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path for JSON summary.",
    )
    parser.add_argument(
        "--eval-aux-2d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate auxiliary 2D keypoint prediction quality when available.",
    )
    parser.add_argument(
        "--aux-require-use-auxiliary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For aux eval, only score samples where use_auxiliary is true.",
    )
    return parser.parse_args()


def _step_from_path(path: Path) -> int:
    if path.name.isdigit():
        return int(path.name)
    match = re.search(r"(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _is_checkpoint_dir(path: Path) -> bool:
    return (path / "params").exists() or (path / "model.safetensors").exists()


def resolve_checkpoint_dirs(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_path is not None:
        ckpt = Path(download.maybe_download(args.checkpoint_path))
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")
        return [ckpt]

    if args.checkpoint_dir is None:
        raise ValueError("Set either --checkpoint-path or --checkpoint-dir.")

    root = args.checkpoint_dir.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {root}")

    candidates = [p for p in root.iterdir() if p.is_dir() and _is_checkpoint_dir(p)]
    if not candidates:
        raise ValueError(f"No checkpoint subdirectories found in {root}")

    return sorted(candidates, key=_step_from_path)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sample_use_policy(sample: dict[str, Any]) -> bool:
    if "use_policy" not in sample:
        return True
    use_policy = to_numpy(sample["use_policy"]).reshape(-1)
    if use_policy.size == 0:
        return True
    return bool(use_policy[0])


def sample_use_auxiliary(sample: dict[str, Any]) -> bool:
    if "use_auxiliary" not in sample:
        return True
    use_aux = to_numpy(sample["use_auxiliary"]).reshape(-1)
    if use_aux.size == 0:
        return True
    return bool(use_aux[0])


def infer_aux_predictions(policy, obs: dict[str, Any]) -> np.ndarray | None:
    """Return predicted aux keypoints with shape [H, 2], or None if unsupported."""
    if getattr(policy, "_is_pytorch_model", False):
        return None
    model = getattr(policy, "_model", None)
    if model is None or not hasattr(model, "_compute_aux_predictions"):
        return None
    if not bool(getattr(model, "enable_aux_2d", False)):
        return None

    # Reuse the same transform pipeline as policy.infer, then run the aux head.
    ## this prevents double preprocessing. 
    inputs = jax.tree.map(lambda x: x, obs)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    observation = model_lib.Observation.from_dict(inputs)
    observation = model_lib.preprocess_observation(None, observation, train=False)

    pred_2d, _ = model._compute_aux_predictions(observation, train=False)
    return np.asarray(pred_2d[0])


def evaluate_checkpoint(
    policy,
    dataset,
    *,
    indices: list[int],
    require_use_policy: bool,
    eval_aux_2d: bool,
    aux_require_use_auxiliary: bool,
    action_key: str = "actions",
) -> dict[str, Any]:
    num_used = 0
    skipped = 0
    abs_sum = 0.0
    abs_sum_first = 0.0
    count = 0
    count_first = 0
    per_dim_abs_sum: np.ndarray | None = None
    per_dim_abs_sum_first: np.ndarray | None = None
    per_dim_count: np.ndarray | None = None
    per_dim_count_first: np.ndarray | None = None
    aux_supported = False
    aux_samples_used = 0
    aux_samples_skipped = 0
    aux_abs_sum = 0.0
    aux_abs_count = 0
    aux_l2_sum = 0.0
    aux_point_count = 0

    for i in indices:
        sample = dataset[i]
        if require_use_policy and not sample_use_policy(sample):
            skipped += 1
            continue

        if action_key not in sample:
            raise KeyError(f"Sample at index {i} missing action key '{action_key}'")

        gt_actions = to_numpy(sample[action_key]).astype(np.float32)
        obs = {k: v for k, v in sample.items() if k != action_key}
        pred_actions = to_numpy(policy.infer(obs)["actions"]).astype(np.float32)

        if gt_actions.ndim == 1:
            gt_actions = gt_actions[None, :]
        if pred_actions.ndim == 1:
            pred_actions = pred_actions[None, :]

        t = min(gt_actions.shape[0], pred_actions.shape[0])
        d = min(gt_actions.shape[1], pred_actions.shape[1])
        if t <= 0 or d <= 0:
            skipped += 1
            continue

        abs_err = np.abs(pred_actions[:t, :d] - gt_actions[:t, :d])
        first_abs_err = np.abs(pred_actions[0, :d] - gt_actions[0, :d])

        abs_sum += float(abs_err.sum())
        abs_sum_first += float(first_abs_err.sum())
        count += int(abs_err.size)
        count_first += int(first_abs_err.size)
        num_used += 1

        if per_dim_abs_sum is None:
            per_dim_abs_sum = np.zeros((d,), dtype=np.float64)
            per_dim_abs_sum_first = np.zeros((d,), dtype=np.float64)
            per_dim_count = np.zeros((d,), dtype=np.int64)
            per_dim_count_first = np.zeros((d,), dtype=np.int64)

        assert per_dim_count is not None
        assert per_dim_count_first is not None

        if d < per_dim_abs_sum.shape[0]:
            per_dim_abs_sum = per_dim_abs_sum[:d]
            per_dim_abs_sum_first = per_dim_abs_sum_first[:d]
            per_dim_count = per_dim_count[:d]
            per_dim_count_first = per_dim_count_first[:d]
        elif d > per_dim_abs_sum.shape[0]:
            d = per_dim_abs_sum.shape[0]
            abs_err = abs_err[:, :d]
            first_abs_err = first_abs_err[:d]

        per_dim_abs_sum += abs_err.sum(axis=0)
        per_dim_abs_sum_first += first_abs_err
        per_dim_count += abs_err.shape[0]
        per_dim_count_first += 1

        if eval_aux_2d and "aux_keypoints_2d" in sample:
            if aux_require_use_auxiliary and not sample_use_auxiliary(sample):
                aux_samples_skipped += 1
                continue

            pred_aux = infer_aux_predictions(policy, obs)
            if pred_aux is None:
                aux_samples_skipped += 1
                continue
            aux_supported = True

            target_aux = to_numpy(sample["aux_keypoints_2d"]).astype(np.float32)
            if target_aux.ndim == 2:
                target_aux = target_aux
            elif target_aux.ndim == 3:
                target_aux = target_aux[0]
            else:
                aux_samples_skipped += 1
                continue

            if "aux_keypoints_mask" in sample:
                point_mask = to_numpy(sample["aux_keypoints_mask"]).astype(bool).reshape(-1)
            else:
                point_mask = np.ones((target_aux.shape[0],), dtype=bool)

            h = min(pred_aux.shape[0], target_aux.shape[0], point_mask.shape[0])
            if h <= 0:
                aux_samples_skipped += 1
                continue

            pred_aux_h = pred_aux[:h]
            target_aux_h = target_aux[:h]
            valid_mask = point_mask[:h]
            if not np.any(valid_mask):
                aux_samples_skipped += 1
                continue

            diff = pred_aux_h[valid_mask] - target_aux_h[valid_mask]
            abs_diff = np.abs(diff)
            aux_abs_sum += float(abs_diff.sum())
            aux_abs_count += int(abs_diff.size)
            aux_l2_sum += float(np.linalg.norm(diff, axis=-1).sum())
            aux_point_count += int(diff.shape[0])
            aux_samples_used += 1

    if (
        num_used == 0
        or count == 0
        or per_dim_abs_sum is None
        or per_dim_abs_sum_first is None
        or per_dim_count is None
        or per_dim_count_first is None
    ):
        empty_result = {
            "num_samples_used": 0,
            "num_samples_skipped": skipped,
            "mae_all_actions": None,
            "mae_first_action": None,
            "mae_per_dim_all_actions": [],
            "mae_per_dim_first_action": [],
        }
        if eval_aux_2d:
            empty_result.update(
                {
                    "aux_eval_enabled": True,
                    "aux_supported": aux_supported,
                    "aux_samples_used": 0,
                    "aux_samples_skipped": aux_samples_skipped,
                    "aux_mae_xy": None,
                    "aux_mean_l2": None,
                }
            )
        return empty_result

    result = {
        "num_samples_used": num_used,
        "num_samples_skipped": skipped,
        "mae_all_actions": abs_sum / count,
        "mae_first_action": abs_sum_first / count_first,
        "mae_per_dim_all_actions": (per_dim_abs_sum / np.maximum(per_dim_count, 1)).tolist(),
        "mae_per_dim_first_action": (per_dim_abs_sum_first / np.maximum(per_dim_count_first, 1)).tolist(),
    }
    if eval_aux_2d:
        result.update(
            {
                "aux_eval_enabled": True,
                "aux_supported": aux_supported,
                "aux_samples_used": aux_samples_used,
                "aux_samples_skipped": aux_samples_skipped,
                "aux_mae_xy": (aux_abs_sum / aux_abs_count) if aux_abs_count > 0 else None,
                "aux_mean_l2": (aux_l2_sum / aux_point_count) if aux_point_count > 0 else None,
            }
        )
    return result


def main() -> None:
    args = parse_args()

    cfg = train_config_lib.get_config(args.config_name)
    if args.repo_id is not None:
        cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, repo_id=args.repo_id))

    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(data_cfg, cfg.model.action_horizon, cfg.model)

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    all_indices = list(range(args.start_index, len(dataset), args.stride))
    indices = all_indices if args.max_samples <= 0 else all_indices[: args.max_samples]
    if not indices:
        raise ValueError("No samples selected. Check --start-index/--max-samples/--stride.")

    checkpoint_dirs = resolve_checkpoint_dirs(args)
    print(f"Config: {args.config_name}")
    print(f"Dataset repo: {data_cfg.repo_id}")
    print(f"Samples selected: {len(indices)} / {len(dataset)}")
    print(f"Checkpoints to evaluate: {len(checkpoint_dirs)}")

    results: list[dict[str, Any]] = []
    for ckpt_dir in checkpoint_dirs:
        print(f"\nEvaluating checkpoint: {ckpt_dir}")
        policy = policy_config.create_trained_policy(
            cfg,
            ckpt_dir,
            pytorch_device=args.pytorch_device,
        )
        metrics = evaluate_checkpoint(
            policy,
            dataset,
            indices=indices,
            require_use_policy=args.require_use_policy,
            eval_aux_2d=args.eval_aux_2d,
            aux_require_use_auxiliary=args.aux_require_use_auxiliary,
        )
        row = {
            "checkpoint_dir": str(ckpt_dir),
            "step": _step_from_path(ckpt_dir),
            **metrics,
        }
        results.append(row)

        print(
            f"  used={row['num_samples_used']} skipped={row['num_samples_skipped']} "
            f"mae_all={row['mae_all_actions']} mae_first={row['mae_first_action']}"
        )
        if args.eval_aux_2d:
            print(
                f"  aux_supported={row.get('aux_supported')} aux_used={row.get('aux_samples_used')} "
                f"aux_mae_xy={row.get('aux_mae_xy')} aux_mean_l2={row.get('aux_mean_l2')}"
            )

    print("\nSummary:")
    for row in sorted(results, key=lambda x: x["step"]):
        print(
            f"  step={row['step']:>6} "
            f"mae_all={row['mae_all_actions']} "
            f"mae_first={row['mae_first_action']}"
        )
        if args.eval_aux_2d:
            print(
                f"            aux_supported={row.get('aux_supported')} "
                f"aux_mae_xy={row.get('aux_mae_xy')} aux_mean_l2={row.get('aux_mean_l2')}"
            )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(
                {
                    "config_name": args.config_name,
                    "dataset_repo": data_cfg.repo_id,
                    "num_selected_samples": len(indices),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\nWrote: {args.output_json}")


if __name__ == "__main__":
    main()

# uv run python examples/lab/eval_policy_on_dataset.py \
#   --config-name pi05_aux2d_human \
#   --checkpoint-dir /path/to/checkpoints/pi05_aux2d_human/<exp_name> \
#   --max-samples 300 \
#   --require-use-policy \
#   --eval-aux-2d \
#   --aux-require-use-auxiliary \
#   --output-json /tmp/pi05_aux2d_eval.json
