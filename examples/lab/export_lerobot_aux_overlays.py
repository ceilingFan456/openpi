#!/usr/bin/env python3
"""Export LeRobot frames with auxiliary 2D keypoint overlays.

Example:
  uv run python examples/lab/export_lerobot_aux_overlays.py \
    --repo-id ceilingfan456/lab_data_paired_64 \
    --output-dir /tmp/lab_overlays \
    --image-key exterior_image_1_left
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo id.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Optional local dataset root. If omitted, use LeRobot default cache/root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save rendered frames and metadata.",
    )
    parser.add_argument(
        "--image-key",
        type=str,
        default="exterior_image_1_left",
        help="Image observation key to visualize.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start dataset index.")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to export (0 = all).")
    parser.add_argument("--stride", type=int, default=1, help="Sample every N frames.")
    parser.add_argument(
        "--draw-invalid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also draw invalid points in red (default: false).",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=3,
        help="Point radius in pixels.",
    )
    return parser.parse_args()


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_hwc_uint8(image_like: Any) -> np.ndarray:
    image = to_numpy(image_like)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")

    # Convert CHW -> HWC if needed.
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected image channels=3, got shape={image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.nanmax(image)) if image.size > 0 else 1.0
        if max_val <= 1.0:
            image = np.clip(image * 255.0, 0.0, 255.0)
        else:
            image = np.clip(image, 0.0, 255.0)
        image = image.astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def scalar_int(x: Any, default: int = -1) -> int:
    if x is None:
        return default
    arr = to_numpy(x).reshape(-1)
    if arr.size == 0:
        return default
    return int(arr[0])


def draw_points(
    image_hwc: np.ndarray,
    points_xy: np.ndarray,
    valid_mask: np.ndarray,
    *,
    radius: int,
    draw_invalid: bool,
) -> np.ndarray:
    h, w = image_hwc.shape[:2]
    canvas = Image.fromarray(image_hwc)
    draw = ImageDraw.Draw(canvas)

    horizon = min(points_xy.shape[0], valid_mask.shape[0])
    for i in range(horizon):
        xy = points_xy[i]
        is_valid = bool(valid_mask[i])
        if not is_valid and not draw_invalid:
            continue

        x = float(xy[0])
        y = float(xy[1])

        # Keep only points that are finite and inside image bounds.
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if not (0.0 <= x < w and 0.0 <= y < h):
            continue

        if is_valid:
            # Green -> cyan across horizon for easy temporal reading.
            t = i / max(1, horizon - 1)
            color = (0, int(255 * (1.0 - t)), 255)
        else:
            color = (255, 0, 0)

        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            outline=color,
            fill=color,
        )

    return np.array(canvas)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(args.repo_id, root=args.root)
    n = len(dataset)
    print(f"Loaded dataset: repo_id={args.repo_id}, num_frames={n}")

    if args.start_index < 0 or args.start_index >= n:
        raise ValueError(f"--start-index out of range: {args.start_index} (dataset size={n})")
    if args.stride <= 0:
        raise ValueError("--stride must be >= 1")

    selected = list(range(args.start_index, n, args.stride))
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    manifest_path = args.output_dir / "manifest.jsonl"
    total_valid_points = 0
    total_frames = 0
    skipped_no_aux = 0
    skipped_no_image = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx in selected:
            sample = dataset[idx]

            if args.image_key not in sample:
                skipped_no_image += 1
                continue
            if "aux_keypoints_2d" not in sample:
                skipped_no_aux += 1
                continue

            image = to_hwc_uint8(sample[args.image_key])
            points = to_numpy(sample["aux_keypoints_2d"]).astype(np.float32)

            if "aux_keypoints_mask" in sample:
                valid = to_numpy(sample["aux_keypoints_mask"]).astype(bool).reshape(-1)
            else:
                valid = np.ones((points.shape[0],), dtype=bool)

            overlaid = draw_points(
                image,
                points,
                valid,
                radius=args.radius,
                draw_invalid=args.draw_invalid,
            )

            ep_idx = scalar_int(sample.get("episode_index"), default=-1)
            frame_idx = scalar_int(sample.get("frame_index"), default=idx)
            valid_count = int(valid.sum())

            out_name = f"ep_{ep_idx:04d}_frame_{frame_idx:06d}_idx_{idx:06d}.png"
            out_path = args.output_dir / out_name
            Image.fromarray(overlaid).save(out_path)

            record = {
                "dataset_index": idx,
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "image_key": args.image_key,
                "aux_horizon": int(points.shape[0]),
                "valid_points": valid_count,
                "output_file": out_name,
            }
            mf.write(json.dumps(record) + "\n")

            total_valid_points += valid_count
            total_frames += 1

    avg_valid = (total_valid_points / float(total_frames)) if total_frames > 0 else 0.0
    print(f"Exported {total_frames} frames to {args.output_dir}")
    print(f"Average valid points per exported frame: {avg_valid:.3f}")
    print(f"Manifest: {manifest_path}")
    if skipped_no_image > 0:
        print(f"Skipped {skipped_no_image} samples without image key '{args.image_key}'")
    if skipped_no_aux > 0:
        print(f"Skipped {skipped_no_aux} samples without 'aux_keypoints_2d'")


if __name__ == "__main__":
    main()

