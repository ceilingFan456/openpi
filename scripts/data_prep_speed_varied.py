"""Data preparation: Create speed-varied Libero dataset.

Creates three variants (speed 0.5, 1.0, 2.0) and a mixed training dataset.
Outputs to /mnt/default_storage/qiming/datasets/libero_speed_varied/
"""
import json
import os
import shutil
import sys

import numpy as np
from pathlib import Path


def compute_new_length(T, speed_factor, min_length=16, max_length=None):
    if max_length is None:
        max_length = 3 * T
    T_new = int(round(T / speed_factor))
    return max(min_length, min(T_new, max_length))


def interpolate_array(values, old_idx, new_idx, mode="linear"):
    from scipy import interpolate as scipy_interp
    T = len(old_idx)
    T_new = len(new_idx)
    if mode == "nearest":
        nearest_idx = np.clip(np.round(new_idx).astype(int), 0, T - 1)
        return values[nearest_idx]
    result = np.zeros((T_new,) + values.shape[1:], dtype=values.dtype)
    for d in range(values.shape[1] if values.ndim > 1 else 1):
        col = values[:, d] if values.ndim > 1 else values
        f = scipy_interp.interp1d(old_idx, col, kind="linear", fill_value="extrapolate")
        if values.ndim > 1:
            result[:, d] = f(new_idx)
        else:
            result = f(new_idx)
    return result


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

    # Download dataset
    print("Downloading Libero dataset metadata...")
    meta = LeRobotDatasetMetadata("physical-intelligence/libero")
    input_root = meta.root
    fps = meta.fps
    print(f"Input root: {input_root}, FPS: {fps}")

    # Download full dataset
    print("Downloading full Libero dataset...")
    ds = LeRobotDataset("physical-intelligence/libero")
    print(f"Dataset loaded: {len(ds)} frames")

    # Read episode info
    episodes_path = input_root / "meta" / "episodes.jsonl"
    episodes = []
    with open(episodes_path) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Total episodes: {len(episodes)}")

    output_base = Path("/mnt/default_storage/qiming/datasets/libero_speed_varied")
    speed_factors = [0.5, 1.0, 2.0]
    motion_indices = [0, 1, 2, 3, 4, 5]
    gripper_indices = [6]

    # Process each speed factor
    for speed in speed_factors:
        speed_dir = output_base / f"speed_{speed:.1f}"
        print(f"\n=== Creating speed={speed} dataset at {speed_dir} ===")

        # Copy metadata
        speed_dir.mkdir(parents=True, exist_ok=True)
        src_meta = input_root / "meta"
        dst_meta = speed_dir / "meta"
        if dst_meta.exists():
            shutil.rmtree(dst_meta)
        shutil.copytree(src_meta, dst_meta)

        episode_lengths = {}
        data_dir = input_root / "data"

        for i, ep_info in enumerate(episodes):
            ep_idx = ep_info["episode_index"]
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  Processing episode {i+1}/{len(episodes)} (idx={ep_idx})")

            # Find parquet file
            parquet_file = None
            for chunk_dir in sorted(data_dir.glob("chunk-*")):
                ep_file = chunk_dir / f"episode_{ep_idx:06d}.parquet"
                if ep_file.exists():
                    parquet_file = ep_file
                    break

            if parquet_file is None:
                print(f"  WARNING: Episode {ep_idx} parquet not found, skipping")
                continue

            df = pq.read_table(parquet_file).to_pandas()
            T = len(df)

            if speed == 1.0:
                # Just copy the file
                out_chunk = speed_dir / "data" / parquet_file.parent.name
                out_chunk.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parquet_file, out_chunk / parquet_file.name)
                episode_lengths[ep_idx] = T
                continue

            # Resample
            T_new = compute_new_length(T, speed, min_length=16)
            old_idx = np.arange(T, dtype=np.float64)
            new_idx = np.linspace(0, T - 1, T_new)

            new_data = {
                "episode_index": np.full(T_new, ep_idx, dtype=np.int64),
                "frame_index": np.arange(T_new, dtype=np.int64),
                "timestamp": (np.arange(T_new, dtype=np.float64) / fps if fps else np.linspace(0, float(df["timestamp"].iloc[-1]), T_new)),
            }

            if "task_index" in df.columns:
                new_data["task_index"] = np.full(T_new, int(df["task_index"].iloc[0]), dtype=np.int64)

            # Resample each column
            for col in df.columns:
                if col in ("episode_index", "frame_index", "timestamp", "task_index"):
                    continue
                vals = df[col].tolist()
                if isinstance(vals[0], (list, np.ndarray)):
                    arr = np.array(vals, dtype=np.float32)
                    if "action" in col.lower():
                        if arr.shape[1] == 7:
                            motion = interpolate_array(arr[:, motion_indices], old_idx, new_idx, "linear")
                            gripper = interpolate_array(arr[:, gripper_indices], old_idx, new_idx, "nearest")
                            new_arr = np.zeros((T_new, 7), dtype=np.float32)
                            new_arr[:, motion_indices] = motion
                            new_arr[:, gripper_indices] = gripper
                        else:
                            new_arr = interpolate_array(arr, old_idx, new_idx, "linear").astype(np.float32)
                        new_data[col] = [new_arr[j].tolist() for j in range(T_new)]
                    elif "state" in col.lower():
                        new_arr = interpolate_array(arr, old_idx, new_idx, "linear").astype(np.float32)
                        new_data[col] = [new_arr[j].tolist() for j in range(T_new)]
                    else:
                        nearest_idx = np.clip(np.round(new_idx).astype(int), 0, T - 1)
                        new_data[col] = [vals[j] for j in nearest_idx]
                else:
                    arr = np.array(vals)
                    if arr.dtype.kind == "f":
                        new_data[col] = interpolate_array(arr, old_idx, new_idx, "linear")
                    else:
                        nearest_idx = np.clip(np.round(new_idx).astype(int), 0, T - 1)
                        new_data[col] = arr[nearest_idx]

            out_chunk = speed_dir / "data" / "chunk-000"
            out_chunk.mkdir(parents=True, exist_ok=True)
            table = pa.table(new_data)
            pq.write_table(table, out_chunk / f"episode_{ep_idx:06d}.parquet")
            episode_lengths[ep_idx] = T_new

        # Update episode metadata
        updated_lines = []
        with open(speed_dir / "meta" / "episodes.jsonl") as f:
            for line in f:
                ep = json.loads(line)
                if ep["episode_index"] in episode_lengths:
                    ep["length"] = episode_lengths[ep["episode_index"]]
                updated_lines.append(json.dumps(ep))
        with open(speed_dir / "meta" / "episodes.jsonl", "w") as f:
            for line in updated_lines:
                f.write(line + "\n")

        info_path = speed_dir / "meta" / "info.json"
        with open(info_path) as f:
            info = json.load(f)
        info["total_frames"] = sum(episode_lengths.values())
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"  Done: {len(episode_lengths)} episodes, {sum(episode_lengths.values())} total frames")

    # === Create mixed dataset ===
    print("\n=== Creating mixed training dataset ===")
    rng = np.random.RandomState(42)
    mixed_dir = output_base / "mixed"

    src_meta = output_base / "speed_1.0" / "meta"
    dst_meta = mixed_dir / "meta"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(src_meta, dst_meta)

    episode_lengths = {}
    speed_assignments = {}

    for i, ep_info in enumerate(episodes):
        ep_idx = ep_info["episode_index"]
        speed = float(rng.choice(speed_factors))
        speed_assignments[ep_idx] = speed

        src_speed_dir = output_base / f"speed_{speed:.1f}"
        src_parquet = None
        for chunk_dir in sorted((src_speed_dir / "data").glob("chunk-*")):
            ep_file = chunk_dir / f"episode_{ep_idx:06d}.parquet"
            if ep_file.exists():
                src_parquet = ep_file
                break

        if src_parquet is None:
            print(f"  WARNING: Missing episode {ep_idx} at speed {speed}")
            continue

        out_chunk = mixed_dir / "data" / "chunk-000"
        out_chunk.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_parquet, out_chunk / f"episode_{ep_idx:06d}.parquet")

        df = pq.read_table(src_parquet).to_pandas()
        episode_lengths[ep_idx] = len(df)

    # Update mixed metadata
    updated_lines = []
    with open(mixed_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] in episode_lengths:
                ep["length"] = episode_lengths[ep["episode_index"]]
            updated_lines.append(json.dumps(ep))
    with open(mixed_dir / "meta" / "episodes.jsonl", "w") as f:
        for line in updated_lines:
            f.write(line + "\n")

    info_path = mixed_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_frames"] = sum(episode_lengths.values())
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Save assignments
    with open(output_base / "speed_assignments.json", "w") as f:
        json.dump({
            "seed": 42, "speed_choices": speed_factors,
            "assignments": {str(k): v for k, v in speed_assignments.items()},
            "distribution": {str(s): sum(1 for v in speed_assignments.values() if v == s) for s in speed_factors},
        }, f, indent=2)

    dist = {s: sum(1 for v in speed_assignments.values() if v == s) for s in speed_factors}
    print(f"Mixed dataset: {len(episode_lengths)} episodes, {sum(episode_lengths.values())} frames")
    print(f"Speed distribution: {dist}")
    print("Done!")


if __name__ == "__main__":
    main()
