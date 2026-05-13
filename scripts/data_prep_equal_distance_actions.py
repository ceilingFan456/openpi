#!/usr/bin/env python3
"""Create equal-distance position-action LIBERO datasets.

This script rewrites a LeRobot LIBERO dataset so the model still predicts the
standard 7D LIBERO action vector, but the first three action dimensions are
derived from end-effector position deltas sampled at fixed arc-length spacing.

Example input:
    input_root=/mnt/default_storage/qiming/datasets/libero_3task
    ds=0.002
    pos_action_scale=0.0125

Example output:
    output_root=/mnt/default_storage/qiming/datasets/libero_3task_equal_distance_ds002
    state[t, :3] follows the original end-effector path at roughly 2 mm spacing
    actions[t, :3] = (state[t + 1, :3] - state[t, :3]) / pos_action_scale
    actions[t, 3:6] are arc-length-scaled from the source orientation commands
    actions[t, 6] is copied from the nearest source gripper command

The output keeps the original image columns by nearest-neighbor selection and
preserves the original Arrow schema metadata where possible, so OpenPI's normal
LeRobot loader can consume it with the existing pi05 LIBERO configs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


LOGGER = logging.getLogger("equal_distance_actions")
EPSILON = 1e-9


def matrix_from_arrow(table: pa.Table, column_name: str) -> np.ndarray:
    """Return a 2D float matrix from a fixed-size-list or list Arrow column."""
    column = table[column_name].combine_chunks()
    column_type = column.type
    if pa.types.is_fixed_size_list(column_type):
        values = column.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(column), column_type.list_size).astype(np.float32)
    return np.asarray(column.to_pylist(), dtype=np.float32)


def matrix_to_arrow(matrix: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    """Convert a 2D matrix back to an Arrow list column matching arrow_type."""
    if pa.types.is_fixed_size_list(arrow_type):
        value_type = arrow_type.value_type
        flat = matrix.astype(np.float32).reshape(-1)
        values = pa.array(flat, type=value_type)
        return pa.FixedSizeListArray.from_arrays(values, arrow_type.list_size)
    return pa.array(matrix.tolist(), type=arrow_type)


def set_column(table: pa.Table, column_name: str, values: pa.Array) -> pa.Table:
    """Replace an existing column by name."""
    column_index = table.schema.names.index(column_name)
    return table.set_column(column_index, column_name, values)


def numeric_array_for_column(table: pa.Table, column_name: str, values: np.ndarray) -> pa.Array:
    """Build an Arrow numeric array using the original column's Arrow type."""
    arrow_type = table.schema.field(column_name).type
    return pa.array(values, type=arrow_type)


def cumulative_arc_lengths(positions: np.ndarray) -> np.ndarray:
    """Compute cumulative path length for a sequence of 3D positions."""
    if len(positions) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(positions) == 1:
        return np.zeros(1, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(positions.astype(np.float64), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def resampled_arc_grid(total_arc: float, ds: float, min_length: int) -> np.ndarray:
    """Create an arc-length grid with spacing ds and an explicit final endpoint."""
    if total_arc <= EPSILON:
        return np.zeros(max(2, min_length), dtype=np.float64)

    target_arc = np.arange(0.0, total_arc, ds, dtype=np.float64)
    if len(target_arc) == 0 or not np.isclose(target_arc[-1], total_arc):
        target_arc = np.concatenate([target_arc, [total_arc]])
    if len(target_arc) < min_length:
        target_arc = np.linspace(0.0, total_arc, min_length, dtype=np.float64)
    return target_arc


def unique_arc_source(source_arc: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop duplicate arc positions so numpy interpolation remains well-defined."""
    unique_arc, unique_indices = np.unique(source_arc, return_index=True)
    unique_values = values[unique_indices]
    if len(unique_arc) == 1:
        unique_arc = np.array([unique_arc[0], unique_arc[0] + 1.0], dtype=np.float64)
        unique_values = np.concatenate([unique_values, unique_values], axis=0)
    return unique_arc, unique_values


def interpolate_by_arc(values: np.ndarray, source_arc: np.ndarray, target_arc: np.ndarray) -> np.ndarray:
    """Linearly interpolate a 1D or 2D array from source_arc to target_arc."""
    array = np.asarray(values)
    squeeze = array.ndim == 1
    if squeeze:
        array = array[:, None]

    unique_arc, unique_values = unique_arc_source(source_arc, array)
    output = np.zeros((len(target_arc), unique_values.shape[1]), dtype=np.float32)
    for column_index in range(unique_values.shape[1]):
        output[:, column_index] = np.interp(target_arc, unique_arc, unique_values[:, column_index]).astype(np.float32)

    if squeeze:
        return output[:, 0]
    return output


def nearest_rows_by_arc(source_arc: np.ndarray, target_arc: np.ndarray) -> np.ndarray:
    """Map target arc positions to nearest source row indices."""
    right_indices = np.searchsorted(source_arc, target_arc, side="left")
    right_indices = np.clip(right_indices, 0, len(source_arc) - 1)
    left_indices = np.clip(right_indices - 1, 0, len(source_arc) - 1)
    choose_left = np.abs(target_arc - source_arc[left_indices]) <= np.abs(source_arc[right_indices] - target_arc)
    return np.where(choose_left, left_indices, right_indices).astype(np.int64)


def resample_matrix_by_index(
    values: np.ndarray,
    sample_positions: np.ndarray,
    *,
    nearest_indices: list[int] | None = None,
) -> np.ndarray:
    """Resample a matrix by fractional row index with optional nearest columns."""
    old_positions = np.arange(len(values), dtype=np.float64)
    output = np.zeros((len(sample_positions), values.shape[1]), dtype=np.float32)
    nearest_index_set = set(nearest_indices or [])

    for column_index in range(values.shape[1]):
        if column_index in nearest_index_set:
            row_indices = np.clip(np.round(sample_positions).astype(np.int64), 0, len(values) - 1)
            output[:, column_index] = values[row_indices, column_index]
        else:
            output[:, column_index] = np.interp(sample_positions, old_positions, values[:, column_index]).astype(np.float32)
    return output


def make_speed_resampled_source(
    table: pa.Table,
    speed_factor: float,
    min_length: int,
) -> tuple[pa.Table, np.ndarray, np.ndarray]:
    """Optionally resample a source episode by global time speed before arc resampling."""
    source_state = matrix_from_arrow(table, "state")
    source_actions = matrix_from_arrow(table, "actions")
    if speed_factor == 1.0:
        return table, source_state, source_actions

    original_length = table.num_rows
    speed_length = max(min_length, int(round(original_length / speed_factor)))
    sample_positions = np.linspace(0, original_length - 1, speed_length, dtype=np.float64)
    nearest_time_indices = np.clip(np.round(sample_positions).astype(np.int64), 0, original_length - 1)
    source_table = table.take(pa.array(nearest_time_indices))
    speed_state = resample_matrix_by_index(source_state, sample_positions)
    speed_actions = resample_matrix_by_index(source_actions, sample_positions, nearest_indices=[6])
    return source_table, speed_state, speed_actions


def containing_segment_indices(source_arc: np.ndarray, target_arc: np.ndarray) -> np.ndarray:
    """Map target arc positions to the source segment that contains them."""
    if len(source_arc) <= 1:
        return np.zeros(len(target_arc), dtype=np.int64)
    indices = np.searchsorted(source_arc, target_arc, side="right") - 1
    return np.clip(indices, 0, len(source_arc) - 2).astype(np.int64)


def build_equal_distance_actions(
    source_actions: np.ndarray,
    resampled_state: np.ndarray,
    source_arc: np.ndarray,
    target_arc: np.ndarray,
    pos_action_scale: float,
    orientation_mode: str,
    orientation_action_scale: float,
) -> np.ndarray:
    """Build 7D actions for a fixed-arc-distance state trajectory."""
    action_dim = source_actions.shape[1]
    output_actions = np.zeros((len(target_arc), action_dim), dtype=np.float32)

    position_deltas = np.zeros((len(target_arc), 3), dtype=np.float32)
    if len(target_arc) > 1:
        position_deltas[:-1] = np.diff(resampled_state[:, :3], axis=0)
        position_deltas[-1] = position_deltas[-2]
    output_actions[:, :3] = position_deltas / float(pos_action_scale)

    segment_indices = containing_segment_indices(source_arc, target_arc)
    nearest_indices = nearest_rows_by_arc(source_arc, target_arc)

    if action_dim >= 6:
        if orientation_mode == "scaled-original":
            source_segment_lengths = np.diff(source_arc)
            if len(source_segment_lengths) == 0:
                source_segment_lengths = np.ones(1, dtype=np.float64)
            target_segment_lengths = np.linalg.norm(position_deltas.astype(np.float64), axis=1)
            source_lengths = np.maximum(source_segment_lengths[segment_indices], EPSILON)
            scale = (target_segment_lengths / source_lengths).astype(np.float32)[:, None]
            output_actions[:, 3:6] = source_actions[segment_indices, 3:6] * scale
        elif orientation_mode == "nearest-original":
            output_actions[:, 3:6] = source_actions[nearest_indices, 3:6]
        elif orientation_mode == "state-delta":
            orientation_deltas = np.zeros((len(target_arc), 3), dtype=np.float32)
            if len(target_arc) > 1:
                orientation_deltas[:-1] = np.diff(resampled_state[:, 3:6], axis=0)
                orientation_deltas[-1] = orientation_deltas[-2]
            output_actions[:, 3:6] = orientation_deltas / float(orientation_action_scale)
        else:
            raise ValueError(f"Unknown orientation mode: {orientation_mode}")

    if action_dim > 6:
        output_actions[:, 6:] = source_actions[nearest_indices, 6:]

    return output_actions


def rewrite_episode_table(
    table: pa.Table,
    *,
    ds: float,
    fps: int,
    global_start_index: int,
    min_length: int,
    pos_action_scale: float,
    orientation_mode: str,
    orientation_action_scale: float,
    speed_factor: float = 1.0,
) -> tuple[pa.Table, dict[str, float]]:
    """Rewrite one episode table and return the rewritten table plus stats."""
    source_table, source_state, source_actions = make_speed_resampled_source(table, speed_factor, min_length)
    source_arc = cumulative_arc_lengths(source_state[:, :3])
    target_arc = resampled_arc_grid(float(source_arc[-1]), ds, min_length)
    nearest_indices = nearest_rows_by_arc(source_arc, target_arc)

    rewritten = source_table.take(pa.array(nearest_indices))
    new_length = rewritten.num_rows

    resampled_state = interpolate_by_arc(source_state, source_arc, target_arc)
    resampled_state[:, :3] = interpolate_by_arc(source_state[:, :3], source_arc, target_arc)

    resampled_actions = build_equal_distance_actions(
        source_actions,
        resampled_state,
        source_arc,
        target_arc,
        pos_action_scale,
        orientation_mode,
        orientation_action_scale,
    )

    episode_index = int(table["episode_index"][0].as_py())
    task_index = int(table["task_index"][0].as_py()) if "task_index" in table.schema.names else None

    rewritten = set_column(rewritten, "episode_index", numeric_array_for_column(rewritten, "episode_index", np.full(new_length, episode_index)))
    rewritten = set_column(rewritten, "frame_index", numeric_array_for_column(rewritten, "frame_index", np.arange(new_length)))
    rewritten = set_column(rewritten, "timestamp", numeric_array_for_column(rewritten, "timestamp", np.arange(new_length) / float(fps)))
    if task_index is not None:
        rewritten = set_column(rewritten, "task_index", numeric_array_for_column(rewritten, "task_index", np.full(new_length, task_index)))
    if "index" in rewritten.schema.names:
        rewritten = set_column(rewritten, "index", numeric_array_for_column(rewritten, "index", global_start_index + np.arange(new_length)))

    rewritten = set_column(rewritten, "state", matrix_to_arrow(resampled_state, rewritten.schema.field("state").type))
    rewritten = set_column(rewritten, "actions", matrix_to_arrow(resampled_actions, rewritten.schema.field("actions").type))

    stats = {
        "original_length": float(table.num_rows),
        "speed_resampled_length": float(source_table.num_rows),
        "new_length": float(new_length),
        "speed_factor": float(speed_factor),
        "total_arc": float(source_arc[-1]),
        "mean_step_distance": float(np.mean(np.linalg.norm(np.diff(resampled_state[:, :3], axis=0), axis=1))) if new_length > 1 else 0.0,
    }
    return rewritten.replace_schema_metadata(table.schema.metadata), stats


def get_episode_indices(dataset_root: Path) -> list[int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    indices: list[int] = []
    with episodes_path.open(encoding="utf-8") as episodes_file:
        for line in episodes_file:
            if line.strip():
                indices.append(int(json.loads(line)["episode_index"]))
    return sorted(indices)


def copy_and_update_metadata(input_root: Path, output_root: Path, episode_lengths: dict[int, int], summary: dict) -> None:
    """Copy metadata and update frame counts for the rewritten dataset."""
    output_root.mkdir(parents=True, exist_ok=True)
    destination_meta = output_root / "meta"
    if destination_meta.exists():
        shutil.rmtree(destination_meta)
    shutil.copytree(input_root / "meta", destination_meta)

    info_path = destination_meta / "info.json"
    with info_path.open(encoding="utf-8") as info_file:
        info = json.load(info_file)
    info["total_episodes"] = len(episode_lengths)
    info["total_frames"] = int(sum(episode_lengths.values()))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episode_lengths)}"}
    with info_path.open("w", encoding="utf-8") as info_file:
        json.dump(info, info_file, indent=2)

    source_episode_rows = []
    with (input_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as episodes_file:
        for line in episodes_file:
            if line.strip():
                row = json.loads(line)
                if int(row["episode_index"]) in episode_lengths:
                    row["length"] = int(episode_lengths[int(row["episode_index"])])
                    source_episode_rows.append(row)

    with (destination_meta / "episodes.jsonl").open("w", encoding="utf-8") as episodes_file:
        for row in source_episode_rows:
            episodes_file.write(json.dumps(row) + "\n")

    readme_path = input_root / "README.md"
    if readme_path.exists():
        shutil.copy2(readme_path, output_root / "README.md")

    with (output_root / "equal_distance_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)


def create_equal_distance_dataset(
    input_root: Path,
    output_root: Path,
    *,
    ds: float,
    pos_action_scale: float,
    orientation_mode: str,
    orientation_action_scale: float,
    min_length: int,
    max_episodes: int | None,
    episode_speed_choices: list[float] | None,
    speed_seed: int,
    overwrite: bool,
) -> dict:
    """Create one equal-distance action dataset."""
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    info = json.loads((input_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info.get("fps", 10))
    episode_indices = get_episode_indices(input_root)
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    speed_assignments: dict[int, float] = {}
    if episode_speed_choices:
        rng = np.random.RandomState(speed_seed)
        speed_assignments = {episode_index: float(rng.choice(episode_speed_choices)) for episode_index in episode_indices}
    else:
        speed_assignments = {episode_index: 1.0 for episode_index in episode_indices}

    LOGGER.info("Creating %s from %s (%d episodes)", output_root, input_root, len(episode_indices))
    LOGGER.info("Equal-distance settings: ds=%.4f pos_action_scale=%.4f orientation_mode=%s", ds, pos_action_scale, orientation_mode)

    episode_lengths: dict[int, int] = {}
    episode_stats: dict[str, dict[str, float]] = {}
    global_start_index = 0

    for ordinal, episode_index in enumerate(episode_indices, start=1):
        if ordinal == 1 or ordinal % 25 == 0:
            LOGGER.info("  Episode %d/%d", ordinal, len(episode_indices))

        input_file = input_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(input_file)
        rewritten, stats = rewrite_episode_table(
            table,
            ds=ds,
            fps=fps,
            global_start_index=global_start_index,
            min_length=min_length,
            pos_action_scale=pos_action_scale,
            orientation_mode=orientation_mode,
            orientation_action_scale=orientation_action_scale,
            speed_factor=speed_assignments[episode_index],
        )

        output_file = output_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(rewritten, output_file)

        episode_lengths[episode_index] = rewritten.num_rows
        episode_stats[str(episode_index)] = stats
        global_start_index += rewritten.num_rows

    mean_step_distance = float(np.mean([stats["mean_step_distance"] for stats in episode_stats.values()])) if episode_stats else 0.0
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "ds": ds,
        "pos_action_scale": pos_action_scale,
        "orientation_mode": orientation_mode,
        "orientation_action_scale": orientation_action_scale,
        "episode_speed_choices": episode_speed_choices,
        "speed_seed": speed_seed,
        "speed_distribution": {
            str(speed): sum(1 for assigned_speed in speed_assignments.values() if assigned_speed == speed)
            for speed in sorted(set(speed_assignments.values()))
        },
        "num_episodes": len(episode_lengths),
        "total_frames": int(sum(episode_lengths.values())),
        "mean_resampled_step_distance": mean_step_distance,
        "episode_stats": episode_stats,
    }
    copy_and_update_metadata(input_root, output_root, episode_lengths, summary)
    LOGGER.info("Done. Wrote %d frames to %s", summary["total_frames"], output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create equal-distance position-action LIBERO datasets")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ds", type=float, default=0.002)
    parser.add_argument("--pos-action-scale", type=float, default=0.0125)
    parser.add_argument(
        "--orientation-mode",
        choices=["scaled-original", "nearest-original", "state-delta"],
        default="scaled-original",
    )
    parser.add_argument("--orientation-action-scale", type=float, default=0.25)
    parser.add_argument("--min-length", type=int, default=16)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode-speed-choices", type=float, nargs="+", default=None)
    parser.add_argument("--speed-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    create_equal_distance_dataset(
        args.input_root,
        args.output_root,
        ds=args.ds,
        pos_action_scale=args.pos_action_scale,
        orientation_mode=args.orientation_mode,
        orientation_action_scale=args.orientation_action_scale,
        min_length=args.min_length,
        max_episodes=args.max_episodes,
        episode_speed_choices=args.episode_speed_choices,
        speed_seed=args.speed_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()