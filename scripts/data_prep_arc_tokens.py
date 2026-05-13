#!/usr/bin/env python3
"""Re-encode each LIBERO LeRobot episode as a sequence of arc-token frames.

This script rewrites a LIBERO LeRobot dataset so that the action stream is
a sequence of greedy circular-arc tokens instead of dense per-step 7-D
position+orientation+gripper commands. The model interface stays compatible
with the existing pi0.5 LIBERO config (action_dim=7 padded to 32 by
PadStatesAndActions, image/state observation per frame), but the *meaning*
of each row changes:

    actions[t] = [is_eos, is_arc, kappa_signed, delta_s, n_x, n_y, n_z]

Per chunk (default L=20 arc-length segments of ds=5mm = 100mm of EE travel),
the arc fitter produces up to ``K_max=16`` primitives. Each chunk is emitted
as ``K_max`` contiguous frames in the new dataset, all sharing the chunk's
start image / state observation, with the ``j``-th frame holding the ``j``-th
arc token. Padding frames at the tail of a chunk hold an EOS token
(``is_eos=1``).

Example::

    python scripts/data_prep_arc_tokens.py \\
        --input-root /mnt/default_storage/qiming/datasets/libero_3task \\
        --output-root /mnt/default_storage/qiming/datasets/libero_3task_arc_tokens \\
        --ds 0.005 --alpha 0.02 --gripper-width 0.08 --K-max 16

The output keeps the original Arrow schema and metadata where possible so
OpenPI's normal LeRobot loader can ingest it with a pi05 LIBERO config
(``pi05_libero_3task_arc_tokens``) that points at the rewritten root.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Vendored arc-fitting / token-encoding utils (sibling file in scripts/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from _arc_token_utils import (  # noqa: E402
    compute_arc_length,
    encode_chunk_tokens,
    resample_by_arclength,
    TOKEN_DIM,
)


LOGGER = logging.getLogger("data_prep_arc_tokens")
EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Arrow helpers (mirror data_prep_equal_distance_actions.py)
# ---------------------------------------------------------------------------


def matrix_from_arrow(table: pa.Table, column_name: str) -> np.ndarray:
    column = table[column_name].combine_chunks()
    column_type = column.type
    if pa.types.is_fixed_size_list(column_type):
        values = column.values.to_numpy(zero_copy_only=False)
        return values.reshape(len(column), column_type.list_size).astype(np.float32)
    return np.asarray(column.to_pylist(), dtype=np.float32)


def matrix_to_arrow(matrix: np.ndarray, arrow_type: pa.DataType) -> pa.Array:
    if pa.types.is_fixed_size_list(arrow_type):
        value_type = arrow_type.value_type
        flat = matrix.astype(np.float32).reshape(-1)
        values = pa.array(flat, type=value_type)
        return pa.FixedSizeListArray.from_arrays(values, arrow_type.list_size)
    return pa.array(matrix.tolist(), type=arrow_type)


def set_column(table: pa.Table, column_name: str, values: pa.Array) -> pa.Table:
    column_index = table.schema.names.index(column_name)
    return table.set_column(column_index, column_name, values)


def numeric_array_for_column(table: pa.Table, column_name: str, values: np.ndarray) -> pa.Array:
    arrow_type = table.schema.field(column_name).type
    return pa.array(values, type=arrow_type)


# ---------------------------------------------------------------------------
# Chunk → token frames
# ---------------------------------------------------------------------------


def split_resampled_into_chunks(resampled: np.ndarray, L: int) -> list[tuple[int, int]]:
    """Return [(start_idx, end_idx_inclusive), ...] over ``resampled``.

    Chunks share boundary points (chunk i ends at the same index chunk i+1
    starts) and each chunk covers exactly L arc-length segments.
    Tail residue (length < L) becomes one final shorter chunk so we don't
    drop EE motion.
    """
    N = len(resampled)
    if N <= 1:
        return []
    chunk_size = L + 1  # L segments → L+1 vertices
    spans: list[tuple[int, int]] = []
    start = 0
    while start + chunk_size - 1 < N:
        end = start + chunk_size - 1
        spans.append((start, end))
        start = end  # share boundary point
    if start < N - 1:
        spans.append((start, N - 1))
    return spans


def map_resampled_to_source_indices(source_arc: np.ndarray, target_arc: np.ndarray) -> np.ndarray:
    """For each resampled point, return the nearest source row index."""
    right = np.searchsorted(source_arc, target_arc, side="left")
    right = np.clip(right, 0, len(source_arc) - 1)
    left = np.clip(right - 1, 0, len(source_arc) - 1)
    pick_left = np.abs(target_arc - source_arc[left]) <= np.abs(source_arc[right] - target_arc)
    return np.where(pick_left, left, right).astype(np.int64)


def rewrite_episode_to_arc_tokens(
    table: pa.Table,
    *,
    ds: float,
    fps: int,
    epsilon: float,
    K_max: int,
    L: int,
    max_radius: float,
    global_start_index: int,
) -> tuple[pa.Table, dict[str, float]]:
    """Rewrite one episode table as a sequence of K_max-sized token frames.

    Each chunk produces exactly K_max contiguous frames in the output. All K_max
    frames in a chunk share the same image/state observation (taken from the
    source frame nearest to the chunk's start arc position).
    """
    source_state = matrix_from_arrow(table, "state")
    source_actions = matrix_from_arrow(table, "actions")
    action_dim = source_actions.shape[1]

    source_positions = source_state[:, :3].astype(np.float64)
    source_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(source_positions, axis=0), axis=1))])
    total_arc = float(source_arc[-1]) if len(source_arc) else 0.0

    if total_arc <= 1e-6:
        # Degenerate episode (e.g. EE didn't move). Emit one EOS-only chunk.
        out_len = K_max
        rep_idx = np.zeros(out_len, dtype=np.int64)
        rewritten = table.take(pa.array(rep_idx))
        new_actions = np.zeros((out_len, action_dim), dtype=np.float32)
        new_actions[:, 0] = 1.0  # EOS in every slot
        new_state = source_state[rep_idx]
    else:
        # Resample at ds, split into chunks, fit arcs per chunk, lay out as K_max frames each.
        resampled, target_arc = resample_by_arclength(source_positions, ds)
        chunk_spans = split_resampled_into_chunks(resampled, L)
        if not chunk_spans:
            chunk_spans = [(0, len(resampled) - 1)]

        nearest_src = map_resampled_to_source_indices(source_arc, target_arc)

        rep_idx_list: list[int] = []
        token_blocks: list[np.ndarray] = []
        chunk_summary: list[dict] = []
        for (i, j) in chunk_spans:
            chunk_pts = resampled[i:j + 1]
            tokens, K = encode_chunk_tokens(chunk_pts, epsilon=epsilon, K_max=K_max, max_radius=max_radius)
            chunk_summary.append({"chunk_len": int(j - i + 1), "num_real_tokens": int(K)})

            chunk_start_src = int(nearest_src[i])
            rep_idx_list.extend([chunk_start_src] * K_max)

            # Pad tokens to (K_max, action_dim) by zero-padding feature dims after the 7 token features.
            block = np.zeros((K_max, action_dim), dtype=np.float32)
            block[:, :TOKEN_DIM] = tokens.astype(np.float32)
            token_blocks.append(block)

        rep_idx = np.asarray(rep_idx_list, dtype=np.int64)
        out_len = len(rep_idx)
        rewritten = table.take(pa.array(rep_idx))
        new_actions = np.concatenate(token_blocks, axis=0)
        new_state = source_state[rep_idx]

    # Update bookkeeping columns.
    episode_index = int(table["episode_index"][0].as_py())
    task_index = int(table["task_index"][0].as_py()) if "task_index" in table.schema.names else None

    rewritten = set_column(
        rewritten, "episode_index", numeric_array_for_column(rewritten, "episode_index", np.full(out_len, episode_index))
    )
    rewritten = set_column(
        rewritten, "frame_index", numeric_array_for_column(rewritten, "frame_index", np.arange(out_len))
    )
    rewritten = set_column(
        rewritten, "timestamp", numeric_array_for_column(rewritten, "timestamp", np.arange(out_len) / float(fps))
    )
    if task_index is not None:
        rewritten = set_column(
            rewritten, "task_index", numeric_array_for_column(rewritten, "task_index", np.full(out_len, task_index))
        )
    if "index" in rewritten.schema.names:
        rewritten = set_column(
            rewritten, "index",
            numeric_array_for_column(rewritten, "index", global_start_index + np.arange(out_len)),
        )

    rewritten = set_column(rewritten, "state", matrix_to_arrow(new_state, rewritten.schema.field("state").type))
    rewritten = set_column(rewritten, "actions", matrix_to_arrow(new_actions, rewritten.schema.field("actions").type))

    if total_arc > 1e-6:
        num_chunks = len(chunk_summary)
        avg_real = float(np.mean([c["num_real_tokens"] for c in chunk_summary])) if chunk_summary else 0.0
        max_real = int(np.max([c["num_real_tokens"] for c in chunk_summary])) if chunk_summary else 0
    else:
        num_chunks = 1
        avg_real = 0.0
        max_real = 0

    stats = {
        "original_length": float(table.num_rows),
        "new_length": float(out_len),
        "total_arc": total_arc,
        "num_chunks": float(num_chunks),
        "mean_real_tokens_per_chunk": avg_real,
        "max_real_tokens_per_chunk": float(max_real),
    }
    return rewritten.replace_schema_metadata(table.schema.metadata), stats


# ---------------------------------------------------------------------------
# Driver / metadata
# ---------------------------------------------------------------------------


def get_episode_indices(dataset_root: Path) -> list[int]:
    indices: list[int] = []
    with (dataset_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                indices.append(int(json.loads(line)["episode_index"]))
    return sorted(indices)


def copy_and_update_metadata(
    input_root: Path, output_root: Path, episode_lengths: dict[int, int], summary: dict
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    dst_meta = output_root / "meta"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(input_root / "meta", dst_meta)

    info_path = dst_meta / "info.json"
    with info_path.open(encoding="utf-8") as f:
        info = json.load(f)
    info["total_episodes"] = len(episode_lengths)
    info["total_frames"] = int(sum(episode_lengths.values()))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(episode_lengths)}"}
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # Update per-episode lengths in episodes.jsonl
    rows = []
    with (input_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if int(row["episode_index"]) in episode_lengths:
                    row["length"] = int(episode_lengths[int(row["episode_index"])])
                    rows.append(row)
    with (dst_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    readme = input_root / "README.md"
    if readme.exists():
        shutil.copy2(readme, output_root / "README.md")
    with (output_root / "arc_tokens_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def create_arc_tokens_dataset(
    input_root: Path,
    output_root: Path,
    *,
    ds: float,
    epsilon: float,
    K_max: int,
    L: int,
    max_radius: float,
    max_episodes: int | None,
    overwrite: bool,
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset root does not exist: {input_root}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {output_root} (pass --overwrite)")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    info = json.loads((input_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info.get("fps", 10))
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    episode_indices = get_episode_indices(input_root)
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    LOGGER.info("Rewriting %d episodes from %s -> %s", len(episode_indices), input_root, output_root)

    episode_lengths: dict[int, int] = {}
    per_episode_stats: list[dict] = []
    global_index = 0

    for k, episode_index in enumerate(episode_indices):
        rel_path = data_template.format(episode_chunk=0, episode_index=episode_index)
        in_path = input_root / rel_path
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(in_path)
        rewritten, stats = rewrite_episode_to_arc_tokens(
            table,
            ds=ds, fps=fps, epsilon=epsilon, K_max=K_max, L=L, max_radius=max_radius,
            global_start_index=global_index,
        )
        pq.write_table(rewritten, out_path, compression="snappy")
        episode_lengths[episode_index] = rewritten.num_rows
        global_index += rewritten.num_rows
        stats_row = {"episode_index": episode_index, **stats}
        per_episode_stats.append(stats_row)

        if (k + 1) % 25 == 0 or k + 1 == len(episode_indices):
            LOGGER.info(
                "  [%d/%d] ep %d: src=%d frames -> out=%d frames, chunks=%.0f, avg_tokens/chunk=%.2f",
                k + 1, len(episode_indices), episode_index,
                int(stats["original_length"]), int(stats["new_length"]),
                stats["num_chunks"], stats["mean_real_tokens_per_chunk"],
            )

    # Aggregate stats.
    total_orig = float(sum(s["original_length"] for s in per_episode_stats))
    total_new = float(sum(s["new_length"] for s in per_episode_stats))
    total_chunks = float(sum(s["num_chunks"] for s in per_episode_stats))
    avg_real = float(np.mean([s["mean_real_tokens_per_chunk"] for s in per_episode_stats]))
    max_real = float(np.max([s["max_real_tokens_per_chunk"] for s in per_episode_stats]))
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "ds": ds, "epsilon_m": epsilon, "K_max": K_max, "L": L, "max_radius": max_radius,
        "n_episodes": len(per_episode_stats),
        "total_source_frames": total_orig,
        "total_output_frames": total_new,
        "frame_inflation_ratio": total_new / max(total_orig, 1.0),
        "total_chunks": total_chunks,
        "avg_real_tokens_per_chunk": avg_real,
        "max_real_tokens_per_chunk_overall": max_real,
        "per_episode_stats": per_episode_stats,
    }
    copy_and_update_metadata(input_root, output_root, episode_lengths, summary)
    LOGGER.info(
        "[done] %d eps -> %d output frames over %d chunks (avg %.2f real tokens/chunk, max %.0f)",
        len(per_episode_stats), int(total_new), int(total_chunks), avg_real, max_real,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ds", type=float, default=0.005, help="arc-length resampling step (m)")
    parser.add_argument("--alpha", type=float, default=0.02, help="tolerance as fraction of gripper width")
    parser.add_argument("--gripper-width", type=float, default=0.08, help="m")
    parser.add_argument("--K-max", type=int, default=16, help="max primitives per chunk")
    parser.add_argument("--L", type=int, default=20, help="number of ds segments per chunk")
    parser.add_argument("--max-radius", type=float, default=1000.0,
                        help="m, fitted arcs above this become line tokens")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    epsilon = args.alpha * args.gripper_width
    LOGGER.info("epsilon = alpha*gripper_width = %.4f m (%.2f mm)", epsilon, epsilon * 1000.0)
    create_arc_tokens_dataset(
        args.input_root, args.output_root,
        ds=args.ds, epsilon=epsilon, K_max=args.K_max, L=args.L,
        max_radius=args.max_radius, max_episodes=args.max_episodes,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
