#!/usr/bin/env python3
"""Remove extra episodes from a target directory using a reference directory.

Default behavior is dry-run. Use --apply to actually delete files/directories.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


EPISODE_PATTERN = re.compile(r"episode_(\d+)")
DEFAULT_REFERENCE_DIR = Path(
    "/home/t-qimhuang/disk/datasets/danze_data/paired_106/phantom_real_02_25_rgb"
)
DEFAULT_TARGET_DIR = Path(
    "/home/t-qimhuang/disk/datasets/danze_data/paired_106/rendered_videos_and_actions_02_25_fixed_hdf5"
)


def parse_episode_id(path: Path) -> int | None:
    match = EPISODE_PATTERN.search(path.name)
    if match is None:
        return None
    return int(match.group(1))


def collect_episode_ids(root: Path) -> set[int]:
    ids: set[int] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        episode_id = parse_episode_id(path)
        if episode_id is not None:
            ids.add(episode_id)
    return ids


def collect_target_items_to_remove(target_dir: Path, keep_ids: set[int]) -> list[Path]:
    candidates: list[Path] = []
    for path in target_dir.iterdir():
        episode_id = parse_episode_id(path)
        if episode_id is None:
            continue
        if episode_id not in keep_ids:
            candidates.append(path)
    return sorted(candidates)


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find episode IDs in reference dir and remove extra episode_* items in "
            "target dir that are not in reference."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory to read valid episode IDs from.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory to remove extra episode files/directories from.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/directories. Without this flag, only print plan.",
    )
    args = parser.parse_args()

    reference_dir = args.reference_dir.resolve()
    target_dir = args.target_dir.resolve()

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference dir does not exist: {reference_dir}")
    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir does not exist: {target_dir}")

    keep_ids = collect_episode_ids(reference_dir)
    print(f"Reference episodes found: {len(keep_ids)}")

    to_remove = collect_target_items_to_remove(target_dir, keep_ids)
    print(f"Target items marked for removal: {len(to_remove)}")

    for path in to_remove:
        print(f"[REMOVE] {path}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to delete.")
        return

    for path in to_remove:
        delete_path(path)

    print(f"Deleted {len(to_remove)} items.")


if __name__ == "__main__":
    main()
