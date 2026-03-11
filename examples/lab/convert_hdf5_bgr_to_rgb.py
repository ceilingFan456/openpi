#!/usr/bin/env python3
"""Convert HDF5 datasets from BGR to RGB while preserving folder structure.

Example:
    python examples/lab/convert_hdf5_bgr_to_rgb.py \
        --src-root /path/to/source \
        --dst-root /path/to/destination
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively read HDF5 files under --src-root and write converted "
            "copies under --dst-root with the same subfolder structure."
        )
    )
    parser.add_argument("--src-root", type=Path, required=True, help="Source dataset root")
    parser.add_argument("--dst-root", type=Path, required=True, help="Destination dataset root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist",
    )
    return parser.parse_args()


def should_convert_bgr_to_rgb(array: np.ndarray) -> bool:
    # Heuristic for image-like arrays: at least 3 dims and channel-last with 3 channels.
    return array.ndim >= 3 and array.shape[-1] == 3 and np.issubdtype(array.dtype, np.integer)


def convert_dataset_data(data: np.ndarray) -> tuple[np.ndarray, bool]:
    if should_convert_bgr_to_rgb(data):
        return data[..., ::-1], True
    return data, False


def copy_attrs(src_obj: h5py.AttributeManager, dst_obj: h5py.AttributeManager) -> None:
    for key in src_obj.keys():
        dst_obj[key] = src_obj[key]


def copy_group(src_group: h5py.Group, dst_group: h5py.Group, stats: dict[str, int]) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)

    for name, item in src_group.items():
        if isinstance(item, h5py.Group):
            new_group = dst_group.create_group(name)
            copy_group(item, new_group, stats)
            continue

        if isinstance(item, h5py.Dataset):
            data = item[()]
            converted_data, converted = convert_dataset_data(data)
            if converted:
                stats["datasets_converted"] += 1

            create_kwargs = {
                "shape": converted_data.shape,
                "dtype": converted_data.dtype,
            }
            if item.chunks is not None:
                create_kwargs["chunks"] = item.chunks
            if item.compression is not None:
                create_kwargs["compression"] = item.compression
            if item.compression_opts is not None:
                create_kwargs["compression_opts"] = item.compression_opts
            if item.shuffle:
                create_kwargs["shuffle"] = item.shuffle
            if item.fletcher32:
                create_kwargs["fletcher32"] = item.fletcher32
            if item.scaleoffset is not None:
                create_kwargs["scaleoffset"] = item.scaleoffset
            if item.maxshape is not None:
                create_kwargs["maxshape"] = item.maxshape

            dst_dataset = dst_group.create_dataset(name, **create_kwargs)
            dst_dataset[...] = converted_data
            copy_attrs(item.attrs, dst_dataset.attrs)
            stats["datasets_total"] += 1


def convert_one_file(src_path: Path, dst_path: Path) -> dict[str, int]:
    stats = {"datasets_total": 0, "datasets_converted": 0}
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src_h5, h5py.File(dst_path, "w") as dst_h5:
        copy_group(src_h5, dst_h5, stats)

    return stats


def main() -> None:
    args = parse_args()

    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    h5_files = sorted(
        p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in {".hdf5", ".h5"}
    )
    if not h5_files:
        print(f"No HDF5 files found under: {src_root}")
        return

    print(f"Found {len(h5_files)} HDF5 files under {src_root}")

    grand_total = 0
    grand_converted = 0
    for src_path in h5_files:
        rel_path = src_path.relative_to(src_root)
        dst_path = dst_root / rel_path

        if dst_path.exists() and not args.overwrite:
            print(f"[SKIP] Destination exists (use --overwrite): {dst_path}")
            continue

        file_stats = convert_one_file(src_path, dst_path)
        grand_total += file_stats["datasets_total"]
        grand_converted += file_stats["datasets_converted"]

        print(
            f"[OK] {src_path} -> {dst_path} "
            f"(converted {file_stats['datasets_converted']}/{file_stats['datasets_total']} datasets)"
        )

    print(
        f"Done. Converted {grand_converted}/{grand_total} datasets across {len(h5_files)} files."
    )


if __name__ == "__main__":
    main()
