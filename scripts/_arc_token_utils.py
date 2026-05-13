"""Self-contained 3D arc fitting + token encoding utilities.

This is a vendored, dependency-free copy of the geometry tokenization code
from the ``arc_length_action`` repo (kept in-repo so AMLT jobs do not need
that sibling repo on PYTHONPATH).

Functions:
  * ``compute_arc_length`` / ``resample_by_arclength``
  * ``fit_circle_arc_3d`` / ``sample_arc_3d`` / ``make_line_segment``
  * ``greedy_arc_simplification``
  * ``primitive_to_token`` (7-D ``[is_eos, is_arc, kappa, delta_s, n_x, n_y, n_z]``)
  * ``pad_tokens_to_kmax``

If any change is made here, mirror it in ``src/geometry_tokenization`` of the
arc_length_action repo (and vice-versa) to keep the proxy and the real
training run in sync.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Arc-length resampling
# ---------------------------------------------------------------------------


def compute_arc_length(points: np.ndarray) -> Tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D, got shape {points.shape}")
    if len(points) == 0:
        return np.zeros((0,), dtype=np.float64), 0.0
    if len(points) == 1:
        return np.zeros((1,), dtype=np.float64), 0.0
    diffs = np.diff(points, axis=0)
    seg = np.linalg.norm(diffs, axis=-1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s, float(s[-1])


def interpolate_points_by_s(points: np.ndarray, s: np.ndarray, query_s: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    query_s = np.asarray(query_s, dtype=np.float64)
    if len(points) < 2:
        if len(points) == 1:
            return np.tile(points[0:1], (len(query_s), 1))
        raise ValueError("Need at least one point to interpolate.")
    q = np.clip(query_s, s[0], s[-1])
    idx = np.searchsorted(s, q, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 2)
    s0, s1 = s[idx], s[idx + 1]
    denom = s1 - s0
    safe = denom > 1e-12
    alpha = np.where(safe, (q - s0) / np.where(safe, denom, 1.0), 0.0)
    p0, p1 = points[idx], points[idx + 1]
    return (1.0 - alpha[:, None]) * p0 + alpha[:, None] * p1


def resample_by_arclength(points: np.ndarray, ds: float) -> Tuple[np.ndarray, np.ndarray]:
    if ds <= 0:
        raise ValueError(f"ds must be positive, got {ds}")
    s, total = compute_arc_length(points)
    if total <= 1e-12:
        return points[:1].copy().astype(np.float64), np.zeros((1,), dtype=np.float64)
    query_s = np.arange(0.0, total, ds, dtype=np.float64)
    if len(query_s) == 0 or query_s[-1] < total - 1e-12:
        query_s = np.append(query_s, total)
    return interpolate_points_by_s(points, s, query_s), query_s


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def _max_error(gt: np.ndarray, recon: np.ndarray) -> float:
    if gt.shape != recon.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {recon.shape}")
    return float(np.max(np.linalg.norm(gt - recon, axis=-1)))


# ---------------------------------------------------------------------------
# Circle / arc fitting
# ---------------------------------------------------------------------------


def _fit_circle_2d(points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
    x, y = points_2d[:, 0], points_2d[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=-1)
    b = -(x ** 2 + y ** 2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a_, b_, c_ = sol
    cx, cy = -a_ / 2.0, -b_ / 2.0
    r2 = cx ** 2 + cy ** 2 - c_
    return np.array([cx, cy], dtype=np.float64), float(np.sqrt(max(r2, 1e-16)))


def make_line_segment(p0: np.ndarray, p1: np.ndarray) -> Dict[str, object]:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    return {"type": "line", "p0": p0.copy(), "p1": p1.copy(), "length": float(np.linalg.norm(p1 - p0))}


def fit_circle_arc_3d(points: np.ndarray, max_radius: float = 1e3) -> Dict[str, object]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return make_line_segment(points[0], points[-1])
    centroid = points.mean(axis=0)
    X = points - centroid
    try:
        _, sv, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return make_line_segment(points[0], points[-1])
    if vh.shape[0] < 3 or sv[1] < 1e-9 * max(sv[0], 1e-12):
        return make_line_segment(points[0], points[-1])
    u, v, n = vh[0], vh[1], vh[2]
    px, py = X @ u, X @ v
    pts2 = np.stack([px, py], axis=-1)
    try:
        c2, r = _fit_circle_2d(pts2)
    except np.linalg.LinAlgError:
        return make_line_segment(points[0], points[-1])
    if not np.isfinite(r) or r > max_radius or r < 1e-9:
        return make_line_segment(points[0], points[-1])
    c3 = centroid + c2[0] * u + c2[1] * v
    rel = points - c3
    angles = np.unwrap(np.arctan2(rel @ v, rel @ u))
    return {
        "type": "arc", "center": c3, "radius": float(r), "normal": n,
        "basis_u": u, "basis_v": v,
        "theta0": float(angles[0]), "theta1": float(angles[-1]),
    }


def sample_arc_3d(arc: Dict[str, object], num_points: int) -> np.ndarray:
    th = np.linspace(arc["theta0"], arc["theta1"], num_points)
    return (
        arc["center"][None, :]
        + arc["radius"] * np.cos(th)[:, None] * arc["basis_u"][None, :]
        + arc["radius"] * np.sin(th)[:, None] * arc["basis_v"][None, :]
    )


def sample_primitive(primitive: Dict[str, object], num_points: int) -> np.ndarray:
    if primitive.get("type", "arc") == "line":
        a = np.linspace(0.0, 1.0, num_points)
        return (1.0 - a)[:, None] * primitive["p0"][None, :] + a[:, None] * primitive["p1"][None, :]
    return sample_arc_3d(primitive, num_points)


def _fit_and_error(points: np.ndarray, i: int, j: int, max_radius: float):
    cand = points[i:j + 1]
    if len(cand) < 2:
        return None, np.inf
    if len(cand) == 2:
        prim = make_line_segment(cand[0], cand[-1])
        return prim, _max_error(cand, sample_primitive(prim, len(cand)))
    try:
        prim = fit_circle_arc_3d(cand, max_radius=max_radius)
        err = _max_error(cand, sample_primitive(prim, len(cand)))
    except Exception:
        return None, np.inf
    if not np.isfinite(err):
        return None, np.inf
    return prim, err


def greedy_arc_simplification(
    points: np.ndarray, epsilon: float, min_points: int = 3, max_radius: float = 1e3,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int]]]:
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    primitives: List[Dict[str, object]] = []
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < N - 1:
        first_j = min(i + max(min_points, 2) - 1, N - 1)
        first_prim, first_err = _fit_and_error(points, i, first_j, max_radius)
        if first_prim is None or first_err > epsilon:
            prim = make_line_segment(points[i], points[i + 1])
            primitives.append(prim); segments.append((i, i + 1))
            i = i + 1
            continue
        best_j, best_prim = first_j, first_prim
        step = max(1, first_j - i)
        j = first_j
        while j < N - 1:
            next_j = min(j + step, N - 1)
            prim, err = _fit_and_error(points, i, next_j, max_radius)
            if prim is not None and err <= epsilon:
                best_j, best_prim = next_j, prim
                if next_j == N - 1:
                    break
                j = next_j
                step *= 2
            else:
                lo, hi = best_j, next_j
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    prim, err = _fit_and_error(points, i, mid, max_radius)
                    if prim is not None and err <= epsilon:
                        lo, best_j, best_prim = mid, mid, prim
                    else:
                        hi = mid
                break
        primitives.append(best_prim); segments.append((i, best_j))
        i = best_j
    return primitives, segments


# ---------------------------------------------------------------------------
# Token encoding
# ---------------------------------------------------------------------------

# Token layout: [is_eos, is_arc, kappa_signed, delta_s, n_x, n_y, n_z]
TOKEN_DIM = 7


def _signed_kappa(prim: Dict[str, object]) -> float:
    k = 1.0 / max(prim["radius"], 1e-9)
    dtheta = float(prim["theta1"] - prim["theta0"])
    return k if dtheta >= 0 else -k


def primitive_to_token(prim: Dict[str, object], sub_points: np.ndarray) -> np.ndarray:
    _, length = compute_arc_length(sub_points)
    if prim.get("type") == "line":
        return np.array([0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0], dtype=np.float64)
    n = np.asarray(prim["normal"], dtype=np.float64)
    nn = float(np.linalg.norm(n))
    if nn > 1e-12:
        n = n / nn
    return np.array(
        [0.0, 1.0, _signed_kappa(prim), float(length), float(n[0]), float(n[1]), float(n[2])],
        dtype=np.float64,
    )


def encode_chunk_tokens(
    chunk_points: np.ndarray, epsilon: float, K_max: int, max_radius: float = 1e3,
) -> Tuple[np.ndarray, int]:
    """Run greedy arc simplification and pack tokens into a [K_max, 7] block.

    Returns (tokens, num_real) where:
      - tokens[:num_real, :] are the real arc/line primitives
      - tokens[num_real, :] is an EOS token (is_eos=1) when num_real < K_max
      - tokens[num_real+1:, :] are zero-padded (also acts as EOS-after-EOS)
    """
    primitives, segments = greedy_arc_simplification(
        chunk_points, epsilon=epsilon, max_radius=max_radius,
    )
    tokens = np.zeros((K_max, TOKEN_DIM), dtype=np.float32)
    K = min(len(primitives), K_max)
    for j in range(K):
        prim = primitives[j]
        i0, i1 = segments[j]
        tokens[j] = primitive_to_token(prim, chunk_points[i0:i1 + 1]).astype(np.float32)
    if K < K_max:
        tokens[K, 0] = 1.0  # EOS
    return tokens, K
