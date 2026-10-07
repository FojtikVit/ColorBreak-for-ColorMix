"""Mesh adjacency, patch connected components, printability cleanup and exact mean repair.

A *patch* is a connected set of target triangles sharing the same level.  Patches that
are too small (area) or too thin (PCA width in the tangent plane) are merged into a
neighbouring patch.  Finally whole patches are moved to a nearby level until the
area-weighted mean ratio vector matches the target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------------------- adjacency
def triangle_adjacency(vertices: np.ndarray, triangles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pairs of triangles sharing an edge, and the length of that edge.

    Returns ``(pairs (E,2) int, lengths (E,) float)``.  Edges shared by more than two
    triangles (non-manifold) contribute consecutive pairs.
    """
    t = triangles
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    owner = np.tile(np.arange(len(t)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    i = np.nonzero(same)[0]
    pairs = np.stack([owner[i], owner[i + 1]], axis=1)
    lengths = np.linalg.norm(vertices[e[i, 0]] - vertices[e[i, 1]], axis=1)
    return pairs, lengths


def face_areas_centroids_normals(vertices: np.ndarray, triangles: np.ndarray):
    a = vertices[triangles[:, 0]]
    b = vertices[triangles[:, 1]]
    c = vertices[triangles[:, 2]]
    n = np.cross(b - a, c - a)
    nl = np.linalg.norm(n, axis=1)
    areas = 0.5 * nl
    normals = n / np.where(nl > 0, nl, 1.0)[:, None]
    return areas, (a + b + c) / 3.0, normals


# ------------------------------------------------------------------------ components
def components(labels: np.ndarray, pairs: np.ndarray, n: int) -> Tuple[int, np.ndarray]:
    """Connected components of the graph restricted to edges joining equal labels."""
    keep = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    p = pairs[keep]
    g = sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)


def pca_widths(comp_of: np.ndarray, n_comp: int, cent: np.ndarray, areas: np.ndarray) -> np.ndarray:
    """Tangent-plane width of every component: sqrt(12) * second singular value of the
    area-weighted centroid cloud.  Equals the width of a strip and ~0.87x the diameter of a
    disk, and is insensitive to the zigzag of triangle-edge boundaries and surface roughness."""
    w = areas / np.maximum(np.bincount(comp_of, weights=areas, minlength=n_comp)[comp_of], 1e-12)
    mean = np.stack([np.bincount(comp_of, weights=w * cent[:, i], minlength=n_comp) for i in range(3)], axis=1)
    d = cent - mean[comp_of]
    cov = np.zeros((n_comp, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cov[:, i, j] = np.bincount(comp_of, weights=w * d[:, i] * d[:, j], minlength=n_comp)
            cov[:, j, i] = cov[:, i, j]
    ev = np.linalg.eigvalsh(cov)  # ascending
    return np.sqrt(12.0 * np.maximum(ev[:, 1], 0.0))


@dataclass
class PatchStats:
    comp_of: np.ndarray  # (N,) component id per triangle
    n: int
    area: np.ndarray  # (C,)
    boundary: np.ndarray  # (C,) length of border to other components (+ external)
    level: np.ndarray  # (C,) level index
    shared: sparse.csr_matrix  # (C,C) shared boundary length between components
    width: np.ndarray  # (C,) estimated patch width in mm


def patch_stats(
    level_of_tri: np.ndarray,
    areas: np.ndarray,
    pairs: np.ndarray,
    lengths: np.ndarray,
    ext_boundary: Optional[np.ndarray] = None,
    cent: Optional[np.ndarray] = None,
) -> PatchStats:
    """``ext_boundary``: per-triangle length of edges shared with non-target triangles.
    ``cent``: triangle centroids; if given, widths are PCA based, else 2*area/boundary."""
    n_comp, comp_of = components(level_of_tri, pairs, len(level_of_tri))
    area = np.bincount(comp_of, weights=areas, minlength=n_comp)
    ca, cb = comp_of[pairs[:, 0]], comp_of[pairs[:, 1]]
    diff = ca != cb
    ca, cb, ln = ca[diff], cb[diff], lengths[diff]
    shared = sparse.coo_matrix((ln, (ca, cb)), shape=(n_comp, n_comp)).tocsr()
    shared = shared + shared.T
    boundary = np.asarray(shared.sum(axis=1)).ravel()
    if ext_boundary is not None:
        boundary = boundary + np.bincount(comp_of, weights=ext_boundary, minlength=n_comp)
    level = np.zeros(n_comp, dtype=np.int64)
    level[comp_of] = level_of_tri
    if cent is not None:
        width = pca_widths(comp_of, n_comp, cent, areas)
    else:
        width = np.where(boundary > 0, 2.0 * area / np.maximum(boundary, 1e-12), np.inf)
    return PatchStats(comp_of, n_comp, area, boundary, level, shared, width)


# ---------------------------------------------------------------------------- helpers
def _as_matrix(level_ratio: np.ndarray) -> np.ndarray:
    R = np.asarray(level_ratio, dtype=np.float64)
    if R.ndim == 1:
        R = np.stack([R, 1.0 - R], axis=1)
    return R


def mean_error(level_of_tri: np.ndarray, areas: np.ndarray, R: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (areas[:, None] * R[level_of_tri]).sum(axis=0) / areas.sum() - target


def err_norm(delta: np.ndarray) -> float:
    """Comparable to a 1-D ratio error: |delta| / sqrt(2) (equals |d| for delta = (d, -d))."""
    return float(np.linalg.norm(delta) / np.sqrt(2.0))


# ---------------------------------------------------------------------------- cleanup
def cleanup_small_patches(
    level_of_tri: np.ndarray,
    areas: np.ndarray,
    pairs: np.ndarray,
    lengths: np.ndarray,
    level_ratio: np.ndarray,
    target_mean,
    min_area: float,
    min_width: float,
    ext_boundary: Optional[np.ndarray] = None,
    cent: Optional[np.ndarray] = None,
    max_passes: int = 30,
    log=None,
) -> np.ndarray:
    """Merge patches with area < min_area or width < min_width into a neighbour.

    Among neighbours whose shared border is at least 70 % of the longest shared border,
    the one whose level best reduces the running mean error is chosen.
    """
    R = _as_matrix(level_ratio)
    target = np.asarray(target_mean, dtype=np.float64)
    if target.ndim == 0:
        target = np.array([float(target), 1.0 - float(target)])
    lvl = level_of_tri.copy()
    total = areas.sum()
    for it in range(max_passes):
        st = patch_stats(lvl, areas, pairs, lengths, ext_boundary, cent)
        bad = (st.area < min_area) | (st.width < min_width)
        isolated = bad & ~(np.asarray(st.shared.sum(axis=1)).ravel() > 0)
        bad &= ~isolated  # isolated patches (no target neighbours) cannot be merged
        n_bad = int(bad.sum())
        if log:
            log(f"  cleanup pass {it + 1}: {st.n} patches, {n_bad} too small/thin")
        if n_bad == 0:
            if log and isolated.any():
                log(
                    f"  note: {int(isolated.sum())} small patch(es) are isolated pieces of the original painting and were left as they are"
                )
            break
        delta = mean_error(lvl, areas, R, target)
        new_level = st.level.copy()
        shared = st.shared.tolil()
        for c in np.nonzero(bad)[0][np.argsort(st.area[bad])]:
            nbrs = np.array(shared.rows[c])
            if len(nbrs) == 0:
                continue
            w = np.array(shared.data[c])
            cand = nbrs[w >= 0.7 * w.max()]
            errs = [np.linalg.norm(delta + st.area[c] * (R[new_level[k]] - R[new_level[c]]) / total) for k in cand]
            k = cand[int(np.argmin(errs))]
            delta = delta + st.area[c] * (R[new_level[k]] - R[new_level[c]]) / total
            new_level[c] = new_level[k]
        lvl = new_level[st.comp_of]
    return lvl


def repair_mean(
    level_of_tri: np.ndarray,
    areas: np.ndarray,
    pairs: np.ndarray,
    lengths: np.ndarray,
    level_ratio: np.ndarray,
    target_mean,
    tol: float = 1e-4,
    n_nearest: int = 3,
    log=None,
) -> np.ndarray:
    """Move whole patches to a nearby level until the mean error norm is below ``tol``.

    Smallest patches first, candidate levels = the ``n_nearest`` levels in ratio space
    around the patch's current level; the move that reduces the error most is taken.
    Geometry does not change, so no new slivers can appear.
    """
    R = _as_matrix(level_ratio)
    target = np.asarray(target_mean, dtype=np.float64)
    if target.ndim == 0:
        target = np.array([float(target), 1.0 - float(target)])
    lvl = level_of_tri.copy()
    total = areas.sum()
    st = patch_stats(lvl, areas, pairs, lengths)
    comp_level = st.level.copy()
    delta = (st.area[:, None] * R[comp_level]).sum(axis=0) / total - target
    if log:
        log(f"  mean before repair: {np.round(delta + target, 5).tolist()} (target {np.round(target, 5).tolist()})")
    d2 = ((R[:, None, :] - R[None, :, :]) ** 2).sum(axis=2)
    by_dist = np.argsort(d2, axis=1)[:, 1:]
    changed = 0
    order = np.argsort(st.area)
    # several passes, widening the candidate set (3 nearest levels, then 6, then all)
    for width in (n_nearest, 2 * n_nearest, R.shape[0]):
        for _ in range(3):
            moved = 0
            for c in order:
                if err_norm(delta) < tol:
                    break
                cur = comp_level[c]
                best, best_err = None, np.linalg.norm(delta)
                for new in by_dist[cur, :width]:
                    d_new = delta + st.area[c] * (R[new] - R[cur]) / total
                    e = np.linalg.norm(d_new)
                    if e < best_err:
                        best, best_err = new, e
                if best is not None:
                    delta = delta + st.area[c] * (R[best] - R[cur]) / total
                    comp_level[c] = best
                    moved += 1
            changed += moved
            if err_norm(delta) < tol or moved == 0:
                break
        if err_norm(delta) < tol:
            break
    if log:
        log(
            f"  mean after repair:  {np.round(delta + target, 5).tolist()}; {changed} patches moved to a neighbouring level"
        )
        if err_norm(delta) >= tol:
            log(
                f"  note: tolerance {tol:g} not reached (error {err_norm(delta):.2g}); the smallest patch changes the mean by ~{st.area.min() * 0.05 / total:.1e}"
            )
    return comp_level[st.comp_of]
