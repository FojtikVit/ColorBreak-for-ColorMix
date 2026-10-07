"""Random irregular patches ("cells") on a triangle soup.

Every target triangle is assigned to one cell.  Cells are the cells of a *power
diagram* (weighted Voronoi diagram) of seed points that were Poisson-disk sampled on
the surface, computed in a *warped, z-stretched* metric:

* z-stretch  - the z axis is shrunk before measuring distances, so cells are taller
  than wide (more layers per patch, fewer horizontal borders);
* warp       - a smooth random vector field displaces space before the distance is
  measured, so cell borders wobble instead of being straight Voronoi edges;
* weights    - random additive weights give cells of varying size.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .field import GaussianField

HEX_CELL_AREA = np.sqrt(3) / 2  # area of a regular hexagon of width 1 (inradius 1/2)


@dataclass
class Cells:
    cell_of: np.ndarray  # (N,) cell index per target triangle
    seeds: np.ndarray  # (C, 3) seed points in real (world) coordinates
    spacing: float  # Poisson-disk spacing actually used (in the stretched metric)


def poisson_disk_thin(points: np.ndarray, spacing: float, rng: np.random.Generator) -> np.ndarray:
    """Greedy dart throwing: indices of an r-separated subset of ``points`` in random order."""
    order = rng.permutation(len(points))
    tree = cKDTree(points)
    blocked = np.zeros(len(points), dtype=bool)
    accepted = []
    for i in order:
        if blocked[i]:
            continue
        accepted.append(i)
        blocked[tree.query_ball_point(points[i], spacing, return_sorted=False)] = True
    return np.array(accepted, dtype=np.int64)


def make_cells(
    centroids: np.ndarray,
    areas: np.ndarray,
    patch_size: float,
    rng: np.random.Generator,
    z_stretch: float = 0.65,
    irregularity: float = 0.35,
    size_jitter: float = 0.5,
    k_neighbours: int = 12,
) -> Cells:
    d = float(patch_size)
    total_area = float(areas.sum())
    metric = np.array([1.0, 1.0, z_stretch])
    m = centroids * metric

    # Warp the metric space with three smooth Gaussian fields of length scale d.
    if irregularity > 0:
        warp = np.stack([GaussianField(d, rng)(m) for _ in range(3)], axis=1)
        m = m + irregularity * d * warp

    # Target number of cells: a cell is ~d wide and ~d/z_stretch tall in real space.
    cell_area_real = HEX_CELL_AREA * d * d / z_stretch
    n_target = max(4, int(round(total_area / cell_area_real)))

    # Candidate seeds: area-weighted sample of triangles.
    n_cand = min(len(centroids), max(20 * n_target, 2000))
    cand = rng.choice(len(centroids), size=n_cand, replace=True, p=areas / total_area)
    cand = np.unique(cand)

    # Poisson-disk thinning, adjusting the spacing so the cell count lands near n_target.
    spacing = 0.8 * d
    for _ in range(4):
        sel = poisson_disk_thin(m[cand], spacing, rng)
        ratio = len(sel) / n_target
        if 0.9 < ratio < 1.1:
            break
        spacing *= np.sqrt(ratio)
    seed_idx = cand[sel]
    seeds_m = m[seed_idx]

    # Power diagram: argmin |x - s|^2 - w_j over the k nearest seeds.
    weights = rng.uniform(0.0, 1.0, size=len(seeds_m)) * (size_jitter * d) ** 2
    tree = cKDTree(seeds_m)
    k = min(k_neighbours, len(seeds_m))
    dist, idx = tree.query(m, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    power = dist**2 - weights[idx]
    cell_of = idx[np.arange(len(m)), power.argmin(axis=1)]

    return Cells(cell_of=cell_of, seeds=centroids[seed_idx], spacing=spacing)


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    """Map arbitrary integer labels to 0..K-1 preserving first-appearance order of sorted labels."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv
