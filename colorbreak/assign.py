"""Assign a mix level to every cell.

1. Each cell gets standard-normal values ``z = rho*G + sqrt(1-rho^2)*eps`` where G is a
   smooth multi-octave Gaussian field (large-scale mottling) and eps is per-cell blue noise
   (high-pass filtered white noise on the cell adjacency graph).
2. A discrete distribution ``p`` over the levels (points of a simplex) is derived from a
   Beta (2 colours) or Dirichlet (3 colours) prior fitted so that the *discretised*
   distribution has the requested spread; an exponential tilt makes the mean exact.
3. Cells are cut at area-weighted quantiles of ``p`` along one (2 colours) or two
   (3 colours, hierarchical) z fields, so the realised area fractions equal ``p`` up to one
   cell and tone varies coherently in space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import optimize, sparse
from scipy.stats import beta as beta_dist

from .field import MultiOctaveField, standardise

LAMBDA_CAP = 50.0


# ----------------------------------------------------------------------------- values
def cell_adjacency(
    cell_of: np.ndarray, tri_pairs: np.ndarray, n_cells: int, lengths: Optional[np.ndarray] = None
) -> sparse.csr_matrix:
    """Symmetric adjacency between cells sharing triangle edges (0/1, or summed edge length)."""
    a, b = cell_of[tri_pairs[:, 0]], cell_of[tri_pairs[:, 1]]
    keep = a != b
    a, b = a[keep], b[keep]
    w = np.ones(len(a)) if lengths is None else lengths[keep]
    m = sparse.coo_matrix((w, (a, b)), shape=(n_cells, n_cells)).tocsr()
    m = m + m.T
    if lengths is None:
        m.data[:] = 1.0
    return m


def blue_noise(
    n: int, adj: sparse.csr_matrix, rng: np.random.Generator, kappa: float = 0.7, sweeps: int = 1
) -> np.ndarray:
    """White noise made 'blue' by subtracting a fraction of the neighbour mean."""
    eps = rng.standard_normal(n)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    for _ in range(sweeps):
        eps = eps - kappa * (adj @ eps) / deg
        eps = standardise(eps)
    return eps


def cell_values(
    seeds: np.ndarray,
    cell_areas: np.ndarray,
    adj: sparse.csr_matrix,
    rng: np.random.Generator,
    coherence: float = 0.5,
    mottle_scale: float = 10.0,
) -> np.ndarray:
    rho = float(np.sqrt(max(0.0, min(1.0, coherence))))
    g = standardise(MultiOctaveField(mottle_scale, rng)(seeds), cell_areas)
    eps = blue_noise(len(seeds), adj, rng)
    z = rho * g + np.sqrt(1.0 - rho * rho) * eps
    return standardise(z, cell_areas)


# ------------------------------------------------------------------------ distribution
@dataclass
class Distribution:
    p: np.ndarray
    mean: np.ndarray  # (k,)
    std: float  # spread: sqrt(sum_i Var_i / 2)
    params: dict
    warnings: List[str]


def spread_of(p: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:
    m = p @ R
    var = (p[:, None] * (R - m) ** 2).sum()
    return m, float(np.sqrt(max(var, 0.0) / 2.0))


# --- two colours -----------------------------------------------------------------------
def level_bins(r: np.ndarray) -> np.ndarray:
    """Bin edges in ratio space: midpoints between consecutive levels, closed at 0 and 1."""
    mids = 0.5 * (r[1:] + r[:-1])
    return np.concatenate([[0.0], mids, [1.0]])


def discretise_beta(a: float, b: float, edges: np.ndarray) -> np.ndarray:
    q = np.clip(np.diff(beta_dist.cdf(edges, a, b)), 0.0, None)
    s = q.sum()
    return q / s if s > 0 else np.full(len(q), 1.0 / len(q))


def tilt_to_mean(q: np.ndarray, r: np.ndarray, mu: float) -> Tuple[np.ndarray, float]:
    """Exponential tilting p_j ∝ q_j exp(lambda r_j) with sum p r = mu (minimal-KL mean fix)."""
    pos = q > 0
    if not pos.any():
        raise ValueError("empty distribution")
    rmin, rmax = r[pos].min(), r[pos].max()
    if not (rmin - 1e-12 <= mu <= rmax + 1e-12):
        raise ValueError(f"target mean {mu:.4f} is outside the range of available levels [{rmin:.4f}, {rmax:.4f}]")
    if abs(rmax - rmin) < 1e-12:
        return q, 0.0

    def mean_at(lam: float) -> float:
        w = q * np.exp(lam * (r - mu))
        return float((w * r).sum() / w.sum())

    lo, hi = -1.0, 1.0
    while mean_at(lo) > mu and lo > -1e6:
        lo *= 2
    while mean_at(hi) < mu and hi < 1e6:
        hi *= 2
    if mean_at(lo) > mu or mean_at(hi) < mu:
        p = np.zeros_like(q)
        p[np.argmin(np.abs(r - mu))] = 1.0
        return p, 0.0
    lam = optimize.brentq(lambda l: mean_at(l) - mu, lo, hi, xtol=1e-12, maxiter=500)
    w = q * np.exp(lam * (r - mu))
    return w / w.sum(), lam


def fit_distribution_1d(r: np.ndarray, mu: float, sigma: float) -> Tuple[np.ndarray, dict, List[str]]:
    """Discrete distribution over scalar levels ``r`` (sorted) with exact mean and std ≈ sigma."""
    warnings: List[str] = []
    if len(r) == 1:
        return np.ones(1), {}, ["only one level available"]
    order = np.argsort(r)
    rs = r[order]
    edges = level_bins(rs)
    sigma = max(float(sigma), 1e-4)
    var_max = mu * (1.0 - mu)
    s2 = min(sigma * sigma, 0.95 * var_max) if var_max > 0 else sigma * sigma
    nu = var_max / s2 - 1.0 if s2 > 0 else 10.0
    a0, b0 = max(mu * nu, 1e-3), max((1.0 - mu) * nu, 1e-3)

    def resid(log_ab: np.ndarray) -> np.ndarray:
        q = discretise_beta(np.exp(log_ab[0]), np.exp(log_ab[1]), edges)
        m = float((q * rs).sum())
        s = float(np.sqrt((q * (rs - m) ** 2).sum()))
        return np.array([(m - mu) / max(sigma, 1e-3), (s - sigma) / max(sigma, 1e-3)])

    sol = optimize.least_squares(resid, np.log([a0, b0]), bounds=(np.log(1e-3), np.log(1e4)), xtol=1e-12, ftol=1e-12)
    a, b = np.exp(sol.x)
    q = discretise_beta(a, b, edges)
    ps, lam = tilt_to_mean(q, rs, mu)
    p = np.empty_like(ps)
    p[order] = ps
    return p, {"alpha": float(a), "beta": float(b), "tilt": float(lam)}, warnings


# --- three colours ---------------------------------------------------------------------
def simplex_grid(k: int, n: int) -> np.ndarray:
    """Interior barycentric grid points (i/n, ...) with every i >= 1 (margin 1/n from the edges)."""
    pts = []
    if k == 2:
        for i in range(1, n):
            pts.append((i / n, (n - i) / n))
    else:
        for i in range(1, n - 1):
            for j in range(1, n - i):
                l = n - i - j
                if l >= 1:
                    pts.append((i / n, j / n, l / n))
    return np.array(pts, dtype=np.float64)


def dirichlet_discretise(alpha: np.ndarray, R: np.ndarray, grid: np.ndarray, nearest: np.ndarray) -> np.ndarray:
    """Mass per level: Dirichlet(alpha) density on the grid, each point given to its nearest level."""
    logd = ((alpha - 1.0)[None, :] * np.log(grid)).sum(axis=1)
    logd -= logd.max()
    d = np.exp(logd)
    q = np.bincount(nearest, weights=d, minlength=len(R))
    q = np.maximum(q, 1e-6 * q.max() if q.max() > 0 else 1.0)
    return q / q.sum()


def feasible_interior(R: np.ndarray, mu: np.ndarray, mask: np.ndarray) -> bool:
    """Is mu a strict convex combination of the levels in ``mask``?  (small LP)"""
    pts = R[mask]
    n = len(pts)
    if n == 0:
        return False
    if n == 1:
        return bool(np.abs(pts[0] - mu).max() < 1e-9)
    # maximise t s.t. w_j >= t, sum w = 1, sum w r_j = mu   (variables: w_1..w_n, t)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_eq = np.zeros((R.shape[1] + 1, n + 1))
    A_eq[:-1, :n] = pts.T
    A_eq[-1, :n] = 1.0
    b_eq = np.concatenate([mu, [1.0]])
    A_ub = np.zeros((n, n + 1))
    A_ub[:, :n] = -np.eye(n)
    A_ub[:, -1] = 1.0
    res = optimize.linprog(
        c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1)] * n + [(None, None)], method="highs"
    )
    return bool(res.success and -res.fun > 1e-9)


def tilt_to_mean_vec(q: np.ndarray, R: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """p_j ∝ q_j exp(lambda . r_j) with p @ R = mu.  Damped Newton on the first k-1 coordinates
    (the last is redundant because rows of R sum to 1).  Returns (p, lambda, converged)."""
    k = R.shape[1]
    X = R[:, : k - 1] - mu[None, : k - 1]
    lam = np.zeros(k - 1)
    logq = np.log(np.maximum(q, 1e-300))
    converged = False
    for _ in range(200):
        logw = logq + X @ lam
        logw -= logw.max()
        p = np.exp(logw)
        p /= p.sum()
        g = p @ X  # mean error
        if np.abs(g).max() < 1e-12:
            converged = True
            break
        cov = (p[:, None] * X).T @ X
        cov += 1e-12 * np.eye(k - 1)
        step = np.linalg.solve(cov, -g)
        # damping: limit the step so |lam| stays sane
        scale = min(1.0, 5.0 / max(np.abs(step).max(), 1e-12))
        lam = lam + scale * step
        if np.abs(lam).max() > LAMBDA_CAP:
            lam = np.clip(lam, -LAMBDA_CAP, LAMBDA_CAP)
            break
    logw = logq + X @ lam
    logw -= logw.max()
    p = np.exp(logw)
    p /= p.sum()
    converged = converged or bool(np.abs(p @ X).max() < 1e-9)
    return p, lam, converged


def fit_distribution(R: np.ndarray, mu: np.ndarray, sigma: float) -> Distribution:
    """Discrete distribution over levels ``R`` (L, k) with mean ``mu`` (exact) and spread ≈ sigma.

    Spread is sqrt(sum_i Var(X_i) / 2), which equals the std of the first component for k = 2.
    """
    R = np.asarray(R, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    L, k = R.shape
    warnings: List[str] = []
    sigma = max(float(sigma), 1e-4)

    # Levels that use filaments absent from the target cannot carry mass (mean has zeros there).
    active = mu > 1e-9
    usable = ~(R[:, ~active] > 1e-9).any(axis=1) if (~active).any() else np.ones(L, dtype=bool)
    if usable.sum() == 0:
        raise ValueError("no level is compatible with the target ratio")
    p = np.zeros(L)
    params: dict = {}

    if active.sum() == 1:
        j = np.nonzero(usable)[0]
        best = j[np.argmin(np.abs(R[j] - mu).sum(axis=1))]
        p[best] = 1.0
        warnings.append("target is a pure colour; nothing to spread")
    elif active.sum() == 2:
        comp = np.nonzero(active)[0][0]
        idx = np.nonzero(usable)[0]
        pr, params, w = fit_distribution_1d(R[idx, comp], float(mu[comp]), sigma)
        p[idx] = pr
        warnings += w
    else:
        idx = np.nonzero(usable)[0]
        Ru = R[idx]
        if not feasible_interior(Ru, mu, np.ones(len(idx), dtype=bool)):
            raise ValueError("target ratio is not inside the convex hull of the available levels")
        grid = simplex_grid(k, 240)
        d2 = ((grid[:, None, :] - Ru[None, :, :]) ** 2).sum(axis=2)
        nearest = d2.argmin(axis=1)

        def tilted(kappa: float) -> Tuple[np.ndarray, bool]:
            q = dirichlet_discretise(kappa * mu, Ru, grid, nearest)
            pp, _, ok = tilt_to_mean_vec(q, Ru, mu)
            return pp, ok

        def g(logk: float) -> float:
            pp, _ = tilted(float(np.exp(logk)))
            return spread_of(pp, Ru)[1] - sigma

        lo, hi = np.log(0.1), np.log(1e4)
        glo, ghi = g(lo), g(hi)
        if glo <= 0:  # even the flattest prior is too narrow for the requested spread
            logk = lo
            warnings.append(f"requested spread {sigma:.3f} exceeds what these levels allow; using the widest")
        elif ghi >= 0:
            logk = hi
            warnings.append(f"requested spread {sigma:.3f} is below what these levels allow; using the narrowest")
        else:
            logk = optimize.brentq(g, lo, hi, xtol=1e-4)
        kappa = float(np.exp(logk))
        pu, ok = tilted(kappa)
        if not ok:
            warnings.append("mean could not be matched exactly by tilting (target close to the hull boundary)")
        p[idx] = pu
        params = {"kappa": kappa, "alpha": (kappa * mu).tolist()}

    m, s = spread_of(p, R)
    if abs(s - sigma) > 0.25 * sigma + 1e-3:
        warnings.append(f"requested spread {sigma:.3f} not achievable with these levels; realised {s:.3f}")
    pure = (R > 1 - 1e-9).any(axis=1)
    for j in np.nonzero(pure & (p > 0.25))[0]:
        warnings.append(f"{100 * p[j]:.0f}% of the area goes to a pure colour (large spread for this mean)")
    if (p > 0.02).sum() < 3:
        warnings.append(
            "fewer than three levels carry more than 2% of the area; consider a larger spread or finer levels"
        )
    return Distribution(p, m, s, params, warnings)


# -------------------------------------------------------------------------- assignment
def assign_levels(z: np.ndarray, cell_areas: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Level index per cell: cut the z-sorted cells at the area-weighted quantiles of p."""
    order = np.argsort(z, kind="stable")
    cum = np.cumsum(cell_areas[order])
    cum /= cum[-1]
    mid = cum - 0.5 * cell_areas[order] / cell_areas.sum()
    edges = np.cumsum(p)
    edges[-1] = 1.0 + 1e-9
    lvl_sorted = np.clip(np.searchsorted(edges, mid, side="right"), 0, len(p) - 1)
    out = np.empty(len(z), dtype=np.int64)
    out[order] = lvl_sorted
    return out


def principal_axes(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Weighted PCA axes of the level cloud (rows, descending variance)."""
    m = p @ R
    X = R - m
    cov = (p[:, None] * X).T @ X
    w, v = np.linalg.eigh(cov)
    return v[:, ::-1].T


def assign_levels_nd(
    z1: np.ndarray, z2: Optional[np.ndarray], cell_areas: np.ndarray, p: np.ndarray, R: np.ndarray, n_bands: int = 4
) -> np.ndarray:
    """Hierarchical quantile cut.  Levels sorted along principal axis 1 are grouped into bands
    (cells cut by z1 at the band masses); inside each band levels are sorted along axis 2 and
    the band's cells are cut by z2.  Exact area fractions, coherence in both colour directions.
    With z2 = None (two colours) this is the plain 1-D cut along the ratio axis."""
    active = np.nonzero(p > 0)[0]
    axes = principal_axes(R, p)
    c1 = R[active] @ axes[0]
    order1 = active[np.argsort(c1, kind="stable")]
    if z2 is None or R.shape[1] < 3 or len(order1) < 4:
        return order1[assign_levels(z1, cell_areas, p[order1])]
    # bands of adjacent levels with roughly equal mass
    cum = np.cumsum(p[order1])
    nb = min(n_bands, len(order1))
    band_of_level = np.minimum((cum - 0.5 * p[order1]) * nb, nb - 1).astype(int)
    band_mass = np.bincount(band_of_level, weights=p[order1], minlength=nb)
    band_mass = band_mass[band_mass > 0]
    bands = [order1[band_of_level == b] for b in range(nb) if (band_of_level == b).any()]
    band_cell = assign_levels(z1, cell_areas, band_mass / band_mass.sum())
    out = np.empty(len(z1), dtype=np.int64)
    for b, lv in enumerate(bands):
        cells = np.nonzero(band_cell == b)[0]
        if len(cells) == 0:
            continue
        c2 = R[lv] @ axes[1]
        lv2 = lv[np.argsort(c2, kind="stable")]
        pb = p[lv2] / p[lv2].sum()
        out[cells] = lv2[assign_levels(z1[cells] * 0 + z2[cells], cell_areas[cells], pb)]
    return out
