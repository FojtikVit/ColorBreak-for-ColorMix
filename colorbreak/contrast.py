"""Neighbour contrast: make adjacent patches print *out of phase*.

Two patches whose layer cycles coincide (same level, or cycles where one divides the other)
have their filament changes at the same layers, so their stripes line up across the border.
The kernel ``K`` measures that alignment; simulated annealing swaps the levels of pairs of
cells (preserving area fractions, hence mean and spread) to reduce the aligned border length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

MAX_LAYERS = 120
SAME_LEVEL_BONUS = 0.1


def alignment(
    seq_a: Sequence[int], seq_b: Sequence[int], ratios_a: Dict[int, float], ratios_b: Dict[int, float]
) -> float:
    """Aligned fraction of layers printing the same filament, minus the uncorrelated expectation."""
    na, nb = len(seq_a), len(seq_b)
    L = na * nb // math.gcd(na, nb)
    if L > MAX_LAYERS:
        L = MAX_LAYERS
    same = sum(1 for i in range(L) if seq_a[i % na] == seq_b[i % nb]) / L
    base = sum(ratios_a.get(f, 0.0) * ratios_b.get(f, 0.0) for f in set(ratios_a) | set(ratios_b))
    return max(0.0, same - base)


def kernel_matrix(seqs: List[Sequence[int]], ratios: List[Dict[int, float]], ids: Sequence[int]) -> np.ndarray:
    """Symmetric (n, n) alignment kernel between n "paint states" given their layer sequences,
    filament ratio dicts and extruder ids (equal ids get a bonus so equal neighbours are avoided)."""
    n = len(seqs)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            v = alignment(seqs[i], seqs[j], ratios[i], ratios[j])
            if ids[i] == ids[j]:
                v += SAME_LEVEL_BONUS
            K[i, j] = K[j, i] = v
    return K


@dataclass
class ContrastResult:
    level_of_cell: np.ndarray
    f_before: float
    f_min: float
    f_after: float
    sweeps: int


def aligned_fraction(adj: sparse.csr_matrix, state: np.ndarray, K: np.ndarray) -> float:
    coo = sparse.triu(adj, k=1).tocoo()
    if coo.nnz == 0:
        return 0.0
    return float((coo.data * K[state[coo.row], state[coo.col]]).sum() / coo.data.sum())


def _anneal(
    state: np.ndarray,
    n_movable: int,
    area: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    K: np.ndarray,
    rng: np.random.Generator,
    max_sweeps: int,
    stop_at: Optional[float],
    total_w: float,
) -> Tuple[np.ndarray, float, int]:
    """Metropolis swaps of two movable cells with similar area.  Returns (state, f, sweeps)."""
    state = state.copy()
    n = n_movable
    if n < 2:
        return state, _energy(state, indptr, indices, data, K) / total_w, 0
    order_by_area = np.argsort(area[:n])
    rank = np.empty(n, dtype=np.int64)
    rank[order_by_area] = np.arange(n)
    window = max(2, n // 20)

    def local(i: int, li: int) -> float:
        nb = indices[indptr[i] : indptr[i + 1]]
        return float((data[indptr[i] : indptr[i + 1]] * K[li, state[nb]]).sum())

    E = _energy(state, indptr, indices, data, K)
    # temperature scale: typical positive edge cost
    coo_costs = []
    for i in range(n):
        nb = indices[indptr[i] : indptr[i + 1]]
        c = data[indptr[i] : indptr[i + 1]] * K[state[i], state[nb]]
        coo_costs.append(c[c > 0])
    costs = np.concatenate(coo_costs) if coo_costs else np.zeros(1)
    T0 = float(np.median(costs)) if costs.size else 1.0
    if not T0 > 0:
        T0 = float(data.mean()) * 0.1 if data.size else 1.0
    T_end = 0.1 * T0
    sweeps_done = 0
    best_E, best_state = E, state.copy()
    stall = 0
    for sweep in range(max_sweeps):
        T = T0 * (T_end / T0) ** (sweep / max(1, max_sweeps - 1))
        accepted = 0
        ii = rng.integers(0, n, size=n)
        jj_off = rng.integers(-window, window + 1, size=n)
        uu = rng.random(n)
        for t in range(n):
            i = int(ii[t])
            j = int(order_by_area[(rank[i] + int(jj_off[t])) % n])
            li, lj = int(state[i]), int(state[j])
            if i == j or li == lj:
                continue
            ai, aj = area[i], area[j]
            if ai > 1.2 * aj or aj > 1.2 * ai:
                continue
            before = local(i, li) + local(j, lj)
            state[i], state[j] = lj, li
            after = local(i, lj) + local(j, li)
            # if i and j are adjacent the shared edge is counted twice in both sums; it cancels in the
            # difference only if its value is unchanged, which holds since K is symmetric.
            dE = after - before
            if dE <= 0 or uu[t] < math.exp(-dE / T):
                E += dE
                accepted += 1
            else:
                state[i], state[j] = li, lj
        sweeps_done = sweep + 1
        if E < best_E - 1e-12:
            best_E, best_state = E, state.copy()
            stall = 0
        else:
            stall += 1
        if stop_at is not None and E / total_w <= stop_at:
            break
        if stall >= 15 and accepted == 0:
            break
    if stop_at is None:
        return best_state, best_E / total_w, sweeps_done
    return state, E / total_w, sweeps_done


def _energy(state: np.ndarray, indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, K: np.ndarray) -> float:
    e = 0.0
    n = len(indptr) - 1
    for i in range(n):
        nb = indices[indptr[i] : indptr[i + 1]]
        m = nb > i
        if m.any():
            e += float((data[indptr[i] : indptr[i + 1]][m] * K[state[i], state[nb[m]]]).sum())
    return e


def apply_contrast(
    level_of_cell: np.ndarray,
    cell_area: np.ndarray,
    adj: sparse.csr_matrix,
    K: np.ndarray,
    contrast: float,
    rng: np.random.Generator,
    fixed_state: Optional[np.ndarray] = None,
    max_sweeps: int = 120,
) -> ContrastResult:
    """Reduce the aligned border fraction by level swaps.

    ``adj``: (C+F, C+F) weighted adjacency; the first C nodes are movable cells, the remaining F
    nodes are fixed neighbours whose kernel rows are ``fixed_state`` (indices into K).
    ``contrast`` in [0, 1]: 1 keeps the fully annealed state, c stops when
    f = f_before - c (f_before - f_min) is reached, 0 does nothing.
    """
    n = len(level_of_cell)
    state = np.concatenate(
        [level_of_cell, fixed_state if fixed_state is not None else np.zeros(0, dtype=np.int64)]
    ).astype(np.int64)
    area = np.concatenate([cell_area, np.zeros(len(state) - n)])
    adj = adj.tocsr()
    coo = sparse.triu(adj, k=1).tocoo()
    total_w = float(coo.data.sum()) if coo.nnz else 1.0
    f0 = aligned_fraction(adj, state, K)
    if contrast <= 0 or n < 2 or coo.nnz == 0:
        return ContrastResult(level_of_cell.copy(), f0, f0, f0, 0)
    s_full, f_min, sw = _anneal(state, n, area, adj.indptr, adj.indices, adj.data, K, rng, max_sweeps, None, total_w)
    if contrast >= 1.0:
        return ContrastResult(s_full[:n], f0, f_min, f_min, sw)
    target = f0 - contrast * (f0 - f_min)
    s_part, f_part, sw2 = _anneal(
        state, n, area, adj.indptr, adj.indices, adj.data, K, rng, max_sweeps, target, total_w
    )
    return ContrastResult(s_part[:n], f0, f_min, f_part, sw2)
