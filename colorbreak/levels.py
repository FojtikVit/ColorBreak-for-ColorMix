"""Mix levels: which extruders (physical or virtual) are available as patch colours,
what ratio vector they *nominally* have, what they *effectively* print with, and how to
create missing ones.

A blend of k physical filaments ("components") is a point of the (k-1)-simplex.  Every level
of a run is expressed as a ratio vector over the same ordered filament tuple; pure colours
and 2-colour blends of a 3-colour source are points with zero entries.

Effective ratio: PrusaSlicer prints a blend with the shortest layer cycle ``n`` (2..64) for
which ``counts_i = max(1, round(r_i*n))`` reproduce the ratios within 0.03
(``quantise_component_counts`` in ``Biz/Algorithms/VirtualExtruder.cpp``).  E.g. 0.15/0.85
prints as 1 layer in 6 (0.1667).  The cycle is indexed by the global layer number, so every
region painted with the same virtual extruder is in phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BLEND_WEIGHT_RESOLUTION = 64
BLEND_QUANTISE_MAX_ERROR = 0.03
MAX_BLEND_COMPONENTS = 3
MAX_EXTRUDER_ID = 255


# ------------------------------------------------------------------ PrusaSlicer ports
def round_half_away(x: float) -> float:
    """C ``std::round`` semantics (half away from zero)."""
    return math.floor(x + 0.5) if x >= 0 else -math.floor(-x + 0.5)


def quantise_counts(ratios: Sequence[float]) -> List[int]:
    """Port of PrusaSlicer's quantise_component_counts (layer counts per component)."""
    total = sum(ratios)
    counts: List[int] = []
    for n in range(2, BLEND_WEIGHT_RESOLUTION + 1):
        counts = [max(1, int(round_half_away(r / total * n))) for r in ratios]
        s = sum(counts)
        err = max(abs(r / total - c / s) for r, c in zip(ratios, counts))
        if err <= BLEND_QUANTISE_MAX_ERROR:
            break
    g = 0
    for c in counts:
        g = math.gcd(g, c)
    if g > 1:
        counts = [c // g for c in counts]
    return counts


def canonical_counts_sequence(counts: Sequence[int]) -> List[int]:
    """Port of build_canonical_cycle: component index per layer of one cycle."""
    n = sum(counts)
    emitted = [0] * len(counts)
    seq: List[int] = []
    for slot in range(n):
        best, best_def = 0, -math.inf
        for i, c in enumerate(counts):
            deficit = (slot + 1) * c / n - emitted[i]
            if deficit > best_def:
                best, best_def = i, deficit
        emitted[best] += 1
        seq.append(best)
    return seq


def effective_ratio_vec(vec: Sequence[float]) -> Tuple[np.ndarray, int]:
    """(printed ratio vector, cycle length) for a nominal ratio vector (zeros allowed)."""
    v = np.asarray(vec, dtype=np.float64)
    active = np.nonzero(v > 1e-9)[0]
    out = np.zeros_like(v)
    if len(active) <= 1:
        if len(active) == 1:
            out[active[0]] = 1.0
        return out, 1
    counts = quantise_counts([float(v[i]) for i in active])
    s = sum(counts)
    for i, c in zip(active, counts):
        out[i] = c / s
    return out, s


def effective_ratio(r: float) -> Tuple[float, int]:
    """Two-colour convenience: (printed fraction of component A, cycle) for nominal r."""
    eff, cyc = effective_ratio_vec([r, 1.0 - r])
    return float(eff[0]), cyc


def layer_sequence(filaments: Sequence[int], vec: Sequence[float]) -> List[int]:
    """Physical extruder id printed in each layer of one cycle (starts at global layer 0)."""
    v = np.asarray(vec, dtype=np.float64)
    active = [i for i in range(len(v)) if v[i] > 1e-9]
    if len(active) == 0:
        return [int(filaments[0])]
    if len(active) == 1:
        return [int(filaments[active[0]])]
    counts = quantise_counts([float(v[i]) for i in active])
    return [int(filaments[active[c]]) for c in canonical_counts_sequence(counts)]


def canonical_cycle(ratio_a: float) -> List[int]:
    """Two-colour cycle as 1 (component A) / 0 (component B) per layer."""
    return [1 if f == 1 else 0 for f in layer_sequence((1, 2), [ratio_a, 1.0 - ratio_a])]


# ---------------------------------------------------------------------------- levels
@dataclass
class Level:
    """One usable patch colour, expressed over the filament tuple of the run."""

    extruder_id: int  # triangle state to paint with
    nominal: np.ndarray  # (k,) nominal fractions
    effective: np.ndarray  # (k,) printed fractions
    cycle: int  # layer cycle length (1 for pure colours)
    colour: str  # hex colour for previews
    is_new: bool = False  # created by colorbreak (must be added to the project)

    def ratio(self, mode: str) -> np.ndarray:
        return self.effective if mode == "effective" else self.nominal

    @property
    def n_colours(self) -> int:
        return int((self.nominal > 1e-9).sum())

    def label(self) -> str:
        return "/".join(f"{100 * x:g}" for x in self.nominal) + "%"


class IdAllocator:
    """Hands out unused virtual extruder ids (shared across a whole run)."""

    def __init__(self, used: Sequence[int], physical_count: int):
        self.next_id = max([physical_count] + [int(u) for u in used]) + 1

    def take(self) -> int:
        if self.next_id > MAX_EXTRUDER_ID:
            raise ValueError("no free virtual extruder id (max 255)")
        v = self.next_id
        self.next_id += 1
        return v


def blend_of(ve: dict) -> Optional[Dict[int, float]]:
    """{physical extruder: normalised ratio} for a 2- or 3-component fullspectrum blend."""
    if ve.get("kind", "fullspectrum") != "fullspectrum":
        return None
    comps = ve.get("components", [])
    if not 2 <= len(comps) <= MAX_BLEND_COMPONENTS:
        return None
    if any("position" in c for c in comps):
        return None
    d: Dict[int, float] = {}
    for c in comps:
        d[int(c["extruder"])] = d.get(int(c["extruder"]), 0.0) + float(c.get("ratio", 0.0))
    tot = sum(d.values())
    if tot <= 0 or len(d) < 2:
        return None
    return {e: r / tot for e, r in d.items()}


def vec_over(filaments: Sequence[int], blend: Dict[int, float]) -> Optional[np.ndarray]:
    """Ratio vector of ``blend`` over ``filaments``; None if it uses other filaments."""
    if not set(blend) <= set(filaments):
        return None
    return np.array([blend.get(f, 0.0) for f in filaments], dtype=np.float64)


# ---------------------------------------------------------------------------- colours
def _hex_to_rgb(h: str) -> np.ndarray:
    h = (h or "").lstrip("#")
    if len(h) < 6:
        return np.array([0.5, 0.5, 0.5])
    return np.array([int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _rgb_to_hex(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(round(float(np.clip(c, 0, 1)) * 255)):02x}" for c in rgb)


def fit_pure_colours(ves: List[dict], filaments: Sequence[int], filament_colours: Sequence[str]) -> np.ndarray:
    """(k, 3) RGB of the pure filaments, least-squares fitted from the existing virtual extruder
    colours of this filament set (``colour ≈ Σ r_i c_i``).  Falls back to the project's filament
    colours when the fit is underdetermined (fewer than k independent blends)."""
    rows, rhs = [], []
    for ve in ves:
        b = blend_of(ve)
        if b is None or not ve.get("color"):
            continue
        v = vec_over(filaments, b)
        if v is None:
            continue
        rows.append(v)
        rhs.append(_hex_to_rgb(ve["color"]))
    fallback = np.stack(
        [
            _hex_to_rgb(filament_colours[f - 1]) if f - 1 < len(filament_colours) else np.array([0.5, 0.5, 0.5])
            for f in filaments
        ]
    )
    if len(rows) >= len(filaments):
        A = np.stack(rows)
        if np.linalg.matrix_rank(A, tol=1e-6) == len(filaments):
            sol, *_ = np.linalg.lstsq(A, np.stack(rhs), rcond=None)
            return np.clip(sol, 0.0, 1.0)
    return fallback


def colour_for(vec: np.ndarray, pure_rgb: np.ndarray) -> str:
    return _rgb_to_hex(np.asarray(vec) @ pure_rgb)


# --------------------------------------------------------------------------- building
def _grid(k: int, n: int) -> List[Tuple[int, ...]]:
    if k == 1:
        return [(n,)]
    out = []
    for i in range(n + 1):
        for rest in _grid(k - 1, n - i):
            out.append((i,) + rest)
    return out


def parse_level_spec(spec: str, k: int) -> Optional[List[np.ndarray]]:
    """Wanted non-pure nominal vectors for ``spec``; None means 'auto' (existing only)."""
    if spec == "auto":
        return None
    if spec.startswith("step:"):
        n = int(round(1.0 / float(spec[5:])))
        pts = [np.array(g, dtype=np.float64) / n for g in _grid(k, n)]
        return [p for p in pts if (p > 1e-9).sum() >= 2]
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            v = np.array([float(x) for x in item.split("/")], dtype=np.float64)
            if len(v) != k:
                raise ValueError(f"level {item!r} has {len(v)} components, expected {k}")
        else:
            if k != 2:
                raise ValueError(f"level {item!r}: give {k} components separated by '/'")
            r = float(item)
            v = np.array([r, 1.0 - r])
        v = v / v.sum()
        if (v > 1e-9).sum() >= 2:
            out.append(v)
    return out


def build_levels(
    virtual_extruders: List[dict],
    filaments: Sequence[int],
    pure_rgb: np.ndarray,
    ids: IdAllocator,
    spec: str = "auto",
    include_pure: bool = True,
) -> List[Level]:
    """Assemble the level set over ``filaments`` (sorted tuple of physical extruder ids).

    ``spec``: ``auto`` = existing blends whose filaments are a subset of ``filaments``;
    ``step:X`` = full barycentric grid of step X (edges are 2-colour, interior 3-colour);
    explicit ``a/b/c,...`` (or ``r1,r2,...`` fractions of the first filament for k = 2).
    Missing blends get negative placeholder ids; call ``assign_ids`` for the ones actually used.
    """
    k = len(filaments)
    existing: List[Tuple[np.ndarray, dict]] = []
    for ve in virtual_extruders:
        b = blend_of(ve)
        if b is None:
            continue
        v = vec_over(filaments, b)
        if v is not None:
            existing.append((v, ve))

    wanted = parse_level_spec(spec, k)
    if wanted is None:
        wanted = [v for v, _ in existing]
    else:
        wanted = [v for v, _ in existing] + wanted  # existing blends are free, keep them

    levels: List[Level] = []
    if include_pure:
        for i, f in enumerate(filaments):
            v = np.zeros(k)
            v[i] = 1.0
            levels.append(Level(int(f), v, v.copy(), 1, colour_for(v, pure_rgb)))
    seen: List[np.ndarray] = []
    placeholder = -1  # real ids are taken from ``ids`` only for levels that end up used
    for v in wanted:
        if any(np.abs(v - s).max() < 1e-6 for s in seen):
            continue
        seen.append(v)
        eff, cyc = effective_ratio_vec(v)
        match = next((ve for ev, ve in existing if np.abs(ev - v).max() < 1e-6), None)
        if match is not None:
            levels.append(Level(int(match["id"]), v, eff, cyc, match.get("color") or colour_for(v, pure_rgb)))
        else:
            levels.append(Level(placeholder, v, eff, cyc, colour_for(v, pure_rgb), is_new=True))
            placeholder -= 1
    levels.sort(key=lambda l: tuple(-x for x in l.nominal))
    return levels


def assign_ids(levels: Sequence[Level], ids: IdAllocator) -> None:
    """Give real virtual extruder ids to new levels (call only for levels that are used)."""
    for l in levels:
        if l.is_new and l.extruder_id < 0:
            l.extruder_id = ids.take()


def dedupe_levels(levels: List[Level]) -> Tuple[List[Level], List[str]]:
    """Levels with the same printed ratio produce the same layer sequence and are in phase, so
    only one of them is kept: an existing extruder if possible, else the nominally closest."""
    groups: Dict[Tuple[int, ...], List[Level]] = {}
    for l in levels:
        groups.setdefault(tuple(int(round(x * 1e6)) for x in l.effective), []).append(l)
    kept: List[Level] = []
    notes: List[str] = []
    for g in groups.values():
        g.sort(key=lambda l: (l.is_new, float(np.abs(l.nominal - l.effective).sum())))
        kept.append(g[0])
        if len(g) > 1:
            notes.append(
                ", ".join(l.label() for l in g)
                + f" all print {g[0].label() if False else '/'.join(f'{100 * x:.1f}' for x in g[0].effective)}%, keeping {g[0].label()}"
            )
    kept.sort(key=lambda l: tuple(-x for x in l.nominal))
    return kept, notes


def new_virtual_extruder_entry(level: Level, filaments: Sequence[int]) -> dict:
    """JSON entry for a virtual extruder created by colorbreak."""
    comps = [{"extruder": int(f), "ratio": round(float(r), 6)} for f, r in zip(filaments, level.nominal) if r > 1e-9]
    return {"color": level.colour, "components": comps, "id": level.extruder_id, "kind": "fullspectrum"}


def phase_std(levels: Sequence[Level], share: Sequence[float], patch_layers: float, filaments: Sequence[int]) -> float:
    """Area-weighted RMS deviation (per sqrt(2), i.e. comparable to ``spread``) of the ratio a
    patch of ``patch_layers`` layers really receives from the level's printed ratio, over all
    phases of the layer cycle."""
    h = max(1, int(round(patch_layers)))
    var = 0.0
    for lv, w in zip(levels, share):
        if w <= 0 or lv.cycle <= 1:
            continue
        seq = layer_sequence(filaments, lv.nominal)
        n = len(seq)
        idx = {f: i for i, f in enumerate(filaments)}
        counts = np.zeros((n, len(filaments)))
        for ph in range(n):
            for i in range(h):
                counts[ph, idx[seq[(ph + i) % n]]] += 1.0 / h
        m = counts.mean(axis=0)
        var += w * float(((counts - m) ** 2).sum(axis=1).mean()) / 2.0
    return math.sqrt(var)
