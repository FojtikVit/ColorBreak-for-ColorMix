"""Command line interface: ``colorbreak inspect`` and ``colorbreak break``."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from . import paintcodes as pc
from .assign import assign_levels_nd, cell_adjacency, cell_values, fit_distribution
from .cleanup import (
    cleanup_small_patches,
    components,
    face_areas_centroids_normals,
    patch_stats,
    repair_mean,
    triangle_adjacency,
)
from .contrast import apply_contrast, kernel_matrix
from .levels import (
    IdAllocator,
    assign_ids,
    blend_of,
    build_levels,
    dedupe_levels,
    effective_ratio_vec,
    fit_pure_colours,
    layer_sequence,
    new_virtual_extruder_entry,
    phase_std,
    vec_over,
)
from .patches import make_cells, relabel_consecutive
from .preview import colours_from_hex, render_views, write_ply
from .threemf import Project

PRESETS = {"subtle": 0.05, "medium": 0.08, "bold": 0.12}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def fmt_vec(v, digits=4) -> str:
    return "/".join(f"{float(x):.{digits}f}" for x in v)


# ------------------------------------------------------------------------------ inspect
def cmd_inspect(args: argparse.Namespace) -> int:
    prj = Project(args.input)
    print(f"{args.input}")
    print(f"  physical extruders: {prj.physical_extruder_count()}  colours: {', '.join(prj.extruder_colours())}")
    print(f"  layer height: {prj.layer_height()} mm")
    ves = prj.virtual_extruders()
    if ves:
        print("  virtual extruders:")
        for ve in ves:
            b = blend_of(ve)
            if b:
                fil = sorted(b)
                vec = np.array([b[f] for f in fil])
                eff, cyc = effective_ratio_vec(vec)
                nom = " / ".join(f"{100 * b[f]:4.1f}% ext{f}" for f in fil)
                print(
                    f"    id {ve['id']:3d}  {nom:42s} prints as {fmt_vec(eff, 3)} (cycle of {cyc} layers)   colour {ve.get('color', '-')}"
                )
            else:
                print(
                    f"    id {ve['id']:3d}  {ve.get('kind', 'fullspectrum')} with {len(ve.get('components', []))} components"
                )
    for vid, p in prj.paintings.items():
        mesh = prj.world_mesh(vid)
        areas, _, _ = face_areas_centroids_normals(mesh.vertices, mesh.triangles)
        print(
            f"  volume object {vid}: {len(mesh.triangles)} triangles, {areas.sum():.0f} mm^2, "
            f"{len(p.codes)} painted, painting version {p.version}"
        )
        state_area: Dict[int, float] = {}
        for t, code in p.codes.items():
            ls = pc.leaf_states(code)
            for s in ls:
                state_area[s] = state_area.get(s, 0.0) + areas[t] / len(ls)
        unp = areas.sum() - sum(state_area.values())
        if unp > 1e-6:
            state_area[0] = unp
        for s in sorted(state_area):
            name = "unpainted (volume default)" if s == 0 else f"extruder {s}"
            print(f"    {name:28s} {state_area[s]:9.1f} mm^2  ({100 * state_area[s] / areas.sum():5.1f}%)")
    return 0


# -------------------------------------------------------------------------------- break
class Geometry:
    def __init__(self, prj: Project, vid: int):
        self.mesh = prj.world_mesh(vid)
        self.areas, self.cents, self.norms = face_areas_centroids_normals(self.mesh.vertices, self.mesh.triangles)
        self.pairs, self.lengths = triangle_adjacency(self.mesh.vertices, self.mesh.triangles)


class Breaker:
    """Holds project state shared by all extruder runs of one invocation."""

    def __init__(self, prj: Project, args: argparse.Namespace):
        self.prj = prj
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.phys = prj.physical_extruder_count()
        self.ves = prj.virtual_extruders()
        self.filament_colours = prj.extruder_colours()
        self.ids = IdAllocator([ve["id"] for ve in self.ves], self.phys)
        self.original: Dict[int, Dict[int, str]] = {vid: dict(p.codes) for vid, p in prj.paintings.items()}
        self.geo: Dict[int, Geometry] = {}
        self.frozen: Dict[int, np.ndarray] = {}
        self.colour_map: Dict[int, str] = {}
        self.added: List[dict] = []
        self.layer_h = prj.layer_height()
        self.t0 = time.time()

    def geometry(self, vid: int) -> Geometry:
        if vid not in self.geo:
            self.geo[vid] = Geometry(self.prj, vid)
            self.frozen[vid] = np.zeros(len(self.geo[vid].mesh.triangles), dtype=bool)
        return self.geo[vid]

    # ----------------------------------------------------------------- target state
    def painted_area_by_state(self) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for vid, codes in self.original.items():
            g = self.geometry(vid)
            for t, code in codes.items():
                ls = pc.leaf_states(code)
                for s in ls:
                    out[s] = out.get(s, 0.0) + g.areas[t] / len(ls)
        return out

    def target_units(self, state: int) -> List[Tuple[int, int, float]]:
        """(volume id, root triangle, fraction of leaves in `state`) from the ORIGINAL painting,
        skipping faces already re-painted by an earlier run."""
        units = []
        for vid, codes in self.original.items():
            self.geometry(vid)  # ensures the frozen mask exists
            fr = self.frozen[vid]
            for t, code in codes.items():
                if fr[t]:
                    continue
                ls = pc.leaf_states(code)
                k = sum(1 for s in ls if s == state)
                if k:
                    units.append((vid, t, k / len(ls)))
        return units

    def current_state(self, vid: int, t: int) -> int:
        code = self.prj.paintings[vid].codes.get(t)
        return pc.leaf_states(code)[0] if code else 0

    def state_blend(self, state: int) -> Optional[Dict[int, float]]:
        if 1 <= state <= self.phys:
            return {state: 1.0}
        ve = next((v for v in self.ves if int(v["id"]) == state), None)
        return blend_of(ve) if ve else None

    # ----------------------------------------------------------------------- run
    def run(self, state: int) -> bool:
        a = self.args
        units = self.target_units(state)
        if not units:
            log(f"extruder {state}: no painted triangles found (use `colorbreak inspect`)")
            return False
        volumes = sorted({u[0] for u in units})
        log(f"\n=== extruder {state}: {len(units)} painted triangles in volume(s) {volumes}")

        # --- components and target --------------------------------------------------
        src = self.state_blend(state)
        if a.components:
            filaments = tuple(int(x) for x in a.components.split(","))
        elif src and len(src) >= 2:
            filaments = tuple(sorted(src))
        else:
            log("error: extruder is not a blend; give its physical extruders with --components A,B[,C]")
            return False
        k = len(filaments)
        if a.target_ratio:
            vals = [float(x) for x in a.target_ratio.split(",")]
            if len(vals) == 1 and k == 2:
                vals = [vals[0], 1.0 - vals[0]]
            if len(vals) != k:
                log(f"error: --target-ratio needs {k} values for components {filaments}")
                return False
            nominal_target = np.array(vals) / sum(vals)
        elif src:
            v = vec_over(filaments, src)
            if v is None:
                log(f"error: extruder {state} uses filaments outside {filaments}")
                return False
            nominal_target = v
        else:
            log("error: cannot infer the target ratio; give --target-ratio")
            return False
        mode = a.ratio_mode
        mu = effective_ratio_vec(nominal_target)[0] if mode == "effective" else nominal_target
        log(f"filaments {filaments}; target {fmt_vec(nominal_target, 3)} nominal -> matching {mode} mean {fmt_vec(mu)}")

        # --- levels -----------------------------------------------------------------
        pure_rgb = fit_pure_colours(self.ves, filaments, self.filament_colours)
        levels = build_levels(self.ves, filaments, pure_rgb, self.ids, spec=a.levels, include_pure=not a.no_pure)
        patch_layers = a.patch_size / a.z_stretch / self.layer_h
        if a.max_cycle == "auto":
            max_cycle = int(np.floor(patch_layers))
        elif a.max_cycle == "none":
            max_cycle = 10**9
        else:
            max_cycle = int(a.max_cycle)
        dropped = [l for l in levels if l.cycle > max_cycle]
        levels = [l for l in levels if l.cycle <= max_cycle]
        if dropped:
            log(
                f"dropping levels whose layer cycle exceeds the patch height (~{patch_layers:.0f} layers): "
                + ", ".join(f"{l.label()} (cycle {l.cycle})" for l in dropped)
                + "  [--max-cycle to override]"
            )
        levels, merged = dedupe_levels(levels)
        if merged:
            log("levels printing identically were merged: " + "; ".join(merged))
        if len(levels) < 2:
            log("error: fewer than two usable levels")
            return False
        R = np.stack([l.ratio(mode) for l in levels])
        try:
            dist = fit_distribution(R, mu, PRESETS[a.preset] if a.preset else a.spread)
        except ValueError as e:
            log(f"error: {e}")
            return False
        for w in dist.warnings:
            log(f"warning: {w}")

        # --- geometry ---------------------------------------------------------------
        frac_lookup = {(vid, t): f for vid, t, f in units}
        all_cent, all_area, all_pairs, all_len, all_ext = [], [], [], [], []
        fixed_edges: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (target local idx, other state, length)
        per_vol: Dict[int, np.ndarray] = {}
        offset: Dict[int, int] = {}
        n_total = 0
        for vid in volumes:
            g = self.geometry(vid)
            is_t = np.zeros(len(g.areas), dtype=bool)
            frac = np.zeros(len(g.areas))
            for (v, t), f in frac_lookup.items():
                if v == vid:
                    is_t[t] = True
                    frac[t] = f
            tidx = np.nonzero(is_t)[0]
            local = -np.ones(len(g.areas), dtype=np.int64)
            local[tidx] = np.arange(len(tidx)) + n_total
            pa, pb = local[g.pairs[:, 0]], local[g.pairs[:, 1]]
            both = (pa >= 0) & (pb >= 0)
            one = (pa >= 0) ^ (pb >= 0)
            ext = np.zeros(len(tidx))
            t_side = np.where(pa[one] >= 0, pa[one], pb[one])
            o_side = np.where(pa[one] >= 0, g.pairs[one, 1], g.pairs[one, 0])
            np.add.at(ext, t_side - n_total, g.lengths[one])
            other_state = np.array([self.current_state(vid, int(t)) for t in o_side], dtype=np.int64)
            fixed_edges.append((t_side, other_state, g.lengths[one]))
            all_pairs.append(np.stack([pa[both], pb[both]], axis=1))
            all_len.append(g.lengths[both])
            all_cent.append(g.cents[tidx])
            all_area.append(g.areas[tidx] * frac[tidx])
            all_ext.append(ext)
            per_vol[vid] = tidx
            offset[vid] = n_total
            n_total += len(tidx)
        cent = np.concatenate(all_cent)
        area = np.concatenate(all_area)
        pairs = np.concatenate(all_pairs)
        plen = np.concatenate(all_len)
        ext_len = np.concatenate(all_ext)
        total_area = float(area.sum())
        log(f"target surface: {n_total} triangles, {total_area:.0f} mm^2  ({time.time() - self.t0:.1f}s)")

        # --- cells ------------------------------------------------------------------
        cells = make_cells(
            cent,
            area,
            a.patch_size,
            self.rng,
            z_stretch=a.z_stretch,
            irregularity=a.irregularity,
            size_jitter=a.size_jitter,
        )
        _, frag = components(cells.cell_of, pairs, n_total)
        cell_of = relabel_consecutive(frag)
        n_cells = int(cell_of.max()) + 1
        cell_area = np.bincount(cell_of, weights=area, minlength=n_cells)
        seeds = np.stack([np.bincount(cell_of, weights=area * cent[:, i], minlength=n_cells) for i in range(3)], axis=1)
        seeds /= np.maximum(cell_area, 1e-12)[:, None]
        log(f"cells: {len(cells.seeds)} seeds -> {n_cells} connected cells, mean area {total_area / n_cells:.2f} mm^2")

        # --- values and assignment --------------------------------------------------
        adj01 = cell_adjacency(cell_of, pairs, n_cells)
        mottle = a.mottle_scale if a.mottle_scale else 4.0 * a.patch_size
        z1 = cell_values(seeds, cell_area, adj01, self.rng, coherence=a.coherence, mottle_scale=mottle)
        z2 = (
            cell_values(seeds, cell_area, adj01, self.rng, coherence=a.coherence, mottle_scale=mottle)
            if k >= 3
            else None
        )
        lvl_cell = assign_levels_nd(z1, z2, cell_area, dist.p, R)

        # --- contrast ---------------------------------------------------------------
        f_before = f_after = None
        if a.contrast > 0:
            adjw = cell_adjacency(cell_of, pairs, n_cells, lengths=plen)
            fixed_states = sorted({int(s) for _, st, _ in fixed_edges for s in np.unique(st) if s > 0})
            seqs = [layer_sequence(filaments, l.nominal) for l in levels]
            ratios = [{f: float(r) for f, r in zip(filaments, l.nominal) if r > 1e-9} for l in levels]
            kids = [l.extruder_id for l in levels]
            fixed_state_idx = []
            for s in fixed_states:
                b = self.state_blend(s)
                if b is None:
                    continue
                fil = tuple(sorted(b))
                seqs.append(layer_sequence(fil, [b[f] for f in fil]))
                ratios.append(dict(b))
                kids.append(s)
                fixed_state_idx.append(s)
            K = kernel_matrix(seqs, ratios, kids)
            F = len(fixed_state_idx)
            if F:
                sidx = {s: i for i, s in enumerate(fixed_state_idx)}
                rows, cols, w = [], [], []
                for t_side, st, ln in fixed_edges:
                    keep = np.array([s in sidx for s in st]) if len(st) else np.zeros(0, dtype=bool)
                    rows.append(cell_of[t_side[keep]])
                    cols.append(np.array([n_cells + sidx[int(s)] for s in st[keep]], dtype=np.int64))
                    w.append(ln[keep])
                rows, cols, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)
                extra = sparse.coo_matrix((w, (rows, cols)), shape=(n_cells + F, n_cells + F)).tocsr()
                big = sparse.bmat([[adjw, None], [None, sparse.csr_matrix((F, F))]]).tocsr() + extra + extra.T
            else:
                big = adjw
            fixed_state = np.array([len(levels) + i for i in range(F)], dtype=np.int64)
            cr = apply_contrast(lvl_cell, cell_area, big, K, a.contrast, self.rng, fixed_state=fixed_state)
            lvl_cell = cr.level_of_cell
            f_before, f_after = cr.f_before, cr.f_after
            log(
                f"contrast: aligned border fraction {cr.f_before:.3f} -> {cr.f_after:.3f} (fully annealed {cr.f_min:.3f}, {cr.sweeps} sweeps)"
            )
        lvl = lvl_cell[cell_of]

        # --- cleanup and repair -----------------------------------------------------
        log("cleanup:")
        lvl = cleanup_small_patches(
            lvl, area, pairs, plen, R, mu, a.min_area, a.min_width, ext_boundary=ext_len, cent=cent, log=log
        )
        lvl = repair_mean(lvl, area, pairs, plen, R, mu, tol=a.tolerance, log=log)

        # --- report -----------------------------------------------------------------
        st = patch_stats(lvl, area, pairs, plen, ext_len, cent)
        share = np.bincount(lvl, weights=area, minlength=len(levels)) / total_area
        assign_ids([l for l, s in zip(levels, share) if l.is_new and s > 0], self.ids)
        Rn = np.stack([l.nominal for l in levels])
        Re = np.stack([l.effective for l in levels])
        print(f"\nResult for extruder {state} ({total_area:.0f} mm^2, {st.n} patches), filaments {filaments}:")
        print(f"  {'extruder':>8s} {'nominal':>18s} {'prints':>18s} {'cycle':>5s} {'target%':>8s} {'area%':>7s}")
        for l, p_t, s in zip(levels, dist.p, share):
            if p_t < 5e-4 and s < 5e-4:
                continue
            tag = " (new)" if l.is_new else ""
            shown = l.extruder_id if l.extruder_id >= 0 else 0
            print(
                f"  {shown:8d} {l.label():>18s} {fmt_vec(l.effective, 3):>18s} {l.cycle:5d} {100 * p_t:7.1f}% {100 * s:6.1f}%{tag}"
            )
        mean_nom = share @ Rn
        mean_eff = share @ Re
        std_eff = float(np.sqrt((share[:, None] * (Re - mean_eff) ** 2).sum() / 2.0))
        print(
            f"  area-weighted mean: nominal {fmt_vec(mean_nom)}, printed {fmt_vec(mean_eff)}   (target {mode} {fmt_vec(mu)};"
            f" original extruder {state} prints {fmt_vec(effective_ratio_vec(nominal_target)[0])})"
        )
        print(f"  spread of patch ratios (printed): {std_eff:.4f}")
        print(
            f"  extra per-patch randomness from layer phase (~{patch_layers:.0f} layers per patch): {phase_std(levels, share, patch_layers, filaments):.4f}"
        )
        if f_before is not None:
            print(f"  aligned border fraction: {f_before:.3f} -> {f_after:.3f}")
        q = np.percentile(st.area, [5, 50, 95])
        wq = np.percentile(st.width[np.isfinite(st.width)], [5, 50, 95])
        print(f"  patch area  mm^2  5%/50%/95%: {q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f}")
        print(f"  patch width mm    5%/50%/95%: {wq[0]:.2f} / {wq[1]:.2f} / {wq[2]:.2f}")

        # --- apply to the painting --------------------------------------------------
        used_new = [l for l, s in zip(levels, share) if l.is_new and s > 0]
        assign_ids(used_new, self.ids)
        for l in used_new:
            entry = new_virtual_extruder_entry(l, filaments)
            self.ves.append(entry)
            self.added.append(entry)
            l.is_new = False
        if used_new:
            log(f"added {len(used_new)} virtual extruder(s): " + ", ".join(str(l.extruder_id) for l in used_new))
        for l in levels:
            self.colour_map[l.extruder_id] = l.colour
        ids = np.array([l.extruder_id for l in levels])
        for vid in volumes:
            p = self.prj.paintings[vid]
            sel = per_vol[vid]
            new_state = ids[lvl[offset[vid] : offset[vid] + len(sel)]]
            for t, s in zip(sel, new_state):
                p.codes[int(t)] = pc.remap(p.codes[int(t)], {state: int(s)})
            self.frozen[vid][sel] = True
        return True

    # ------------------------------------------------------------------- finish
    def finalize_versions(self) -> None:
        for p in self.prj.paintings.values():
            used = set()
            for code in p.codes.values():
                used.update(pc.leaf_states(code))
            p.version = pc.required_version(used)

    def colour_of_state(self, s: int) -> str:
        if s in self.colour_map:
            return self.colour_map[s]
        ve = next((v for v in self.ves if int(v["id"]) == s), None)
        if ve and ve.get("color"):
            return ve["color"]
        if 1 <= s <= len(self.filament_colours):
            return self.filament_colours[s - 1]
        return "#c8c8c8"

    def preview(self, png: Optional[str], ply: Optional[str], scale: float) -> None:
        """One picture / PLY of the whole model: all volumes concatenated."""
        cents, norms, cols, verts, tris = [], [], [], [], []
        n_v = 0
        for vid in sorted(self.prj.paintings):
            g = self.geometry(vid)
            col = np.full((len(g.mesh.triangles), 3), 200.0)
            cache: Dict[int, np.ndarray] = {}
            for t, code in self.prj.paintings[vid].codes.items():
                s = pc.leaf_states(code)[0]
                if s not in cache:
                    cache[s] = colours_from_hex([self.colour_of_state(s)])[0]
                col[t] = cache[s]
            cents.append(g.cents)
            norms.append(g.norms)
            cols.append(col)
            verts.append(g.mesh.vertices)
            tris.append(g.mesh.triangles + n_v)
            n_v += len(g.mesh.vertices)
        if png:
            render_views(np.concatenate(verts), np.concatenate(tris), np.concatenate(cols), png, px_per_mm=scale)
            log(f"preview written to {png}")
        if ply:
            write_ply(ply, np.concatenate(verts), np.concatenate(tris), np.concatenate(cols))
            log(f"PLY written to {ply}")


def cmd_break(args: argparse.Namespace) -> int:
    prj = Project(args.input)
    br = Breaker(prj, args)
    if args.extruder.lower() == "all":
        if args.components or args.target_ratio:
            log("error: --components / --target-ratio can only be used with a single extruder")
            return 2
        area_by_state = br.painted_area_by_state()
        states = [s for s, ar in area_by_state.items() if s > br.phys and ar > 1e-9 and br.state_blend(s)]
        states.sort(key=lambda s: -area_by_state[s])
        if not states:
            log("error: no painted virtual extruders found")
            return 2
        log(
            "breaking virtual extruders (largest area first): "
            + ", ".join(f"{s} ({area_by_state[s]:.0f} mm^2)" for s in states)
        )
    else:
        states = [int(args.extruder)]
    done = 0
    for s in states:
        if br.run(s):
            done += 1
    if done == 0:
        return 2
    if args.preview or args.ply:
        br.preview(args.preview, args.ply, args.preview_scale)
    if args.dry_run:
        log("dry run: no file written")
        return 0
    br.finalize_versions()
    prj.write(args.output, project_modified=bool(br.added))
    log(f"written {args.output}  ({time.time() - br.t0:.1f}s)")
    return 0


# --------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="colorbreak", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="list extruders, virtual extruders and painted areas of a 3mf")
    pi.add_argument("input")
    pi.set_defaults(func=cmd_inspect)

    pb = sub.add_parser(
        "break", help="break the surface painted with one (or every) virtual extruder into random patches"
    )
    pb.add_argument("input")
    pb.add_argument("-o", "--output", required=True, help="output .3mf")
    pb.add_argument(
        "-e",
        "--extruder",
        required=True,
        help="extruder (triangle state) to break up, e.g. 17, or 'all' for every painted virtual extruder",
    )
    pb.add_argument(
        "--components", help="physical extruders A,B[,C] of the blend (only needed if -e is not a virtual extruder)"
    )
    pb.add_argument(
        "--target-ratio",
        help="nominal ratio to aim for: one value (fraction of the first filament, 2 colours) or k comma-separated values in the order of the filaments (default: the source extruder's ratio)",
    )
    pb.add_argument(
        "--ratio-mode",
        choices=["effective", "nominal"],
        default="effective",
        help="match the ratio PrusaSlicer actually prints (default) or the nominal one",
    )
    g = pb.add_argument_group("look")
    g.add_argument(
        "--spread", type=float, default=0.08, help="spread of the patch ratios around the target (default 0.08)"
    )
    g.add_argument("--preset", choices=sorted(PRESETS), help="subtle/medium/bold = spread 0.05/0.08/0.12")
    g.add_argument("--patch-size", type=float, default=2.5, help="typical patch width in mm (default 2.5)")
    g.add_argument(
        "--z-stretch", type=float, default=0.65, help="patches are 1/z-stretch times taller than wide (default 0.65)"
    )
    g.add_argument("--irregularity", type=float, default=0.35, help="wobbliness of patch borders, 0..1 (default 0.35)")
    g.add_argument("--size-jitter", type=float, default=0.5, help="variation of patch sizes, 0..1 (default 0.5)")
    g.add_argument(
        "--coherence", type=float, default=0.5, help="correlation of neighbouring patch tones, 0..1 (default 0.5)"
    )
    g.add_argument(
        "--mottle-scale", type=float, help="length scale of large-scale tone variation in mm (default 4 x patch size)"
    )
    g.add_argument(
        "--contrast", type=float, default=0.0, help="0..1: make neighbouring patches print out of phase (0 = off)"
    )
    g.add_argument(
        "--levels",
        default="auto",
        help="'auto' (existing virtual extruders), 'step:0.1' (simplex grid) or explicit 'a/b/c,...' (missing ones are created)",
    )
    g.add_argument("--no-pure", action="store_true", help="do not use the pure physical colours as patch levels")
    g.add_argument(
        "--max-cycle",
        default="auto",
        help="drop levels whose layer cycle is longer than this ('auto' = patch height in layers, 'none')",
    )
    g.add_argument("--seed", type=int, default=0)
    p = pb.add_argument_group("printability")
    p.add_argument("--min-area", type=float, default=3.0, help="merge patches smaller than this (mm^2, default 3)")
    p.add_argument("--min-width", type=float, default=1.1, help="merge patches thinner than this (mm, default 1.1)")
    p.add_argument("--tolerance", type=float, default=1e-4, help="allowed deviation of the mean ratio (default 1e-4)")
    o = pb.add_argument_group("output")
    o.add_argument("--preview", help="write a PNG preview (3 orthographic views of the whole model)")
    o.add_argument("--preview-scale", type=float, default=14.0, help="preview pixels per mm (default 14)")
    o.add_argument("--ply", help="write a PLY with face colours")
    o.add_argument("--dry-run", action="store_true", help="compute and report, do not write the 3mf")
    pb.set_defaults(func=cmd_break)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
