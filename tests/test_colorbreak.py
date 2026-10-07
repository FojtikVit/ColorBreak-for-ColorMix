import json
import os
import zipfile

import numpy as np
import pytest
from scipy import sparse

from colorbreak import paintcodes as pc
from colorbreak.assign import (
    assign_levels,
    assign_levels_nd,
    fit_distribution,
    fit_distribution_1d,
    tilt_to_mean,
    tilt_to_mean_vec,
)
from colorbreak.cleanup import (
    cleanup_small_patches,
    face_areas_centroids_normals,
    patch_stats,
    repair_mean,
    triangle_adjacency,
)
from colorbreak.contrast import (
    alignment,
    apply_contrast,
    kernel_matrix,
)
from colorbreak.levels import (
    IdAllocator,
    assign_ids,
    build_levels,
    canonical_cycle,
    dedupe_levels,
    effective_ratio,
    effective_ratio_vec,
    fit_pure_colours,
    layer_sequence,
    parse_level_spec,
)
from colorbreak.patches import make_cells

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLES = [os.path.join(ROOT, f) for f in ("sheep_plain.3mf", "sheep_pretty.3mf", "pig.3mf")]

R2 = np.array(
    [
        [0, 1],
        [1 / 13, 12 / 13],
        [1 / 8, 7 / 8],
        [1 / 6, 5 / 6],
        [0.2, 0.8],
        [0.25, 0.75],
        [2 / 7, 5 / 7],
        [1 / 3, 2 / 3],
        [0.4, 0.6],
        [3 / 7, 4 / 7],
        [0.5, 0.5],
        [1, 0],
    ]
)


# ----------------------------------------------------------------------------- codes
@pytest.mark.parametrize(
    "code,states",
    [("4", [1]), ("8", [2]), ("0", [0]), ("CC", [15]), ("1C", [4]), ("00EC", [17]), ("02EC", [19]), ("EEEC", [255])],
)
def test_leaf_codes(code, states):
    assert pc.leaf_states(code) == states
    assert pc.encode(pc.decode(code)) == code
    assert pc.leaf_code(states[0]) == code


def test_split_code_roundtrip():
    code = "00EC4443444344434443"
    root = pc.decode(code)
    assert not root.is_leaf
    assert pc.encode(root) == code
    assert set(root.leaf_states()) <= {1, 17}
    remapped = pc.remap(code, {17: 42})
    assert set(pc.leaf_states(remapped)) <= {1, 42}
    assert pc.remap(remapped, {42: 17}) == code


def test_required_version():
    assert pc.required_version({0, 1, 16}) == 1
    assert pc.required_version({1, 17}) == 2


@pytest.mark.parametrize("path", SAMPLES)
def test_roundtrip_sample_files(path):
    if not os.path.exists(path):
        pytest.skip("sample file missing")
    data = json.loads(zipfile.ZipFile(path).read("Metadata/Slic3r_facets_annotation.json"))
    for entry in data:
        for f in entry.get("mmSegmentationFacets", []):
            assert pc.encode(pc.decode(f["dividing"])) == f["dividing"]


# ---------------------------------------------------------------------------- levels
@pytest.mark.parametrize(
    "r,eff,cycle",
    [
        (0.15, 1 / 6, 6),
        (0.10, 1 / 8, 8),
        (0.05, 1 / 13, 13),
        (0.30, 2 / 7, 7),
        (0.5, 0.5, 2),
        (0.0, 0.0, 1),
        (1.0, 1.0, 1),
    ],
)
def test_effective_ratio(r, eff, cycle):
    e, c = effective_ratio(r)
    assert c == cycle
    assert e == pytest.approx(eff)


def test_effective_ratio_vec_three_colours():
    e, c = effective_ratio_vec([0.35, 0.15, 0.5])
    assert c == 6
    assert np.allclose(e, [2 / 6, 1 / 6, 3 / 6])
    e, c = effective_ratio_vec([0.0, 0.15, 0.85])  # 2-colour point of the 3-simplex
    assert c == 6 and e[0] == 0 and e[1] == pytest.approx(1 / 6)
    e, c = effective_ratio_vec([0, 0, 1])
    assert c == 1 and e[2] == 1


def test_layer_sequence_and_cycle():
    seq = layer_sequence((1, 2, 4), [0.35, 0.15, 0.5])
    assert len(seq) == 6 and sorted(set(seq)) == [1, 2, 4]
    assert seq.count(4) == 3 and seq.count(1) == 2
    for r in (0.15, 0.3, 0.45, 0.5, 0.85):
        s = canonical_cycle(r)
        e, c = effective_ratio(r)
        assert len(s) == c and sum(s) / len(s) == pytest.approx(e)


def test_parse_level_spec():
    g = parse_level_spec("step:0.5", 3)
    assert len(g) == 3  # the three edge midpoints; vertices are pure and excluded
    assert parse_level_spec("auto", 2) is None
    e = parse_level_spec("0.2,0.4", 2)
    assert np.allclose(e[1], [0.4, 0.6])
    e = parse_level_spec("0.2/0.3/0.5", 3)
    assert np.allclose(e[0], [0.2, 0.3, 0.5])


def test_build_levels_creates_and_dedupes():
    ves = [
        {
            "id": 5,
            "kind": "fullspectrum",
            "color": "#404040",
            "components": [{"extruder": 1, "ratio": 0.5}, {"extruder": 2, "ratio": 0.5}],
        }
    ]
    ids = IdAllocator([5], 4)
    pure = fit_pure_colours(ves, (1, 2), ["#000000", "#ffffff", "#ff0000", "#00ff00"])
    lv = build_levels(ves, (1, 2), pure, ids, spec="0.5,0.2,0.175")
    assert [l.extruder_id for l in lv if not l.is_new and l.n_colours == 2] == [5]
    assert sum(l.is_new for l in lv) == 2
    assert all(l.extruder_id < 0 for l in lv if l.is_new)  # placeholders until used
    assign_ids([l for l in lv if l.is_new], ids)
    assert sorted(l.extruder_id for l in lv if l.is_new) == [6, 7]
    kept, notes = dedupe_levels(lv)
    assert len(kept) == len(lv) - 1 and notes  # 20% and 17.5% both print 1-in-5


# ------------------------------------------------------------------------ distribution
@pytest.mark.parametrize("mu,sigma", [(0.1667, 0.05), (0.1667, 0.08), (0.1667, 0.12), (0.5, 0.1)])
def test_fit_1d_hits_mean_and_spread(mu, sigma):
    p, params, w = fit_distribution_1d(R2[:, 0], mu, sigma)
    assert p.sum() == pytest.approx(1.0) and (p >= 0).all()
    assert (p * R2[:, 0]).sum() == pytest.approx(mu, abs=1e-9)
    assert np.sqrt((p * (R2[:, 0] - mu) ** 2).sum()) == pytest.approx(sigma, rel=0.05)


def test_fit_vector_two_colours_matches_1d():
    d = fit_distribution(R2, np.array([1 / 6, 5 / 6]), 0.08)
    assert np.allclose(d.mean, [1 / 6, 5 / 6], atol=1e-9)
    assert d.std == pytest.approx(0.08, rel=0.05)


def three_colour_levels():
    pts = [
        (0.35, 0.15, 0.5),
        (0.5, 0.15, 0.35),
        (0.5, 0.35, 0.15),
        (0.35, 0.5, 0.15),
        (0.15, 0.5, 0.35),
        (0.33, 0.33, 0.34),
        (0.15, 0.15, 0.7),
        (0.7, 0.15, 0.15),
        (0.15, 0.7, 0.15),
        (0.45, 0.1, 0.45),
        (0.45, 0.45, 0.1),
        (0.1, 0.45, 0.45),
        (0.5, 0.25, 0.25),
        (0.25, 0.5, 0.25),
        (0.25, 0.25, 0.5),
    ]
    for a in (0.2, 0.4, 0.6, 0.8):
        pts += [(a, 1 - a, 0), (a, 0, 1 - a), (0, a, 1 - a)]
    pts += [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    return np.stack([effective_ratio_vec(p)[0] for p in pts])


@pytest.mark.parametrize("sigma", [0.05, 0.08, 0.15])
def test_fit_three_colours(sigma):
    R = three_colour_levels()
    mu = effective_ratio_vec([0.35, 0.15, 0.5])[0]
    d = fit_distribution(R, mu, sigma)
    assert np.allclose(d.mean, mu, atol=1e-8)
    assert d.std == pytest.approx(sigma, rel=0.05)
    assert (d.p >= 0).all() and d.p.sum() == pytest.approx(1.0)


def test_fit_three_colours_target_on_edge_uses_edge_levels():
    R = three_colour_levels()
    mu = np.array([0.2, 0.8, 0.0])
    d = fit_distribution(R, mu, 0.08)
    assert np.allclose(d.mean, mu, atol=1e-9)
    assert (d.p[R[:, 2] > 0] == 0).all()


def test_tilt_vec_and_scalar():
    q = np.full(4, 0.25)
    r = np.array([0.0, 0.1, 0.2, 0.3])
    p, lam = tilt_to_mean(q, r, 0.1)
    assert (p * r).sum() == pytest.approx(0.1) and lam < 0
    with pytest.raises(ValueError):
        tilt_to_mean(q, r, 0.5)
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3]], dtype=float)
    p, lam, ok = tilt_to_mean_vec(np.full(4, 0.25), R, np.array([0.5, 0.3, 0.2]))
    assert ok and np.allclose(p @ R, [0.5, 0.3, 0.2], atol=1e-8)


def test_assign_levels_matches_fractions():
    rng = np.random.default_rng(1)
    z = rng.standard_normal(5000)
    a = rng.uniform(0.5, 1.5, 5000)
    p = np.array([0.1, 0.3, 0.6])
    lvl = assign_levels(z, a, p)
    share = np.bincount(lvl, weights=a, minlength=3) / a.sum()
    assert np.allclose(share, p, atol=2e-3)
    assert z[lvl == 0].max() <= z[lvl == 1].min() and z[lvl == 1].max() <= z[lvl == 2].min()


def test_assign_levels_nd_matches_fractions():
    R = three_colour_levels()
    mu = effective_ratio_vec([0.35, 0.15, 0.5])[0]
    d = fit_distribution(R, mu, 0.12)
    rng = np.random.default_rng(2)
    n = 3000
    z1, z2, a = rng.standard_normal(n), rng.standard_normal(n), rng.uniform(0.5, 1.5, n)
    lvl = assign_levels_nd(z1, z2, a, d.p, R)
    share = np.bincount(lvl, weights=a, minlength=len(R)) / a.sum()
    assert np.abs(share - d.p).max() < 3e-3
    assert np.allclose(share @ R, mu, atol=3e-3)


# ---------------------------------------------------------------------------- contrast
def test_alignment_kernel_properties():
    s1 = layer_sequence((1, 2), [0.15, 0.85])
    s2 = layer_sequence((1, 2), [0.2, 0.8])
    assert alignment(s1, s1, {1: 1 / 6, 2: 5 / 6}, {1: 1 / 6, 2: 5 / 6}) > 0.2  # same cycle = aligned
    assert alignment(s1, s2, {1: 1 / 6, 2: 5 / 6}, {1: 0.2, 2: 0.8}) < 0.05  # coprime cycles ~ chance
    K = kernel_matrix([s1, s2, [2]], [{1: 1 / 6, 2: 5 / 6}, {1: 0.2, 2: 0.8}, {2: 1.0}], [17, 16, 2])
    assert K[0, 0] > K[0, 1] and np.allclose(K, K.T)
    assert K[2, 2] == pytest.approx(0.1)  # pure vs same pure: only the equal-level bonus


def test_contrast_reduces_alignment_and_keeps_fractions():
    rng = np.random.default_rng(0)
    n = 30
    # grid graph of cells
    idx = np.arange(n * n).reshape(n, n)
    rows = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
    cols = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])
    adj = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n)).tocsr()
    adj = adj + adj.T
    area = rng.uniform(0.9, 1.1, n * n)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    K = np.eye(4) * 0.5 + 0.1
    lvl = assign_levels(rng.standard_normal(n * n), area, p)
    res = apply_contrast(lvl, area, adj, K, 1.0, rng, max_sweeps=60)
    assert res.f_after < 0.5 * res.f_before
    before = np.bincount(lvl, weights=area, minlength=4)
    after = np.bincount(res.level_of_cell, weights=area, minlength=4)
    assert np.abs(after - before).max() / area.sum() < 0.02
    half = apply_contrast(lvl, area, adj, K, 0.5, rng, max_sweeps=60)
    assert res.f_after - 1e-9 <= half.f_after <= half.f_before


# -------------------------------------------------------------------- synthetic mesh
def grid_mesh(nx, ny, size):
    xs = np.linspace(0, size, nx + 1)
    ys = np.linspace(0, size, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.stack([X.ravel(), np.zeros(X.size), Y.ravel()], axis=1)
    idx = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    T = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return V, T


def test_end_to_end_on_flat_wall_two_colours():
    V, T = grid_mesh(160, 160, 40.0)
    areas, cent, _ = face_areas_centroids_normals(V, T)
    pairs, lengths = triangle_adjacency(V, T)
    rng = np.random.default_rng(3)
    cells = make_cells(cent, areas, 2.5, rng)
    n_cells = cells.cell_of.max() + 1
    assert 150 < n_cells < 400
    R = R2[:8]
    mu = np.array([1 / 6, 5 / 6])
    d = fit_distribution(R, mu, 0.08)
    cell_area = np.bincount(cells.cell_of, weights=areas, minlength=n_cells)
    lvl = assign_levels_nd(rng.standard_normal(n_cells), None, cell_area, d.p, R)[cells.cell_of]
    lvl = cleanup_small_patches(lvl, areas, pairs, lengths, R, mu, 3.0, 1.1, cent=cent)
    lvl = repair_mean(lvl, areas, pairs, lengths, R, mu, tol=1e-4)
    st = patch_stats(lvl, areas, pairs, lengths, cent=cent)
    mean = (areas[:, None] * R[lvl]).sum(axis=0) / areas.sum()
    assert np.linalg.norm(mean - mu) / np.sqrt(2) < 1e-4
    assert st.area.min() >= 3.0 - 1e-9 or st.n == 1
    assert st.width.min() >= 1.1 - 1e-9 or st.n == 1


def test_repair_three_colours():
    V, T = grid_mesh(200, 200, 60.0)
    areas, cent, _ = face_areas_centroids_normals(V, T)
    pairs, lengths = triangle_adjacency(V, T)
    rng = np.random.default_rng(5)
    R = three_colour_levels()
    mu = effective_ratio_vec([0.35, 0.15, 0.5])[0]
    cells = make_cells(cent, areas, 2.5, rng)
    n_cells = cells.cell_of.max() + 1
    cell_area = np.bincount(cells.cell_of, weights=areas, minlength=n_cells)
    d = fit_distribution(R, mu, 0.1)
    lvl = assign_levels_nd(rng.standard_normal(n_cells), rng.standard_normal(n_cells), cell_area, d.p, R)[cells.cell_of]
    lvl = cleanup_small_patches(lvl, areas, pairs, lengths, R, mu, 3.0, 1.1, cent=cent)
    lvl = repair_mean(lvl, areas, pairs, lengths, R, mu, tol=1e-4)
    mean = (areas[:, None] * R[lvl]).sum(axis=0) / areas.sum()
    assert np.linalg.norm(mean - mu) / np.sqrt(2) < 1e-4
