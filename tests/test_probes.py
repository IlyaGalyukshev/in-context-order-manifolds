"""Correctness tests for the offline probe catalog, on SYNTHETIC activations
with a known geometry. A shared rank direction across stimuli must be decodable
(interior survives) and a noise fixture must collapse — the same fork the real
experiment resolves.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from icom.probes import (depth_stats, intrinsic_dim, load_records, mlp_cv_spearman,
                         probe_with_null, rsa_rank, stack_layer, transfer_spearman, twonn)
from icom.probes.geometry import project
from icom.probes.linear import cv_spearman, reduce

D = 24
LAYERS = 7
SIG = 4                       # the manifold lives at this layer; others are noise


def _basis(seed):
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((D, 4)))[0]     # orthonormal columns
    return B[:, 0], B[:, 1], B[:, 2], B[:, 3]


def _emit(kind, r, N, u, v, w, x2, rng, amp=1.0, noise=0.05):
    """Feature vector for rank r (1..N) under a fixture geometry."""
    base = noise * rng.standard_normal(D)
    if kind == "arc":                                    # semicircle: linearly decodable
        th = np.pi * (r - 1) / (N - 1)
        return base + amp * (np.cos(th) * u + np.sin(th) * v)
    if kind == "ring":                                   # 300° arc: injective but
        th = np.deg2rad(300) * (r - 1) / (N - 1)         # non-monotonic in any linear
        return base + amp * (np.cos(th) * u + np.sin(th) * v)  # projection -> nonlinear wins
    if kind == "grid":                                   # 2D sheet: rank axis + independent axis
        cx = (r - 1) / (N - 1)
        cy = rng.random()
        return base + amp * (cx * u + cy * v)
    return base                                          # "null": pure noise


def _write_acts(root, kind, model="synth", family="s1_size", condition="shuffle",
                Ns=(9, 12), n_stim=14, seed=0):
    u, v, w, x2 = _basis(seed + 1)
    out = Path(root) / model
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_stim):
        N = Ns[i % len(Ns)]
        X = np.zeros((N, LAYERS, D), dtype=np.float32)
        for li in range(LAYERS):
            for ri in range(N):
                if li == SIG:
                    X[ri, li] = _emit(kind, ri + 1, N, u, v, w, x2, rng)
                else:
                    X[ri, li] = 0.05 * rng.standard_normal(D)
        meta = {"family": family, "condition": condition, "n_items": N,
                "content_key": f"{kind}{i}", "model": model}
        np.savez_compressed(out / f"{kind}{i}.npz",
                            readout=X, ranks=np.arange(1, N + 1),
                            slots=np.arange(1, N + 1),
                            entities=json.dumps([f"e{j}" for j in range(N)]),
                            meta=json.dumps(meta))
    return root


# --------------------------------------------------------------- linear probe
def test_interior_survives_on_manifold_collapses_on_null(tmp_path):
    aacts = _write_acts(tmp_path / "arc", "arc", seed=1)
    nacts = _write_acts(tmp_path / "null", "null", seed=2)
    ra = load_records(aacts, "synth", "s1_size", "shuffle", "readout", N=12)
    rn = load_records(nacts, "synth", "s1_size", "shuffle", "readout", N=12)
    Xa, ya, ga = stack_layer(ra, SIG, interior_only=True)
    Xn, yn, gn = stack_layer(rn, SIG, interior_only=True)
    real_a, null95_a, p_a = probe_with_null(Xa, ya, ga, n_perm=40, seed=0)
    real_n, null95_n, p_n = probe_with_null(Xn, yn, gn, n_perm=40, seed=0)
    assert real_a > null95_a and p_a < 0.05, (real_a, null95_a, p_a)   # SURVIVES
    assert not (real_n > null95_n and p_n < 0.05), (real_n, null95_n, p_n)  # COLLAPSES


def test_noise_layers_are_null_but_signal_layer_decodes(tmp_path):
    acts = _write_acts(tmp_path / "arc", "arc", seed=3)
    recs = load_records(acts, "synth", "s1_size", "shuffle", "readout", N=12)
    sig = cv_spearman(reduce(stack_layer(recs, SIG, True)[0]), *stack_layer(recs, SIG, True)[1:])
    noise_layer = 1
    noi = cv_spearman(reduce(stack_layer(recs, noise_layer, True)[0]),
                      *stack_layer(recs, noise_layer, True)[1:])
    assert sig > 0.8 and noi < 0.5, (sig, noi)


# --------------------------------------------------------------- RSA (metric)
def test_rsa_tracks_rank_on_manifold_not_on_null(tmp_path):
    ra = load_records(_write_acts(tmp_path / "arc", "arc", seed=4), "synth", "s1_size", "shuffle", "readout")
    rn = load_records(_write_acts(tmp_path / "null", "null", seed=5), "synth", "s1_size", "shuffle", "readout")
    rho_a, na = rsa_rank(ra, SIG, interior_only=True)
    rho_n, nn = rsa_rank(rn, SIG, interior_only=True)
    assert rho_a > 0.7 and na > 0, (rho_a, na)
    assert abs(rho_n) < 0.4, rho_n


# --------------------------------------------------------------- curvature
def test_nonlinear_beats_linear_on_ring(tmp_path):
    recs = load_records(_write_acts(tmp_path / "ring", "ring", Ns=(12,), n_stim=30, seed=6),
                        "synth", "s1_size", "shuffle", "readout", N=12)
    X, y, g = stack_layer(recs, SIG, interior_only=False)
    lin = cv_spearman(reduce(X), y, g)
    nonlin = mlp_cv_spearman(X, y, g)
    assert nonlin > lin + 0.08 and nonlin > 0.8, (lin, nonlin)   # curvature captured


# --------------------------------------------------------------- intrinsic dim
def test_twonn_recovers_dimension_on_random_clouds():
    """TwoNN on irregularly-sampled clean manifolds recovers the true dimension.
    (A perfectly uniform grid is a documented degenerate case -> nan, not used.)"""
    rng = np.random.default_rng(0)
    assert abs(twonn(rng.random((600, 1))) - 1) < 0.3, twonn(rng.random((600, 1)))
    assert abs(twonn(rng.random((600, 2))) - 2) < 0.4, twonn(rng.random((600, 2)))
    assert abs(twonn(rng.random((600, 3))) - 3) < 0.6, twonn(rng.random((600, 3)))
    assert np.isnan(twonn(np.linspace(0, 1, 400).reshape(-1, 1)))   # uniform grid -> nan


def _dim_recs(kind, N=16, nst=30, noise=0.01, seed=0):
    """recs with irregular (order-preserving) sampling so TwoNN is well-posed."""
    rng = np.random.default_rng(seed)
    u, v, _, _ = _basis(seed + 1)
    out = []
    for _ in range(nst):
        X = np.zeros((N, LAYERS, D), np.float32)
        ths = np.sort(rng.uniform(0, np.pi, N))            # irregular angles, sorted -> rank
        for r in range(1, N + 1):
            s = (np.cos(ths[r - 1]) * u + np.sin(ths[r - 1]) * v) if kind == "arc" \
                else ((r - 1) / (N - 1) * u + rng.random() * v)
            X[r - 1, SIG] = noise * rng.standard_normal(D) + s
        out.append({"X": X, "ranks": np.arange(1, N + 1), "N": N})
    return out


def test_intrinsic_dim_1d_arc_below_2d_grid():
    id_arc = intrinsic_dim(_dim_recs("arc", seed=7), SIG, interior_only=False)
    id_grid = intrinsic_dim(_dim_recs("grid", seed=8), SIG, interior_only=False)
    assert id_arc < id_grid - 0.3, (id_arc, id_grid)     # 1D order < 2D grid
    assert id_arc < 2.3, id_arc


# --------------------------------------------------------------- transfer
def test_transfer_across_N_on_manifold(tmp_path):
    acts = _write_acts(tmp_path / "arc", "arc", Ns=(9, 12), n_stim=20, seed=9)
    recs9 = load_records(acts, "synth", "s1_size", "shuffle", "readout", N=9)
    recs12 = load_records(acts, "synth", "s1_size", "shuffle", "readout", N=12)
    a = stack_layer(recs9, SIG, interior_only=True)
    b = stack_layer(recs12, SIG, interior_only=True)
    assert transfer_spearman(a[0], a[1], b[0], b[1]) > 0.7   # shared rank code transfers


# --------------------------------------------------------------- depth stats
def test_depth_stats_onset_peak_band():
    scores = [0.10, 0.10, 0.20, 0.50, 0.70, 0.60, 0.30]
    null95 = [0.15] * 7
    d = depth_stats(scores, null95)
    assert d["onset_layer"] == 2 and d["peak_layer"] == 4
    assert d["band_layers"] == [2, 6]
    assert d["peak_frac"] == pytest.approx(4 / 6, abs=1e-3)


# --------------------------------------------------------------- projection (viz math)
def test_projection_and_figure(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import visualize_manifold as vm
    aacts = _write_acts(tmp_path / "arc", "arc", seed=10)
    nacts = _write_acts(tmp_path / "null", "null", seed=11)
    # projection axis aligns with rank on the arc, not on the null
    Xa, ca = vm.collect(aacts, "synth", "s1_size", "shuffle", "readout", SIG, interior_only=True)
    Xn, cn = vm.collect(nacts, "synth", "s1_size", "shuffle", "readout", SIG, interior_only=True)
    Pa, Pn = project(Xa, "pca", 2), project(Xn, "pca", 2)
    assert vm.rank_axis_alignment(Pa, ca) > 0.8
    assert vm.rank_axis_alignment(Pn, cn) < 0.5
    # contact sheet + 3D html write real files
    png = tmp_path / "sheet.png"
    vm.contact_sheet([("order", str(aacts)), ("coherence-null", str(nacts))],
                     "synth", "s1_size", "shuffle", "readout", [1, SIG], "pca", True, str(png))
    assert png.exists() and png.stat().st_size > 1000
    html = tmp_path / "m3d.html"
    vm.interactive3d(str(aacts), "synth", "s1_size", "shuffle", "readout", SIG, "pca", True, str(html))
    assert html.exists() and html.stat().st_size > 1000


# ---------------------------------------- FWER-corrected verdict (winner's curse)
def _sweep():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import probe_sweep as ps
    return ps


def test_fwer_verdict_manifold_on_arc_not_on_noise(tmp_path):
    ps = _sweep()
    arc = _write_acts(tmp_path / "arc", "arc", Ns=(12,), n_stim=16, seed=20)
    noise = _write_acts(tmp_path / "null", "null", Ns=(12,), n_stim=16, seed=21)
    s_arc, _ = ps.sweep_cell(str(arc), "synth", "s1_size", "shuffle", 12, "readout", 80, 0)
    s_noi, _ = ps.sweep_cell(str(noise), "synth", "s1_size", "shuffle", 12, "readout", 80, 0)
    assert s_arc["verdict"] == "MANIFOLD" and s_arc["interior_peak_p_fwer"] < 0.05
    assert s_arc["depth_peak_layer"] == SIG                 # localizes the signal layer
    # under a global null the max-over-layers correction must NOT declare a manifold
    assert s_noi["verdict"] != "MANIFOLD" and s_noi["interior_peak_p_fwer"] > 0.05


# ---------------------------------------- hardening (twonn / null / non-finite)
def test_twonn_edge_cases():
    rng = np.random.default_rng(0)
    assert np.isnan(twonn(rng.random((4, 2))))             # <5 survivors -> nan, never +inf
    v = twonn(rng.random((30, 2)))
    assert np.isfinite(v) and not np.isinf(v)


def test_probe_with_null_degenerate_is_nan_not_tiny():
    X = np.ones((20, 5))                                    # constant -> spearman nan
    y = np.tile(np.linspace(0, 1, 5), 4)
    g = np.repeat(np.arange(4), 5)
    r, n95, p = probe_with_null(X, y, g, n_perm=20)
    assert np.isnan(p), (r, n95, p)                         # not the spurious 1/(1+n_perm)


def test_stack_layer_drops_nonfinite_rows(tmp_path):
    acts = _write_acts(tmp_path / "arc", "arc", Ns=(12,), n_stim=6, seed=30)
    recs = load_records(acts, "synth", "s1_size", "shuffle", "readout", N=12)
    recs[0]["X"][2, SIG, 0] = np.inf                        # poison one entity at SIG
    X, y, g = stack_layer(recs, SIG, interior_only=False)
    assert np.isfinite(X).all()                            # poisoned row dropped, no crash
