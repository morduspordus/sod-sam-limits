"""The band-restricted cut (sodsam.fusion_band) equals the full cut with the fixed pixels clamped by
+-BIG terminal capacities, and fuse_full equals fusion2's linear/paper model with --band."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import fusion
from sodsam.fusion import BIG, color_weights, energy, minimize, pairwise_tables
from sodsam.fusion2 import fuse2, prepare
from sodsam.fusion_band import fuse_full, linear_unary, load_linear_params, minimize_band

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fusion2 import random_input, random_params


def random_problem(rng, h, w):
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    unary1 = rng.uniform(-60, 60, (h, w))
    m_sam = rng.integers(0, 4, (h, w)).astype(np.float64)
    w_r, w_d = color_weights(img, rng.uniform(5, 50), rng.uniform(0, 30), 0.01)
    tables = pairwise_tables(m_sam, w_r, w_d, rng.uniform(-30, 0), "full", "paper")
    return unary1, tables


def test_band_cut_equals_clamped_cut():
    rng = np.random.default_rng(0)
    for t in range(100):
        h, w = rng.integers(2, 12, 2)
        unary1, tables = random_problem(rng, h, w)
        free = rng.random((h, w)) < rng.uniform(0.1, 0.9)
        fixed = rng.random((h, w)) < 0.5
        x = minimize_band(unary1, tables, free, fixed)
        assert x.dtype == bool and x.shape == (h, w)
        assert np.array_equal(x[~free], fixed[~free])
        ref, n_clipped = minimize(np.where(free, unary1, np.where(fixed, -BIG, BIG)), tables)
        assert n_clipped == 0
        assert abs(energy(x, unary1, tables) - energy(ref, unary1, tables)) < 1e-6, t
        assert np.array_equal(x, ref), t
    # everything free: the plain cut; nothing free: the fixed labeling
    unary1, tables = random_problem(rng, 8, 9)
    fixed = rng.random((8, 9)) < 0.5
    assert np.array_equal(minimize_band(unary1, tables, np.ones((8, 9), bool), fixed), minimize(unary1, tables)[0])
    assert np.array_equal(minimize_band(unary1, tables, np.zeros((8, 9), bool), fixed), fixed)


def test_fuse_full_matches_fusion2(tmp_path):
    """Integer counts: fuse_full is fuse2's linear/paper model, with the band it is the --band model;
    outside the band the SOD label; the json loader checks the model."""
    rng = np.random.default_rng(1)
    diff = 0
    for _ in range(10):
        inp = random_input(rng, 30, 40)
        inp["sod"][:] = 0
        inp["sod"][8:22, 10:30] = rng.integers(128, 256, (14, 20))
        inp["sod"][12:18, 14:26] = 0
        x = prepare(inp)
        th = random_params(rng, "linear")
        counts = np.stack(inp["sam"]).astype(np.float32).sum(0)
        ref = fuse2(x, "linear", "paper", "binary", th) > 0
        assert np.allclose(linear_unary(inp["sod"], counts, th), __import__("sodsam.fusion2", fromlist=["unary1"]).unary1("linear", th, x))
        assert np.array_equal(fuse_full(inp["img"], inp["sod"], counts, th, 1000.0), ref)
        for r in (1.5, 3.0, 6.0):
            info = {}
            got = fuse_full(inp["img"], inp["sod"], counts, th, r, info=info)
            band = fuse2(x, "linear", "paper", "binary", {**th, "band_w": r}, band=True) > 0
            assert np.array_equal(got, band), r
            far = x["dist"] * max(x["s"].shape) > r
            assert np.array_equal(got[far], x["sod_mask"][far])
            assert info["n_free"] == (~far).sum() and 0 < info["band_frac"] < 1
            diff += (got != ref).sum()
    assert diff > 0
    # fractional counts (overlapping crops) are accepted and give a valid labeling
    counts = rng.uniform(0, 3, (30, 40)).astype(np.float32)
    got = fuse_full(inp["img"], inp["sod"], counts, th, 4.0)
    assert got.dtype == bool and got.shape == (30, 40)
    # an empty SOD map: no band, the (empty) SOD mask
    empty = np.zeros((30, 40), np.uint8)
    assert not fuse_full(inp["img"], empty, counts, th, 4.0).any()
    p = tmp_path / "res.json"
    p.write_text(json.dumps({"model": ["linear", "paper", "binary"], "params": {**th, "extra": 1.0}}))
    assert load_linear_params(p) == {k: th[k] for k in ("a", "b", "d", "sigma", "lam", "c_neg")}
    p.write_text(json.dumps({"model": ["table", "paper", "binary"], "params": th}))
    try:
        load_linear_params(p)
        raise AssertionError("accepted a table model")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile
    test_band_cut_equals_clamped_cut()
    test_fuse_full_matches_fusion2(Path(tempfile.mkdtemp()))
    print("all tests passed")
