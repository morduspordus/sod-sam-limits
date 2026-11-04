"""Learned log-odds unary: table fit / lookup on synthetic data, feature shapes, fuse2 with unary logodds."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import unary_learn as ul
from sodsam.fusion2 import fuse2, param_ranges, prepare


def random_input(rng, h=24, w=32):
    return {"img": rng.integers(0, 256, (h, w, 3)).astype(np.uint8),
            "sod": rng.integers(0, 256, (h, w)).astype(np.uint8),
            "sam": [rng.random((h, w)) < 0.5 for _ in range(3)],
            "scores": rng.random(3).tolist(), "gt": rng.random((h, w)) < 0.4}


def test_pattern_and_bins():
    """The 8 mask combinations map to distinct patterns 0..7; SOD and distance bins have the edges of
    the docstring; cells cover 0..N_CELLS - 1 without collisions."""
    rng = np.random.default_rng(0)
    inp = random_input(rng, 4, 2)
    seen = set()
    for bits in range(8):
        inp["sam"] = [np.full((4, 2), bool(bits >> k & 1)) for k in range(3)]
        p = ul.features(inp)["pattern"]
        assert (p == p[0, 0]).all() and 0 <= p[0, 0] < 8
        seen.add(int(p[0, 0]))
    assert seen == set(range(8))
    f = {"s": np.array([[0.0, 1 / 16 - 1e-6, 1 / 16, 0.5, 1.0]]), "pattern": np.array([[7, 0, 0, 0, 0]]),
         "d_min": np.array([[1.0, 2.0, 6.0, 16.0, np.inf]])}
    c = ul.cell_index(f)
    assert c.tolist() == [[(7 * 16 + 0) * 4 + 0, 1, 1 * 4 + 2, 8 * 4 + 3, 15 * 4 + 3]]
    assert ul.N_CELLS == 512 and list(ul.D_EDGES[:-1]) == [0, 2, 6, 16] and np.isinf(ul.D_EDGES[-1])
    # bijection: every (pattern, sbin, dbin) triple gives its own cell
    pat, sb, db = np.meshgrid(np.arange(8), np.arange(16), np.arange(4), indexing="ij")
    f = {"s": (sb + 0.5) / 16, "pattern": pat, "d_min": np.array([1.0, 3.0, 10.0, 100.0])[db]}
    assert sorted(ul.cell_index(f).ravel().tolist()) == list(range(512))


def test_table_fit_and_lookup():
    """A cell whose pixels are all GT = 1 gets a large positive log-odds, an all-0 cell a negative one,
    an unseen cell 0; save / load round-trips; the clip holds."""
    rng = np.random.default_rng(1)
    h, w = 40, 50
    inp = random_input(rng, h, w)
    inp["sam"] = [np.ones((h, w), bool)] * 3                    # pattern 7 everywhere, no mask boundary
    inp["sod"] = np.where(np.arange(w) < 25, 255, 0).astype(np.uint8)[None].repeat(h, 0)
    inp["gt"] = inp["sod"] > 0                                   # the GT is the SOD mask
    f = ul.features(inp)
    n1, n0 = ul.table_counts(f, inp["gt"])
    assert n1.sum() == inp["gt"].sum() and n0.sum() == (~inp["gt"]).sum()
    lo = ul.table_logodds(n1, n0)
    c = ul.cell_index(f)
    far_in = c[20, 2]                                             # s = 1, far from the SOD boundary
    assert n0[far_in] == 0 and n1[far_in] > 100 and lo[far_in] > 4
    assert lo[c[20, 47]] < -4 and lo[(0 * 16 + 5) * 4 + 1] == 0
    assert np.array_equal(ul.table_predict(lo, f) > 0, inp["gt"])
    assert ul.table_logodds(np.array([10 ** 9]), np.array([0])).max() == ul.LO_CLIP
    with tempfile.TemporaryDirectory() as d:
        ul.save_table(Path(d) / "t.npz", n1, n0, {"note": "test"})
        with np.load(Path(d) / "t.npz") as z:
            assert len(z["s_edges"]) == 17 and z["s_edges"][-1] == 1.0 and str(z["layout"]).startswith("cell")
        assert np.array_equal(ul.load_table(Path(d) / "t.npz"), lo)


def test_feature_shapes_and_distances():
    rng = np.random.default_rng(2)
    for _ in range(5):
        h, w = rng.integers(10, 40, 2)
        inp = random_input(rng, h, w)
        f = ul.features(inp)
        for name in ul.FEATURES:
            assert np.isscalar(f[name]) or f[name].shape == (h, w), name
        X = ul.feature_matrix(f)
        assert X.shape == (h * w, len(ul.FEATURES)) and X.dtype == np.float32 and np.isfinite(X).all()
        idx = ul.sample_pixels(f, rng, 50)
        assert len(idx) <= 50 and np.array_equal(ul.feature_matrix(f, idx), X[idx])
        c = ul.cell_index(f)
        assert c.shape == (h, w) and c.min() >= 0 and c.max() < ul.N_CELLS
    m = np.zeros((20, 30), bool)
    m[5:15, 10:20] = True
    d = ul.signed_dist(m)
    assert d[10, 15] == -5 and d[5, 15] == -1 and d[4, 15] == 1 and np.isclose(d[0, 0], np.hypot(5, 10))
    assert np.isinf(ul.signed_dist(np.ones((3, 3), bool))).all() and ul.signed_dist(np.ones((3, 3), bool))[0, 0] < 0


def test_fuse2_logodds():
    """lo = +8 everywhere gives the all-1 mask for a > 0 (d inside its range), lo = -8 the all-0 mask;
    a random lo without a pairwise term is the sign of d - a lo; missing lo is an error."""
    rng = np.random.default_rng(3)
    r = param_ranges("logodds", "paper")
    assert set(r) == {"a", "d", "sigma", "lam", "c_neg"} and r["a"] == (0, 300) and r["d"] == (-300, 300)
    for _ in range(5):
        inp = random_input(rng)
        th = {"a": rng.uniform(40, 300), "d": rng.uniform(-300, 300), "sigma": 20.0, "lam": 10.0, "c_neg": -15.0}
        for pw in ("none", "paper"):
            x = prepare({**inp, "lo": np.full((24, 32), 8.0, np.float32)})
            assert (fuse2(x, "logodds", pw, "binary", th) == 255).all(), pw
            x = prepare({**inp, "lo": np.full((24, 32), -8.0, np.float32)})
            assert (fuse2(x, "logodds", pw, "binary", th) == 0).all(), pw
        lo = rng.uniform(-8, 8, (24, 32)).astype(np.float32)
        x = prepare({**inp, "lo": lo})
        assert np.array_equal(fuse2(x, "logodds", "none", "binary", th) > 0, th["d"] - th["a"] * lo < 0)
    try:
        fuse2(prepare(inp), "logodds", "none", "binary", th)
        raise AssertionError("logodds accepted an image without lo")
    except ValueError:
        pass


if __name__ == "__main__":
    test_pattern_and_bins()
    test_table_fit_and_lookup()
    test_feature_shapes_and_distances()
    test_fuse2_logodds()
    print("all tests passed")
