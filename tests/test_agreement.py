"""Checks of the offset sweep, the isotonic calibration and the agreement maps (sodsam/agreement.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import agreement
from sodsam.fusion import color_sqdiff, minimize
from sodsam.ideas import _tables

TH = {"a": 30, "b": 50, "d": 40, "sigma": 20, "lam": 15, "c_neg": -10}


def _x(h=24, w=32, seed=0):
    """A random record built like tests/test_ideas.py: a noisy object, three SAM masks, a noisy image."""
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), bool)
    gt[5:18, 8:24] = True
    s = np.clip(gt * 0.6 + rng.random((h, w)) * 0.4, 1 / 255, 254 / 255)
    part = np.zeros((h, w), bool)
    part[5:18, 8:15] = True
    n = gt.astype(float) + part + (rng.random((h, w)) < 0.3)
    img = (np.clip(np.repeat(s[..., None], 3, 2) + rng.normal(0, 0.1, (h, w, 3)), 0, 1) * 255).astype(np.uint8)
    return {"L": np.log(s / (1 - s)), "n": n, "k": 3, "sqdiff": color_sqdiff(img), "gt": gt, "sod_mask": s >= 0.5}


def test_sweep_nested_random():
    for seed in range(5):
        x = _x(seed=seed)
        ds = np.linspace(-400, 400, 21)
        for reuse in (True, False):
            flip, info, raw = agreement.sweep_offsets(x, TH, ds, reuse=reuse, return_raw=True)
            assert info["nonnested_pixels"] == 0, (seed, reuse, info)
            for t in range(len(ds) - 1):        # foreground grows as d decreases
                assert not (raw[t + 1] & ~raw[t]).any()
            assert flip.min() >= -1 and flip.max() <= len(ds) - 1
        assert (flip == -1).sum() == 0 and (flip == len(ds) - 1).sum() == 0   # the grid ends: full / empty


def test_flip_matches_direct_cuts():
    x = _x(seed=3)
    ds = np.linspace(-300, 300, 15)
    flip, _ = agreement.sweep_offsets(x, TH, ds)                    # reused search trees
    n3 = x["n"] * 3.0 / x["k"]
    base = -TH["a"] * x["L"] - TH["b"] * (n3 - 1.5)
    tables = _tables(x, n3, TH)
    for t in (0, 3, 7, 10, 14):
        direct, _ = minimize(ds[t] + base, tables)
        assert (direct == (flip >= t)).all(), t
    d = agreement.flip_offset(flip, ds)
    assert (d[flip >= 0] == ds[flip[flip >= 0]]).all() and (d[flip < 0] < ds[0]).all()


def test_default_offsets():
    ds = agreement.default_offsets(TH)
    assert len(ds) == 33 and np.isclose(ds, TH["d"]).sum() == 1 and (np.diff(ds) > 0).all()
    R = TH["a"] * np.log(254) + 1.5 * TH["b"] + TH["lam"]
    assert ds[0] <= -R and ds[-1] >= R


def test_isotonic():
    rng = np.random.default_rng(0)
    v = rng.random(2000) * 10
    t = 1 / (1 + np.exp(-(v - 5))) + rng.normal(0, 0.2, v.size)
    m = agreement.fit_isotonic(v, t)
    assert (np.diff(m["y"]) >= -1e-12).all()
    y = agreement.apply_isotonic(m, np.linspace(-1, 11, 500))
    assert (np.diff(y) >= -1e-12).all()
    # a monotone target is reproduced exactly
    v = np.repeat(np.arange(10.0), 3)
    m = agreement.fit_isotonic(v, v ** 2 / 100)
    assert np.allclose(agreement.apply_isotonic(m, v), v ** 2 / 100)
    # a violating pair is pooled to its weighted mean
    m = agreement.fit_isotonic([0, 1, 2], [0.0, 0.8, 0.2], [1, 1, 3])
    assert np.allclose(m["y"], [0.0, 0.35, 0.35])


def test_agreement_from_answers():
    shape = (4, 5)
    a = np.zeros(shape, bool)
    a[0] = True
    b = np.zeros(shape, bool)
    b[:2] = True
    # subject 0: two sessions (a, b); subject 1: one session (b); subject 2: one answer of another shape (dropped)
    A, n = agreement.agreement_from_answers([(0, a), (0, b), (1, b), (2, np.ones((3, 3), bool))], shape)
    assert n == 2
    assert np.allclose(A[0], 1.0) and np.allclose(A[1], (0.5 + 1.0) / 2) and np.allclose(A[2:], 0.0)
    # one answer per subject: the plain fraction of subjects
    A, n = agreement.agreement_from_answers([(j, b if j < 3 else a) for j in range(12)], shape)
    assert n == 12 and np.allclose(A[1], 3 / 12) and np.allclose(A[0], 1.0)


def test_isotonic_l1_brute_force():
    """The L1 fit reaches the minimum weighted absolute error over all non-decreasing maps whose values are
    target values (an L1 optimum always exists among those), by enumeration on small problems."""
    import itertools
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(2, 7))
        x = np.sort(rng.integers(0, 4, n)).astype(float)
        t = rng.integers(0, 4, n) / 3
        w = rng.integers(1, 4, n).astype(float)
        m = agreement.fit_isotonic(x, t, w, loss="l1")
        assert np.all(np.diff(m["y"]) >= 0)
        cost = float((w * np.abs(agreement.apply_isotonic(m, x) - t)).sum())
        xs = np.unique(x)
        best = min(sum(w[i] * abs(dict(zip(xs, f))[x[i]] - t[i]) for i in range(n))
                   for f in itertools.product(np.unique(t), repeat=len(xs)) if all(a <= b for a, b in zip(f, f[1:])))
        assert abs(cost - best) < 1e-9, (cost, best)


if __name__ == "__main__":
    test_sweep_nested_random()
    test_flip_matches_direct_cuts()
    test_default_offsets()
    test_isotonic()
    test_agreement_from_answers()
    print("all tests passed")
