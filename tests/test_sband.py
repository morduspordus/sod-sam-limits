"""The paper_sb pairwise of fusion2 (SAM boundary band): rho = 0 is exactly paper, the boundary distance
matches brute force, the band scales exactly the Potts weights, and the cut is the exact minimizer."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import fusion
from sodsam.fusion2 import band_factors, fuse2, param_ranges, prepare, sam_boundary_dist, unary1


def random_input(rng, h=24, w=32, blobs=True):
    """Three SAM masks that share a blob (so the count jumps by >= 2 on part of its boundary)."""
    yy, xx = np.mgrid[:h, :w]
    sam = []
    for _ in range(3):
        if blobs:
            cy, cx, r = h / 2 + rng.normal(0, 2), w / 2 + rng.normal(0, 2), min(h, w) / 3 + rng.normal(0, 1.5)
            sam.append((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
        else:
            sam.append(rng.random((h, w)) < 0.5)
    return {"img": rng.integers(0, 256, (h, w, 3)).astype(np.uint8),
            "sod": rng.integers(0, 256, (h, w)).astype(np.uint8),
            "sam": sam, "gt": rng.random((h, w)) < 0.4}


def paper_th(rng):
    return {"a": rng.uniform(0, 60), "b": rng.uniform(0, 200), "d": rng.uniform(-100, 100),
            "sigma": rng.uniform(5, 50), "lam": rng.uniform(1, 50), "c_neg": rng.uniform(-50, 0)}


def test_rho_zero_is_paper():
    rng = np.random.default_rng(0)
    for trial in range(30):
        x = prepare(random_input(rng, blobs=trial % 2 == 0))
        th = paper_th(rng)
        ref = fuse2(x, "linear", "paper", "binary", th)
        for r in (0.5, 3.0, 16.0):
            assert np.array_equal(fuse2(x, "linear", "paper_sb", "binary", {**th, "rho": 0.0, "r_sb": r}), ref)
    assert list(param_ranges("linear", "paper_sb"))[-2:] == ["rho", "r_sb"]
    assert param_ranges("linear", "paper_sb")["rho"][0] == 0.0


def _brute_dist(n3):
    h, w = n3.shape
    pts = [(y, x) for y in range(h) for x in range(w)
           if any(0 <= y + dy < h and 0 <= x + dx < w and (n3[y, x] >= 1.5) != (n3[y + dy, x + dx] >= 1.5)
                  for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)))]
    if not pts:
        return None
    p = np.array(pts, float)
    yy, xx = np.mgrid[:h, :w]
    return np.sqrt(((yy[..., None] - p[:, 0]) ** 2 + (xx[..., None] - p[:, 1]) ** 2).min(-1))


def test_boundary_distance_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(30):
        h, w = int(rng.integers(3, 20)), int(rng.integers(3, 20))
        n3 = rng.integers(0, 4, (h, w)).astype(float) if trial % 3 else np.zeros((h, w))
        if trial % 3 == 1:     # sparse jumps
            n3 = np.where(rng.random((h, w)) < 0.05, 3.0, 0.0)
        d = sam_boundary_dist(n3)
        b = _brute_dist(n3)
        if b is None:
            assert d is None
        else:
            assert d.shape == (h, w) and np.allclose(d, b, atol=1e-3), (trial, np.abs(d - b).max())
    # no majority anywhere, or majority everywhere: no boundary
    assert sam_boundary_dist(np.tile([0, 1, 1, 0, 1.0], (5, 1))) is None
    assert sam_boundary_dist(np.full((4, 5), 2.0)) is None
    # masks offset by one pixel each (count steps by 1): the boundary is the majority edge (columns 2|3)
    d = sam_boundary_dist(np.tile([0, 0, 1, 2, 3, 3.0], (4, 1)))
    assert d is not None and (d[:, 2:4] == 0).all() and (d[:, [0, 1, 4, 5]] > 0).all()


def test_boundary_contains_repulsive_pairs():
    """Every pair whose count changes by >= 2 (6 masks: steps of 0.5 on the 3-mask scale) is on B."""
    rng = np.random.default_rng(4)
    for k in (3, 6):
        for _ in range(30):
            n3 = rng.integers(0, k + 1, (12, 15)) * 3.0 / k
            d = sam_boundary_dist(n3)
            jr = np.abs(n3[:, 1:] - n3[:, :-1]) >= 2
            jd = np.abs(n3[1:] - n3[:-1]) >= 2
            if jr.any() or jd.any():
                assert d is not None
                assert (d[:, 1:][jr] == 0).all() and (d[:, :-1][jr] == 0).all()
                assert (d[1:][jd] == 0).all() and (d[:-1][jd] == 0).all()


def test_weights_scaled_exactly():
    """paper_sb tables = paper tables with the Potts weights times the band factors (repulsive pairs as paper)."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = prepare(random_input(rng))
        th = {**paper_th(rng), "rho": rng.uniform(0, 1), "r_sb": rng.uniform(0.5, 16)}
        n3 = x["n"] * 3.0 / x["k"]
        d = sam_boundary_dist(n3)
        assert d is not None
        fr, fd = band_factors(d, th["rho"], th["r_sb"])
        assert (fr >= 1 - th["rho"] - 1e-12).all() and (fr <= 1).all() and (fd >= 1 - th["rho"] - 1e-12).all()
        # pairs touching B get exactly 1 - rho
        jr = np.abs(n3[:, 1:] - n3[:, :-1]) >= 2
        assert np.allclose(fr[jr], 1 - th["rho"])
        w = fusion.color_weights(x["img"], th["sigma"], th["lam"], 0.01)
        ref = fusion.pairwise_tables(n3, w[0] * fr, w[1] * fd, th["c_neg"], "full")
        lab, _ = fusion.minimize(unary1("linear", th, x), ref)
        assert np.array_equal(fuse2(x, "linear", "paper_sb", "binary", th) > 0, lab)


def test_exact_minimizer_brute_force():
    rng = np.random.default_rng(3)
    for trial in range(40):
        h, w = (3, 4) if trial % 2 else (4, 3)
        inp = random_input(rng, h, w, blobs=False)
        inp["sam"] = [rng.random((h, w)) < 0.5 for _ in range(3)]
        x = prepare(inp)
        th = {**paper_th(rng), "rho": rng.uniform(0, 1), "r_sb": rng.uniform(0.5, 4)}
        th["d"] = rng.uniform(-60, 60)
        out = fuse2(x, "linear", "paper_sb", "binary", th) > 0
        n3 = x["n"] * 3.0 / x["k"]
        d = sam_boundary_dist(n3)
        wr, wd = fusion.color_weights(x["img"], th["sigma"], th["lam"], 0.01)
        if d is not None:
            fr, fd = band_factors(d, th["rho"], th["r_sb"])
            wr, wd = wr * fr, wd * fd
        tables = fusion.pairwise_tables(n3, wr, wd, th["c_neg"], "full")
        u = unary1("linear", th, x)
        best = min(fusion.energy(np.array(bits).reshape(h, w), u, tables)
                   for bits in itertools.product((0, 1), repeat=h * w))
        assert abs(fusion.energy(out.astype(int), u, tables) - best) < 1e-6, trial


def test_band_lowers_cost_near_sam_boundary_only():
    """rho = 1 with a small r: a pair far from B keeps its weight; a pair on B pays nothing."""
    n3 = np.zeros((20, 20))
    n3[5:15, 5:15] = 3
    d = sam_boundary_dist(n3)
    fr, fd = band_factors(d, 1.0, 1.0)
    assert fr[10, 4] == 0.0 and fr[10, 14] == 0.0          # pairs across the square's edge
    assert fr[10, 3] == 0.0 and fd[3, 10] == 0.0           # one pixel on B: D = min over the pair = 0
    assert abs(fr[10, 2] - (1 - np.exp(-0.5))) < 1e-6      # pixels at 2 and 1 px from B: D = 1
    assert fr[10, 9] > 0.99 and fr[0, 0] > 0.99              # centre and far corner untouched


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
