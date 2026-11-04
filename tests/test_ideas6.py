"""The +thin and +sel modifiers of fusion2: off at their lower bounds (exactly the base model), the thin
region matches a brute-force morphology, the unary / Potts weights change exactly as documented, the
selected SAM mask matches brute force, and the cut is the exact minimizer."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import fusion
from sodsam.fusion2 import fuse2, param_ranges, prepare, sel_region, thin_mask, thin_region, unary1


def random_input(rng, h=24, w=32, blobs=True):
    yy, xx = np.mgrid[:h, :w]
    sam = []
    for _ in range(3):
        if blobs:
            cy, cx, r = h / 2 + rng.normal(0, 2), w / 2 + rng.normal(0, 2), min(h, w) / 3 + rng.normal(0, 1.5)
            sam.append((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2)
        else:
            sam.append(rng.random((h, w)) < 0.5)
    sod = rng.integers(0, 256, (h, w)).astype(np.uint8)
    if blobs:   # a blob with a thin bar, so the thin region is not empty
        sod = np.where(sam[0], 220, 30).astype(np.uint8)
        sod[h // 2, :] = 230
    return {"img": rng.integers(0, 256, (h, w, 3)).astype(np.uint8), "sod": sod, "sam": sam,
            "scores": rng.uniform(0.3, 1.0, 3).tolist(), "gt": rng.random((h, w)) < 0.4}


def paper_th(rng):
    return {"a": rng.uniform(0, 60), "b": rng.uniform(0, 200), "d": rng.uniform(-100, 100),
            "sigma": rng.uniform(5, 50), "lam": rng.uniform(1, 50), "c_neg": rng.uniform(-50, 0)}


def test_off_is_base():
    rng = np.random.default_rng(0)
    for trial in range(30):
        x = prepare(random_input(rng, blobs=trial % 2 == 0))
        th = paper_th(rng)
        ref = fuse2(x, "linear", "paper", "binary", th)
        for r in (1.0, 2.6, 6.0):
            assert np.array_equal(fuse2(x, "linear+thin", "paper", "binary", {**th, "g_b": 0.0, "g_p": 0.0, "r_t": r}), ref)
        for tau, beta in ((0.0, -1.0), (0.5, 0.3), (1.0, 1.0)):
            assert np.array_equal(fuse2(x, "linear+sel", "paper", "binary", {**th, "e": 0.0, "tau_s": tau, "beta_s": beta}), ref)
    for mod, first in (("thin", ("g_b", "g_p")), ("sel", ("e",))):
        r = param_ranges(f"linear+{mod}", "paper")
        assert all(r[k][0] == 0 for k in first)


def _brute_thin(mask, r):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(bool)
    offs = [(dy - r, dx - r) for dy in range(2 * r + 1) for dx in range(2 * r + 1) if se[dy, dx]]
    h, w = mask.shape

    def erode(m):   # outside the image counts as foreground (OpenCV's default border)
        return np.array([[all(m[y + dy, x + dx] for dy, dx in offs if 0 <= y + dy < h and 0 <= x + dx < w)
                          for x in range(w)] for y in range(h)])

    def dilate(m):  # outside the image counts as background
        return np.array([[any(m[y - dy, x - dx] for dy, dx in offs if 0 <= y - dy < h and 0 <= x - dx < w)
                          for x in range(w)] for y in range(h)])

    thin = mask & ~dilate(erode(mask))
    return dilate(thin)


def test_thin_mask_brute_force():
    rng = np.random.default_rng(1)
    for trial in range(25):
        h, w = int(rng.integers(4, 18)), int(rng.integers(4, 18))
        m = rng.random((h, w)) < (0.3 + 0.5 * rng.random())
        for r in (1, 2, 3):
            assert np.array_equal(thin_mask(m, r), _brute_thin(m, r)), (trial, r)
    # a 1-pixel bar on a 9x9 square: the bar (and its surroundings) is thin, the square's core is not
    m = np.zeros((20, 30), bool)
    m[5:14, 3:12] = True
    m[9, 12:28] = True
    T = thin_mask(m, 2)
    assert T[9, 20] and T[8, 20] and not T[9, 7] and not T[2, 20]
    assert not thin_mask(np.zeros((5, 5), bool), 2).any() and not thin_mask(m, 0).any()


def test_thin_unary_and_weights_exact():
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = prepare(random_input(rng))
        th = {**paper_th(rng), "g_b": rng.uniform(0, 1), "g_p": rng.uniform(0, 1), "r_t": rng.uniform(1, 6)}
        T = thin_mask(x["sod_mask"], int(round(th["r_t"]))).astype(float)
        assert T.any() and not T.all()
        assert np.array_equal(thin_region(x, th["r_t"]), T)
        sam_term = 3 * th["b"] * (x["n"] / x["k"] - 0.5)
        u_ref = th["d"] - th["a"] * x["L"] - (1 - th["g_b"] * T) * sam_term
        assert np.allclose(unary1("linear+thin", th, x), u_ref)
        wr, wd = fusion.color_weights(x["img"], th["sigma"], th["lam"], 0.01)
        wr = wr * (1 - th["g_p"] * np.maximum(T[:, 1:], T[:, :-1]))
        wd = wd * (1 - th["g_p"] * np.maximum(T[1:], T[:-1]))
        assert (wr >= 0).all() and (wd >= 0).all()
        tables = fusion.pairwise_tables(x["n"] * 3.0 / x["k"], wr, wd, th["c_neg"], "full")
        lab, _ = fusion.minimize(u_ref, tables)
        assert np.array_equal(fuse2(x, "linear+thin", "paper", "binary", th) > 0, lab)


def test_sel_brute_force():
    rng = np.random.default_rng(3)
    n_none = 0
    for _ in range(60):
        inp = random_input(rng, blobs=False)
        x = prepare(inp)
        tau, beta = rng.uniform(0, 1), rng.uniform(-1, 1)
        s, c = x["s"], np.array(inp["scores"])
        g = [sum(s[p] - tau for p in zip(*np.nonzero(m))) / s.sum() + beta * c[k] for k, m in enumerate(inp["sam"])]
        U = sel_region(x["sel"], tau, beta)
        if max(g) <= 0:
            assert U is None
            n_none += 1
        else:
            assert np.array_equal(U, inp["sam"][int(np.argmax(g))].astype(float))
        th = {**paper_th(rng), "e": rng.uniform(0, 300), "tau_s": tau, "beta_s": beta}
        u0 = unary1("linear", th, x)
        u = unary1("linear+sel", th, x)
        assert np.allclose(u, u0 if U is None else u0 - th["e"] * (2 * U - 1))
    assert 0 < n_none < 60      # both branches exercised


def test_exact_minimizer_brute_force():
    rng = np.random.default_rng(4)
    for trial in range(40):
        h, w = (3, 4) if trial % 2 else (4, 3)
        inp = random_input(rng, h, w, blobs=False)
        inp["sod"] = rng.integers(0, 256, (h, w)).astype(np.uint8)
        x = prepare(inp)
        unary = "linear+thin+sel"
        th = {**paper_th(rng), "g_b": rng.uniform(0, 1), "g_p": rng.uniform(0, 1), "r_t": 1.0,
              "e": rng.uniform(0, 100), "tau_s": rng.uniform(0, 1), "beta_s": rng.uniform(-1, 1)}
        th["d"] = rng.uniform(-60, 60)
        out = fuse2(x, unary, "paper", "binary", th) > 0
        T = thin_mask(x["sod_mask"], 1).astype(float)
        wr, wd = fusion.color_weights(x["img"], th["sigma"], th["lam"], 0.01)
        wr = wr * (1 - th["g_p"] * np.maximum(T[:, 1:], T[:, :-1]))
        wd = wd * (1 - th["g_p"] * np.maximum(T[1:], T[:-1]))
        tables = fusion.pairwise_tables(x["n"] * 3.0 / x["k"], wr, wd, th["c_neg"], "full")
        u = unary1(unary, th, x)
        best = min(fusion.energy(np.array(bits).reshape(h, w), u, tables)
                   for bits in itertools.product((0, 1), repeat=h * w))
        assert abs(fusion.energy(out.astype(int), u, tables) - best) < 1e-6, trial


def test_thin_needs_linear():
    rng = np.random.default_rng(5)
    x = prepare(random_input(rng))
    th = {f"f{s}{n}": 0.0 for s in range(2) for n in range(4)}
    th.update(a=1.0, g_b=0.5, g_p=0.0, r_t=2.0)
    try:
        unary1("table_lin+thin", th, x)
    except ValueError:
        return
    raise AssertionError("+thin accepted a non-linear base")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
