"""Checks of the multi-layer graph constructions in sodsam/ideas.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import ideas


def _x(h=20, w=30, seed=0):
    rng = np.random.default_rng(seed)
    gt = np.zeros((h, w), bool)
    gt[5:15, 8:22] = True
    s = np.clip(gt * 0.8 + rng.random((h, w)) * 0.2, 1 / 255, 254 / 255)
    part = np.zeros((h, w), bool)
    part[5:15, 8:15] = True
    n = gt.astype(float) + part + (gt & (rng.random((h, w)) < 0.9))        # 3 masks: whole, part, noisy whole
    img = (np.repeat(s[..., None], 3, 2) * 255).astype(np.uint8)
    from sodsam.fusion import color_sqdiff
    return {"L": np.log(s / (1 - s)), "n": n, "k": 3, "sqdiff": color_sqdiff(img), "gt": gt,
            "sod_mask": s >= 0.5}, part


def test_nested_containment_and_coupling():
    x, part = _x()
    th = {"a": 5, "b_in": 60, "b_out": 10, "d_in": 60, "d_out": 10, "mu": 0, "sigma": 20, "lam": 5, "c_neg": -5}
    inner, outer = ideas.nested_cut(x, th)
    assert not (inner & ~outer).any(), "inner must be inside outer"
    assert outer.sum() > inner.sum() > 0
    # a large coupling closes the gap: the two answers coincide
    th["mu"] = 1000
    inner, outer = ideas.nested_cut(x, th)
    assert (inner == outer).all()
    # containment binds: a pixel that inner wants (strongly) but outer does not is resolved consistently
    x2 = dict(x)
    th2 = dict(th, mu=0, d_in=-500, d_out=500)                          # inner wants everything, outer nothing
    inner, outer = ideas.nested_cut(x2, th2)
    assert not (inner & ~outer).any()


def _ordinal_th(**kw):
    th = {"a": 5, "b": 40, "d": 10, "delta": 30, "mu": 3, "sigma": 20, "lam": 5, "c_neg": -5}
    th.update(kw)
    return th


def test_ordinal_nested():
    x, _ = _x()
    masks = ideas.ordinal_cut(x, _ordinal_th(), K=3)
    assert len(masks) == 3
    for outer, inner in zip(masks[:-1], masks[1:]):
        assert not (inner & ~outer).any(), "layer j+1 must be inside layer j"
    assert masks[0].sum() > masks[2].sum() > 0


def test_ordinal_k2_is_nested():
    x, _ = _x()
    th = _ordinal_th()
    outer, inner = ideas.ordinal_cut(x, th, K=2)
    th2 = {"a": th["a"], "b_in": th["b"], "b_out": th["b"], "d_in": th["d"] + th["delta"], "d_out": th["d"],
           "mu": th["mu"], "sigma": th["sigma"], "lam": th["lam"], "c_neg": th["c_neg"]}
    n_inner, n_outer = ideas.nested_cut(x, th2)
    assert (inner == n_inner).all() and (outer == n_outer).all()


def test_ordinal_delta0_and_k1_equal_single_cut():
    from sodsam.fusion import minimize
    from sodsam.fusion2 import fuse2
    x, _ = _x()
    th = _ordinal_th(delta=0)
    n3 = x["n"] * 3.0 / x["k"]
    u = th["d"] - th["a"] * x["L"] - th["b"] * (n3 - 1.5)
    single, _ = minimize(u, ideas._tables(x, n3, th))
    assert (fuse2(x, "linear", "paper", "binary", th) > 0).tolist() == single.tolist()
    for m in ideas.ordinal_cut(x, th, K=3):
        assert (m == single).all()
    (m,) = ideas.ordinal_cut(x, _ordinal_th(), K=1)
    assert (m == single).all()


def test_scales_majority():
    x, part = _x()
    x["ns"] = [x["n"], x["n"], np.zeros_like(x["n"])]
    th = {"a": 5, "b": 40, "d": 20, "mu": 0, "sigma": 20, "lam": 5, "c_neg": -5}
    m, segs = ideas.scales_cut(x, th)
    assert len(segs) == 3 and (m == (np.mean(segs, 0) >= 0.5)).all()
    assert segs[2].sum() <= segs[0].sum()
    th["mu"] = 1000                                                      # strong coupling: all layers equal
    m, segs = ideas.scales_cut(x, th)
    assert all((s == segs[0]).all() for s in segs)


def test_decision():
    x, part = _x()
    sam = [x["gt"], part, x["gt"] | (np.arange(30)[None, :] < 3)]
    rec = ideas.decision_prepare(x, x["gt"], sam, [0.9, 0.5, 0.4])
    assert rec["F"].shape == (5 + 2 * 6, 5)
    m = ideas.decision(rec, {"w_sod": 0.0, "w_gc": 1.0, "beta": 1.0})
    assert (m == x["gt"]).all()                                          # only the GC answer counts
    m = ideas.decision(rec, {"w_sod": 0.0, "w_gc": 0.0, "beta": 20.0})  # dominated by the best-scored mask
    assert ideas.binary_f(m, x["gt"]) > 0.95


if __name__ == "__main__":
    test_nested_containment_and_coupling()
    test_ordinal_nested()
    test_ordinal_k2_is_nested()
    test_ordinal_delta0_and_k1_equal_single_cut()
    test_scales_majority()
    test_decision()
    print("all tests passed")
