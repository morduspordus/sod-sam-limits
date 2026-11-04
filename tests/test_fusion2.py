"""fusion2 with more than three SAM masks: duplicating the masks must not change the result, and the
table unary must match fusion.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import fusion
from sodsam.fusion2 import as_table_lin, fuse2, param_ranges, prepare, tree_select


def random_input(rng, h=24, w=32):
    return {"img": rng.integers(0, 256, (h, w, 3)).astype(np.uint8),
            "sod": rng.integers(0, 256, (h, w)).astype(np.uint8),
            "sam": [rng.random((h, w)) < 0.5 for _ in range(3)],
            "probs": [rng.integers(0, 256, (h, w)).astype(np.uint8) for _ in range(3)],
            "sam_neg": [rng.random((h, w)) < 0.5 for _ in range(3)],
            "sod2": rng.integers(0, 256, (h, w)).astype(np.uint8),
            "scores": rng.random(3).tolist(), "gt": rng.random((h, w)) < 0.4}


def doubled(inp):
    return {**inp, "sam": inp["sam"] * 2, "probs": inp["probs"] * 2, "scores": inp["scores"] * 2,
            "sam_neg": inp["sam_neg"] * 2}


def random_params(rng, unary, pairwise="paper"):
    return {k: rng.uniform(*r) for k, r in param_ranges(unary, pairwise).items()}


def test_duplicated_masks_give_same_result():
    rng = np.random.default_rng(0)
    th = {"a": 3.0, "b": 40.0, "d": 5.0, "t": 0.3, "sigma": 20.0, "lam": 10.0, "c_neg": -15.0,
          "v": 30.0, "kappa": 40.0, "a_far": 5.0, "d_far": 10.0, "tau": 0.4, "beta1": 0.1,
          "beta2": -0.1, "beta3": 0.05, "g00": 100.0, "g01": -50.0, "g10": 20.0, "g11": -200.0,
          "a2": 2.0, "g": 3.0}
    for _ in range(20):
        inp = random_input(rng)
        x3, x6 = prepare(inp), prepare(doubled(inp))
        for unary in ("linear", "linear_conf", "linear_gate", "linear_prob", "linear+veto+band", "tree",
                      "duo", "duo_plain"):
            for pw in ("none", "potts", "paper"):
                a, b = fuse2(x3, unary, pw, "binary", th), fuse2(x6, unary, pw, "binary", th)
                assert np.array_equal(a, b), (unary, pw)


def random_feat(rng, h=24, w=32):
    return (rng.random((h, w - 1)).astype(np.float32), rng.random((h - 1, w)).astype(np.float32))


def test_table_matches_fusion_and_rejects_six_masks():
    rng = np.random.default_rng(1)
    p = fusion.FusionParams(unary=rng.uniform(-50, 50, (2, 4)).tolist(), sigma=15, sigma_f=0.3, lam=12,
                            c_neg=-20)
    th = {f"f{a}{b}": p.unary[a][b] for a in range(2) for b in range(4)}
    th.update(sigma=p.sigma, sigma_f=p.sigma_f, lam=p.lam, c_neg=p.c_neg, band_w=4.0)
    for _ in range(10):
        inp = random_input(rng)
        inp["feat"] = random_feat(rng)
        for variant, pw in (("unary", "none"), ("potts", "potts"), ("full", "paper"), ("full", "feat"),
                            ("full", "both")):
            for band in (None, 4.0):
                ref = fusion.fuse(inp["img"], inp["sod"], inp["sam"], p, variant, pw, feat=inp["feat"], band=band)
                got = fuse2(prepare(inp), "table", pw, "binary", th, band is not None) > 0
                assert np.array_equal(got, ref), (variant, pw, band)
    try:
        fuse2(prepare(doubled(inp)), "table", "none", "binary", th)
        raise AssertionError("table accepted 6 masks")
    except ValueError:
        pass


def test_table_lin_reduces_to_table():
    """table_lin with a = 0 is the table; a > 0 changes the labels."""
    rng = np.random.default_rng(2)
    th = {k: rng.uniform(*r) for k, r in param_ranges("table_lin", "paper").items()}
    diff = 0
    for _ in range(10):
        x = prepare(random_input(rng))
        for pw in ("none", "paper"):
            a0 = fuse2(x, "table_lin", pw, "binary", {**th, "a": 0.0})
            assert np.array_equal(a0, fuse2(x, "table", pw, "binary", th)), pw
            diff += (fuse2(x, "table_lin", pw, "binary", {**th, "a": 50.0}) != a0).sum()
    assert diff > 0


def test_as_table_lin_is_identical():
    """table and linear models converted to table_lin give the same labels, and lie in its ranges."""
    rng = np.random.default_rng(3)
    lin_r = param_ranges("table_lin", "paper")
    for unary in ("table", "linear"):
        th = {k: rng.uniform(*r) for k, r in param_ranges(unary, "paper").items()}
        conv = as_table_lin(unary, th)
        assert set(conv) == set(lin_r)
        assert all(lin_r[k][0] <= v <= lin_r[k][1] for k, v in conv.items()), unary
        for _ in range(10):
            x = prepare(random_input(rng))
            for pw in ("none", "paper"):
                assert np.array_equal(fuse2(x, unary, pw, "binary", th),
                                      fuse2(x, "table_lin", pw, "binary", conv)), (unary, pw)


def test_modifiers_off_equal_base():
    """v = 0 (veto) and kappa = 0 (band) give the base model; switched on they change labels."""
    rng = np.random.default_rng(4)
    diff = 0
    for _ in range(10):
        x = prepare(random_input(rng))
        for base in ("table_lin", "linear"):
            th = random_params(rng, base + "+veto+band")
            off = {**th, "v": 0.0, "kappa": 0.0}
            for pw in ("none", "paper"):
                ref = fuse2(x, base, pw, "binary", th)
                assert np.array_equal(fuse2(x, base + "+veto+band", pw, "binary", off), ref), (base, pw)
                diff += (fuse2(x, base + "+veto+band", pw, "binary", th) != ref).sum()
    assert diff > 0


def test_dist_and_band():
    """dist is the distance to the SOD boundary in units of max(H, W); far from the boundary a large
    kappa leaves the SOD-only unary."""
    rng = np.random.default_rng(5)
    inp = random_input(rng, 40, 50)
    inp["sod"][:] = 0
    inp["sod"][10:30, 10:40] = 255
    x = prepare(inp)
    assert x["k"] == 3
    assert np.isclose(x["dist"][20, 25], 10 / 50)          # center: 10 rows to the boundary
    assert np.isclose(x["dist"][10, 25], 1 / 50)           # first row inside
    assert np.isclose(x["dist"][0, 0], np.hypot(10, 10) / 50)
    th = random_params(rng, "linear+band")
    th.update(kappa=500.0, d_far=7.0, a_far=2.0)
    u = __import__("sodsam.fusion2", fromlist=["unary1"]).unary1("linear+band", th, x)
    sod_only = th["d_far"] - th["a_far"] * x["L"]
    assert np.allclose(u[0, 0], sod_only[0, 0])
    assert not np.allclose(u[10, 25], sod_only[10, 25])


def test_feat_pairwise_and_band():
    """Zero feature distances make "feat" the paper model of a constant image and "both" the paper
    model; a band wider than the image is no band; a band of width 0.5 returns the SOD mask; feat / both
    without feature distances is an error."""
    rng = np.random.default_rng(9)
    diff = 0
    for _ in range(10):
        inp = random_input(rng)
        th = random_params(rng, "linear", "both")
        th["band_w"] = rng.uniform(2, 8)
        flat = {**inp, "img": np.full_like(inp["img"], 100)}
        zero = (np.zeros((24, 31), np.float32), np.zeros((23, 32), np.float32))
        x = prepare({**inp, "feat": zero})
        assert np.array_equal(fuse2(prepare({**flat, "feat": zero}), "linear", "feat", "binary", th),
                              fuse2(prepare(flat), "linear", "paper", "binary", th))
        assert np.array_equal(fuse2(x, "linear", "both", "binary", th), fuse2(x, "linear", "paper", "binary", th))
        xf = prepare({**inp, "feat": random_feat(rng)})
        ref = fuse2(xf, "linear", "paper", "binary", th)
        diff += (fuse2(xf, "linear", "feat", "binary", th) != ref).sum()
        assert np.array_equal(fuse2(xf, "linear", "paper", "binary", {**th, "band_w": 1000.0}, True), ref)
        sod_mask = fuse2(xf, "linear", "paper", "binary", {**th, "band_w": 0.5}, True) > 0
        assert np.array_equal(sod_mask, xf["sod_mask"])
        diff += (fuse2(xf, "linear", "paper", "binary", th, True) != ref).sum()
        try:
            fuse2(prepare(inp), "linear", "feat", "binary", th)
            raise AssertionError("feat accepted an image without feature distances")
        except ValueError:
            pass
    assert diff > 0


def test_tree_select_picks_the_right_level():
    """Nested masks: the SOD map covering the middle level makes the DP pick the middle component;
    a bonus on the outer level makes it pick the outer one; a hostile map picks nothing."""
    h, w = 40, 60
    outer = np.zeros((h, w), bool); outer[5:35, 5:55] = True
    mid = np.zeros((h, w), bool); mid[10:30, 10:40] = True
    inner = np.zeros((h, w), bool); inner[15:25, 15:25] = True
    rng = np.random.default_rng(6)
    inp = random_input(rng, h, w)
    inp["sam"] = [outer, mid, inner]
    inp["sod"] = np.where(mid, 255, 0).astype(np.uint8)
    x = prepare(inp)
    labs, levels = x["tree"]
    assert [len(l[0]) for l in levels] == [1, 1, 1]
    U = tree_select(x["tree"], 0.5, (0.0, 0.0, 0.0))
    assert np.array_equal(U, mid)
    U = tree_select(x["tree"], 0.5, (0.6, 0.0, 0.0))     # per-pixel bonus > per-pixel loss outside mid
    assert np.array_equal(U, outer)
    inp["sod"][:] = 0
    U = tree_select(prepare(inp)["tree"], 0.5, (0.0, 0.0, 0.0))
    assert not U.any()
    # two separate objects at level 1, each with its own sub-tree
    second = np.zeros((h, w), bool); second[2:8, 56:59] = True
    inp["sam"] = [outer | second, mid | second, inner]
    inp["sod"] = np.where(mid | second, 255, 0).astype(np.uint8)
    x = prepare(inp)
    assert [len(l[0]) for l in x["tree"][1]] == [2, 2, 1]
    assert np.array_equal(tree_select(x["tree"], 0.5, (0.0, 0.0, 0.0)), mid | second)


def test_duo_reduces_to_linear_and_outputs():
    """With the second model equal to the first (same map, same masks) and a2 = g = 0, duo is linear;
    the guided filter gives maps in range and pipeline knows the v2 outputs."""
    from sodsam.guided_filter import guided_filter
    rng = np.random.default_rng(7)
    for _ in range(5):
        inp = random_input(rng)
        same = {**inp, "sod2": inp["sod"], "sam": inp["sam"] * 2, "scores": None}
        th = random_params(rng, "duo")
        off = {**th, "a2": 0.0, "g": 0.0}
        for pw in ("none", "paper"):
            ref = fuse2(prepare(inp), "linear", pw, "binary", th)
            assert np.array_equal(fuse2(prepare(same), "duo", pw, "binary", off), ref), pw
            assert np.array_equal(fuse2(prepare(same), "duo_plain", pw, "binary", off), ref), pw
        soft = fuse2(prepare(inp), "linear", "paper", "soft", th)
        q = guided_filter(inp["img"], soft / 255.0)
        assert q.shape == soft.shape and q.min() >= 0 and q.max() <= 1
    from sodsam import pipeline
    assert pipeline.V2_OUTPUTS == ("binary", "soft", "soft_gf", "soft_crf")


def test_soft_output_keeps_the_cut():
    """Thresholding the soft output at 128 gives the binary output exactly, also for SOD values at 0.5;
    away from the threshold the SOD values are kept."""
    rng = np.random.default_rng(8)
    for _ in range(10):
        inp = random_input(rng)
        inp["sod"][0, :8] = [127, 128, 129, 255, 0, 126, 128, 255]
        x = prepare(inp)
        th = random_params(rng, "linear")
        for pw in ("none", "paper"):
            b = fuse2(x, "linear", pw, "binary", th) > 0
            soft = fuse2(x, "linear", pw, "soft", th)
            assert np.array_equal(soft >= 128, b), pw
            v = np.round(x["s"] * 255)
            keep = (b & (v >= 128)) | (~b & (v <= 127))
            assert np.array_equal(soft[keep], v[keep].astype(np.uint8)), pw


if __name__ == "__main__":
    test_soft_output_keeps_the_cut()
    test_feat_pairwise_and_band()
    test_duo_reduces_to_linear_and_outputs()
    test_modifiers_off_equal_base()
    test_dist_and_band()
    test_tree_select_picks_the_right_level()
    test_as_table_lin_is_identical()
    test_duplicated_masks_give_same_result()
    test_table_matches_fusion_and_rejects_six_masks()
    test_table_lin_reduces_to_table()
    print("all tests passed")
