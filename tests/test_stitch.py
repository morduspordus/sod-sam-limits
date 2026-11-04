"""Stitching of overlapping windows (sodsam.stitch) on synthetic crops."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam.fusion import BIG, color_weights, minimize, pairwise_tables, sod_binary
from sodsam.stitch import P_CLIP, Stitcher, gaussian_weight, seam_name, stitch_all


def setup(rng, h=60, w=80):
    boxes = [(5, 5, 35, 45), (20, 30, 50, 70), (10, 40, 40, 80)]
    probs = [rng.random((b[2] - b[0], b[3] - b[1])).astype(np.float32) for b in boxes]
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    sod = rng.integers(0, 256, (h, w)).astype(np.uint8)
    return boxes, probs, img, sod


def reference(shape, boxes, probs):
    """Full-size per-crop stacks (only for the test)."""
    h, w = shape
    n = len(boxes)
    P = np.zeros((n, h, w))
    W = np.zeros((n, h, w))
    C = np.zeros((n, h, w), bool)
    for k, ((y0, x0, y1, x1), p) in enumerate(zip(boxes, probs)):
        P[k, y0:y1, x0:x1] = p
        W[k, y0:y1, x0:x1] = gaussian_weight(y1 - y0, x1 - x0)
        C[k, y0:y1, x0:x1] = True
    return P, W, C


def test_gaussian_weight():
    g = gaussian_weight(1024, 1024)
    assert g.max() == 1 and g.dtype == np.float32 and np.allclose(g, g.T) and np.allclose(g, g[::-1, ::-1])
    # sigma = size / 4: two sigmas from the centre on one axis
    assert abs(g[511, 0] / g[511, 511] - np.exp(-0.5 * (511.5 / 256) ** 2)) < 1e-3
    assert gaussian_weight(600, 900).shape == (600, 900)


def test_avg_gauss_and_seam():
    rng = np.random.default_rng(0)
    for _ in range(5):
        boxes, probs, img, sod = setup(rng)
        shape = sod.shape
        st = Stitcher(shape, boxes)
        for b, p in zip(boxes, probs):
            st.add(b, p)
        info = {}
        out = stitch_all(img, sod, st, lams=(0.0, 8.0), sigmas=(10.0,), r_s=2, info=info)
        P, W, C = reference(shape, boxes, probs)
        cov = C.any(0)
        n = C.sum(0)
        sodm = sod_binary(sod)
        avg = np.where(cov, P.sum(0) >= 0.5 * np.maximum(n, 1), sodm)
        gauss = np.where(cov, (W * P).sum(0) >= 0.5 * W.sum(0), sodm)
        assert np.array_equal(out["avg"], avg)
        assert np.array_equal(out["gauss"], gauss)
        lab = P >= 0.5
        dis = (n >= 2) & ((lab & C).sum(0) > 0) & ((lab & C).sum(0) < n)
        assert info["n_disagree"] == dis.sum() > 0 and info["n_seam"] >= dis.sum()
        pc = np.clip(P, P_CLIP, 1 - P_CLIP)
        u = -np.divide((W * np.log(pc / (1 - pc))).sum(0), W.sum(0), out=np.zeros(shape), where=W.sum(0) > 0)
        # lam = 0: the seam takes the sign of the weighted logit, everything else the gauss labels
        s0 = out[seam_name(0.0, 10.0)]
        seam = s0 != gauss
        assert np.array_equal(s0[dis], (u < 0)[dis])
        assert seam.sum() > 0 and (cov | ~seam).all()
        # lam = 8: equals the full cut with every non-seam pixel clamped to the gauss label
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        free = (cv2.dilate(dis.astype(np.uint8), k) > 0) & cov
        w_r, w_d = color_weights(img, 10.0, 8.0, 0.01)
        tables = pairwise_tables(np.zeros(shape), w_r, w_d, 0.0, "potts")
        ref, _ = minimize(np.where(free, u, np.where(gauss, -BIG, BIG)), tables)
        assert np.array_equal(out[seam_name(8.0, 10.0)], ref)


def test_no_crops_and_agreement():
    rng = np.random.default_rng(1)
    _, _, img, sod = setup(rng)
    out = stitch_all(img, sod, Stitcher(sod.shape, []), lams=(2.0,), sigmas=(10.0,))
    assert all(np.array_equal(v, sod_binary(sod)) for v in out.values())
    # agreeing crops: no seam, the seam rows are the gauss row
    boxes = [(0, 0, 30, 40), (10, 20, 40, 60)]
    st = Stitcher(sod.shape, boxes)
    for b in boxes:
        st.add(b, np.full((b[2] - b[0], b[3] - b[1]), 0.9, np.float32))
    info = {}
    out = stitch_all(img, sod, st, lams=(2.0,), sigmas=(10.0,), info=info)
    assert info["n_seam"] == 0 and np.array_equal(out[seam_name(2.0, 10.0)], out["gauss"])
    assert out["gauss"][:40, :60][st.covered()].all()


if __name__ == "__main__":
    test_gaussian_weight()
    test_avg_gauss_and_seam()
    test_no_crops_and_agreement()
    print("all tests passed")
