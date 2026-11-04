"""Checks of the label-cost fusion (sodsam/labelcost.py) on a synthetic image."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import gaussian_filter

from sodsam import labelcost
from sodsam.fusion import color_weights


def scene():
    h, w = 40, 60
    part1 = np.zeros((h, w), bool)
    part1[8:30, 10:28] = True
    part2 = np.zeros((h, w), bool)
    part2[12:26, 26:40] = True
    gt = part1 | part2
    distractor = np.zeros((h, w), bool)
    distractor[25:38, 46:58] = True
    dup = part1.copy()
    dup[8, 10:12] = False                                 # near duplicate of part1
    cands = np.stack([part1, part2, gt, distractor, dup])
    iou = np.array([0.9, 0.85, 0.8, 0.95, 0.9])
    s = gaussian_filter(gt.astype(np.float64), 2)
    s = s / s.max()
    L = np.log(np.clip(s, 1 / 255, 254 / 255) / (1 - np.clip(s, 1 / 255, 254 / 255)))
    img = np.where(gt[..., None], 200, 50).astype(np.uint8).repeat(3, -1)
    img[distractor] = 120
    w_r, w_d = color_weights(img, 20.0, 5.0, 0.01)
    return gt, cands, iou, L, w_r, w_d


def iou_of(a, b):
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def test_fit_recovers_the_object():
    gt, cands, iou, L, w_r, w_d = scene()
    u = 0.0 - 10.0 * L                                    # negative inside the blurred object
    costs = 50.0 * (1 - iou) + 20.0
    labels, obj = labelcost.expand_fit(u, cands, costs, w_r, w_d)
    assert iou_of(obj, gt) >= 0.9, iou_of(obj, gt)
    assert not (labels == 4).any(), "distractor chosen"


def test_energy_decreases_over_moves():
    gt, cands, iou, L, w_r, w_d = scene()
    u = 0.0 - 10.0 * L
    costs = 50.0 * (1 - iou) + 20.0
    active = cands.any(0)
    idx = np.full(gt.shape, -1, np.int64)
    idx[active] = np.arange(int(active.sum()))
    labels = np.zeros(gt.shape, np.int64)
    e = labelcost.energy(labels, u, cands, costs, w_r, w_d)
    n_moves = 0
    for _ in range(2):
        for alpha in range(len(cands) + 1):
            new = labelcost._expand(labels, alpha, u, cands, costs, w_r, w_d, active, idx)
            if new is None:
                continue
            e_new = labelcost.energy(new, u, cands, costs, w_r, w_d)
            assert e_new <= e + 1e-6, (alpha, e, e_new)     # an expansion move never raises the energy
            if e_new < e - 1e-6:
                labels, e, n_moves = new, e_new, n_moves + 1
    assert n_moves >= 1
    assert np.isfinite(e)


def test_huge_label_cost_gives_background():
    gt, cands, iou, L, w_r, w_d = scene()
    u = 0.0 - 10.0 * L
    labels, obj = labelcost.expand_fit(u, cands, np.full(len(cands), 1e6), w_r, w_d)
    assert not obj.any()


def test_candidates_prune_and_dedupe():
    gt, cands, iou, L, w_r, w_d = scene()
    sod_mask = gt
    prompted = cands[:1]
    auto = cands[1:]
    M, I = labelcost.candidates(sod_mask, auto, iou[1:], prompted, iou[:1], min_inside=0.3)
    assert len(M) == 3, len(M)                              # part1 (prompted), part2, union; distractor out, dup out
    packed = np.packbits(M, axis=-1)
    back = np.unpackbits(packed, axis=-1, count=M.shape[2]).astype(bool).reshape(M.shape)
    assert (back == M).all()


if __name__ == "__main__":
    test_fit_recovers_the_object()
    test_energy_decreases_over_moves()
    test_huge_label_cost_gives_background()
    test_candidates_prune_and_dedupe()
    print("all tests passed")
