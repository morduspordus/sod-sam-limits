"""Sanity checks of the toolbox replica and of the tuning objective."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import metrics
from sodsam.pipeline import binary_f


def random_gt(rng, h=60, w=80):
    gt = np.zeros((h, w), bool)
    y, x = rng.integers(5, h // 2), rng.integers(5, w // 2)
    gt[y:y + rng.integers(5, h // 2), x:x + rng.integers(5, w // 2)] = True
    return gt


def test_perfect_prediction():
    gt = random_gt(np.random.default_rng(0))
    s = metrics.image_scores(gt.astype(np.uint8) * 255, gt)
    assert s["mae"] == 0 and abs(s["s"] - 1) < 1e-6 and abs(s["e"] - 1) < 1e-3
    assert abs(s["f_curve"].max() - 1) < 1e-4


def test_binary_f_equals_fmax_on_binary_maps():
    rng = np.random.default_rng(1)
    gts = [random_gt(rng) for _ in range(30)]
    preds = [np.logical_xor(g, rng.random(g.shape) < 0.1) for g in gts]
    preds[0][:] = False   # an empty prediction
    fmax = metrics.aggregate([metrics.image_scores(p.astype(np.uint8) * 255, g)
                              for p, g in zip(preds, gts)])["Fmax"]
    mean_f = np.mean([binary_f(p, g) for p, g in zip(preds, gts)])
    assert abs(fmax - mean_f) < 1e-9, (fmax, mean_f)


if __name__ == "__main__":
    test_perfect_prediction()
    test_binary_f_equals_fmax_on_binary_maps()
    print("all tests passed")
