"""SOD metrics, replicating PySODMetrics (py_sod_metrics: Fmeasure curve, MAE, Smeasure, Emeasure adp),
the evaluation code of most recent SOD papers; checked against it by scripts/verify_metrics.py.

Conventions copied from that code:
  * GT foreground = first channel > 128.
  * Every map is first divided by 255 and min-max normalized to [0, 1] (a constant map is kept as map / 255).
  * F^max: the normalized map times 255, truncated to uint8; precision and recall per image at the 256
    thresholds (pixel >= t; precision 0 when nothing is predicted); per-image F (beta^2 = 0.3, 0 when
    P * R = 0) is averaged over all images, an empty GT giving F = 0; F^max is the max over t.
  * MAE: of the normalized map.
  * S_alpha (Fan et al. 2017) of the normalized map.
  * E (Fan et al. 2018, adaptive): the normalized map binarized at >= min(2 * mean, 1), then the
    enhanced-alignment score with denominator H*W - 1.
Before 2026-11-01 the module replicated the MATLAB toolbox of the first submission instead, which differs
in: max (not min-max) normalization with rounding for F, F averaged over non-empty GTs only, MAE of
map / 255, and E binarized at > (strictly) the threshold, which empties binary maps covering half the image.
"""
import numpy as np

EPS = np.finfo(np.float64).eps


def _round_half_away(x):
    return np.floor(x + 0.5)


def _minmax(pred):
    """PySODMetrics' prepare_data: map / 255, then min-max normalized unless constant."""
    x = pred / 255
    if x.max() != x.min():
        x = (x - x.min()) / (x.max() - x.min())
    return x


def f_curve(pred, gt):
    """Per-image F curve over thresholds 255..0 (index 0 = threshold 255); zeros for an empty GT."""
    x = (_minmax(pred) * 255).astype(np.uint8)
    t = np.bincount(x[gt], minlength=256)[::-1].cumsum()
    n = t + np.bincount(x[~gt], minlength=256)[::-1].cumsum()
    n[n == 0] = 1
    precision = t / n
    recall = t / max(np.count_nonzero(gt), 1)
    num = 1.3 * precision * recall
    return num / np.where(num == 0, 1, 0.3 * precision + recall)


def mae(pred, gt):
    return np.abs(_minmax(pred) - gt).mean()


def _object(pred, gt):
    v = pred[gt]
    if v.size == 0:
        return 0.0
    x = v.mean()
    sigma = v.std(ddof=1) if v.size > 1 else 0.0
    return 2.0 * x / (x * x + 1.0 + sigma + EPS)


def _s_object(pred, gt):
    fg = np.where(gt, pred, 0.0)
    bg = np.where(gt, 0.0, 1.0 - pred)
    u = gt.mean()
    return u * _object(fg, gt) + (1 - u) * _object(bg, ~gt)


def _ssim(pred, gt):
    n = pred.size
    if n == 0:
        return 0.0
    g = gt.astype(np.float64)
    x, y = pred.mean(), g.mean()
    sx2 = ((pred - x) ** 2).sum() / (n - 1 + EPS)
    sy2 = ((g - y) ** 2).sum() / (n - 1 + EPS)
    sxy = ((pred - x) * (g - y)).sum() / (n - 1 + EPS)
    alpha = 4 * x * y * sxy
    beta = (x * x + y * y) * (sx2 + sy2)
    if alpha != 0:
        return alpha / (beta + EPS)
    return 1.0 if beta == 0 else 0.0


def _s_region(pred, gt):
    h, w = gt.shape
    total = gt.sum()
    if total == 0:
        X, Y = int(_round_half_away(w / 2)), int(_round_half_away(h / 2))
    else:
        X = int(_round_half_away((gt.sum(0) * np.arange(1, w + 1)).sum() / total))
        Y = int(_round_half_away((gt.sum(1) * np.arange(1, h + 1)).sum() / total))
    area = w * h
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1.0 - w1 - w2 - w3
    q = (_ssim(pred[:Y, :X], gt[:Y, :X]), _ssim(pred[:Y, X:], gt[:Y, X:]),
         _ssim(pred[Y:, :X], gt[Y:, :X]), _ssim(pred[Y:, X:], gt[Y:, X:]))
    return w1 * q[0] + w2 * q[1] + w3 * q[2] + w4 * q[3]


def s_measure(pred, gt):
    p = _minmax(pred)
    y = gt.mean()
    if y == 0:
        return 1.0 - p.mean()
    if y == 1:
        return p.mean()
    q = 0.5 * _s_object(p, gt) + 0.5 * _s_region(p, gt)
    return max(q, 0.0)


def e_measure(pred, gt):
    p = _minmax(pred)
    thr = min(2 * p.mean(), 1.0)
    fm = (p >= thr).astype(np.float64)
    g = gt.astype(np.float64)
    if g.sum() == 0:
        enhanced = 1.0 - fm
    elif (1 - g).sum() == 0:
        enhanced = fm
    else:
        af, ag = fm - fm.mean(), g - g.mean()
        align = 2 * ag * af / (ag * ag + af * af + EPS)
        enhanced = (align + 1) ** 2 / 4
    return enhanced.sum() / (gt.size - 1 + EPS)


def image_scores(pred, gt):
    """All per-image quantities; pred is a uint8 map with the same size as gt."""
    if pred.shape != gt.shape:
        raise ValueError(f"prediction {pred.shape} and GT {gt.shape} differ in size")
    return {
        "f_curve": f_curve(pred, gt),
        "mae": mae(pred, gt),
        "s": s_measure(pred, gt),
        "e": e_measure(pred, gt),
    }


def aggregate(scores):
    """Dataset-level metrics from a list of image_scores() results."""
    curves = [s["f_curve"] for s in scores if s["f_curve"] is not None]
    return {
        "Fmax": float(np.mean(curves, axis=0).max()),
        "MAE": float(np.mean([s["mae"] for s in scores])),
        "E": float(np.mean([s["e"] for s in scores])),
        "S": float(np.mean([s["s"] for s in scores])),
        "n": len(scores),
    }


def format_row(name, m):
    return f"{name:<40s} F={m['Fmax']:.3f} MAE={m['MAE']:.3f} E={m['E']:.3f} S={m['S']:.3f} (n={m['n']})"
