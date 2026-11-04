"""Per-pixel AGREEMENT maps (fraction of people who include a pixel) predicted by a parametric graph cut.

Sweep of the unary offset d of the paper's linear model (all other parameters fixed):
    u_t(1) = d_t - a L(s) - b (n - 1.5),   d_0 < d_1 < ... < d_{T-1}  (ds sorted ascending).
The pairwise terms are submodular and the unary decreases uniformly as d decreases, so the minimizers
are nested: the foreground grows as d decreases. The flip index of a pixel is the largest t at which it
is foreground (-1 = never foreground), so that pixel p is foreground at d_t iff flip(p) >= t, and the
flip offset d*(p) = d_{flip(p)}. Nestedness is enforced explicitly (cumulative OR from high d to low d);
the pixels where the raw solutions were not nested are counted (ties of equal-energy minimizers or a bug).

Calibration: d*(p) -> [0, 1] by isotonic regression (pool-adjacent-violators, fit_isotonic / apply_isotonic),
fitted on one dataset and applied to another (scripts/agreement_eval.py).

Agreement maps (agreement_maps): PASCAL-S, mean over the 12 subjects of their answer; MSOD, mean over the
image's subjects (3 to 7 per image) of the share of the subject's sessions whose union of chosen objects
contains the pixel (every subject has equal weight, whatever the number of sessions).

Default offset grid (default_offsets): the unary without the offset, -a L(s) - b (n - 1.5), lies in [-U, U]
with U = a logit(254/255) + 1.5 b (|L| <= 5.54, n in 0..3), so up to the pairwise terms every pixel is
background for d > U and foreground for d < -U. The grid covers [-R, R], R = U + lam (one Potts weight of
margin for the pairwise terms), with 33 equally spaced values, step 2R / 31, shifted so that the paper's d0 is
exactly on it. samba (a 48.0, b 53.8, lam 40.3, d0 182.6): U = 347, R = 387, step 25.0;
selfreformer (a 7.5, b 170.0, lam 35.5, d0 -53.9): U = 296, R = 332, step 21.4.
Measured on the first 20 PASCAL-S and MSOD images (fine grid, step 50): samba mean foreground fraction 1.000 at
d = -350 and 0.0000 (max 0.0006) at d = 350; selfreformer 0.9999 (min 0.9996) at -300, 0.0001 (max 0.0005) at
300; so the grid ends give empty and full masks. Most pixels flip near the two ends (saturated saliency) and
near d = +-(a 5.54 - 1.5 b); the interior of the grid orders the uncertain pixels.
"""
import time

import maxflow
import numpy as np

from .ideas import _layer, _tables

N_OFFSETS = 33
L_MAX = float(np.log(254.0))          # logit(254/255)


def default_offsets(th, n=N_OFFSETS):
    """n ascending, equally spaced offsets covering [-R, R], R = a L_MAX + 1.5 b + lam, with th["d"] on the grid."""
    R = th["a"] * L_MAX + 1.5 * th["b"] + th["lam"]
    step = 2 * R / (n - 2)
    d0 = float(th["d"])
    j0 = int(np.ceil((d0 + R) / step))
    j0 = min(max(j0, 0), n - 1)
    return d0 + step * (np.arange(n) - j0)


def _base(x, th):
    """The linear unary without the offset: -a L(s) - b (n - 1.5), n on the 3-mask scale (fusion2 linear)."""
    n3 = x["n"] * 3.0 / x["k"]
    return -th["a"] * x["L"] - th["b"] * (n3 - 1.5), n3


def sweep_offsets(x, th, ds=None, reuse=True, return_raw=False):
    """Flip-index map of the offset sweep.

    x = fusion2.prepare(...) record, th = the paper's linear params (a, b, d, sigma, lam, c_neg),
    ds = offsets (sorted ascending here; default default_offsets(th)).
    reuse=True builds the graph once at the highest offset and lowers d by adding the step to every sink
    capacity, then re-solves with the search trees reused (Kohli & Torr); reuse=False rebuilds each cut with
    fusion.minimize's construction. Returns (flip int16 map, info) with info = {"ds", "nonnested_pixels",
    "nonnested_pairs", "time"}; with return_raw also the T x H x W raw solutions.
    """
    t0 = time.perf_counter()
    ds = np.sort(np.asarray(default_offsets(th) if ds is None else ds, dtype=np.float64))
    base, n3 = _base(x, th)
    tables = _tables(x, n3, th)                     # built once per image
    h, w = base.shape
    T = len(ds)
    raw = np.zeros((T, h, w), bool)
    if reuse:
        g = maxflow.Graph[float](h * w, 2 * h * w)
        nodes = g.add_grid_nodes((h, w))
        _layer(g, nodes, ds[-1] + base, tables)
        g.maxflow()
        raw[-1] = g.get_grid_segments(nodes)
        zero = np.zeros((h, w))
        for t in range(T - 2, -1, -1):
            # u(1) = cap_source - cap_sink: lowering d by delta adds delta to every sink capacity
            g.add_grid_tedges(nodes, zero, np.full((h, w), ds[t + 1] - ds[t]))
            g.mark_grid_nodes(nodes)
            g.maxflow(reuse_trees=True)
            raw[t] = g.get_grid_segments(nodes)
    else:
        from .fusion import minimize
        for t in range(T):
            raw[t], _ = minimize(ds[t] + base, tables)
    # enforce nestedness: foreground at d_t if foreground at any d_s >= d_t
    nested = np.logical_or.accumulate(raw[::-1], axis=0)[::-1]
    bad = nested != raw
    flip = np.where(nested.any(0), T - 1 - np.argmax(nested[::-1], axis=0), -1).astype(np.int16)
    info = {"ds": ds, "nonnested_pixels": int(bad.any(0).sum()), "nonnested_pairs": int(bad.sum()),
            "time": time.perf_counter() - t0}
    return (flip, info, raw) if return_raw else (flip, info)


def flip_offset(flip, ds):
    """d*(p) = d_{flip(p)}; never-foreground pixels get one step below the grid."""
    ds = np.asarray(ds, dtype=np.float64)
    below = ds[0] - (ds[1] - ds[0] if len(ds) > 1 else 1.0)
    return np.where(flip >= 0, ds[np.maximum(flip, 0)], below)


# ---------------------------------------------------------------- agreement maps

def agreement_from_answers(answers, shape):
    """Mean over subjects of the share of the subject's answers containing the pixel; answers =
    [(subject, bool mask)], masks of another shape are dropped. Returns (map float64 in [0, 1], n_subjects)."""
    per = {}
    for sj, a in answers:
        if a.shape == tuple(shape):
            per.setdefault(sj, []).append(a.astype(np.float64))
    if not per:
        return None, 0
    return np.mean([np.mean(v, axis=0) for v in per.values()], axis=0), len(per)


def _scripts_on_path():
    """The per-subject loaders live in scripts/multi_answer_eval.py and scripts/annotator_support.py."""
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _answers(ds, id_, shape):
    _scripts_on_path()
    if ds == "PASCAL-S":
        from multi_answer_eval import pascal_answers
        return pascal_answers(id_)
    if ds == "MSOD":
        from annotator_support import sod_sessions
        return [(s["sj"], s["union"]) for s in sod_sessions(id_, shape)]
    raise ValueError(f"no per-subject answers for {ds}")


def agreement_maps(ds, id_, shape=None):
    """Graded agreement map in [0, 1] of an image from the per-subject answers (None if there are none)."""
    if shape is None:
        from . import data
        shape = data.load_gt(ds, id_).shape
    A, _ = agreement_from_answers(_answers(ds, id_, shape), shape)
    return A


def has_answers(ds, id_):
    if ds == "MSOD":
        _scripts_on_path()
        from annotator_support import SOD_RAW
        return (SOD_RAW / f"SO{id_}.mat").exists()
    return True


# ---------------------------------------------------------------- isotonic calibration

def _wmedian(hist, ts):
    """Lower weighted median of a histogram over the sorted target values ts."""
    c = np.cumsum(hist)
    return ts[int(np.searchsorted(c, 0.5 * c[-1]))]


def _fit_isotonic_l1(v, t, w):
    """Non-decreasing least-absolute-deviations fit: PAV with blocks holding a histogram over the distinct
    target values; a block's value is its weighted median (PAV is exact for any separable convex loss)."""
    xs, inv = np.unique(v, return_inverse=True)
    ts, tinv = np.unique(t, return_inverse=True)
    H = np.zeros((len(xs), len(ts)))
    np.add.at(H, (inv, tinv), w)
    keep = H.sum(1) > 0
    xs, H = xs[keep], H[keep]
    hist, med, cnt = [], [], []
    for h in H:
        hist.append(h.copy())
        med.append(_wmedian(h, ts))
        cnt.append(1)
        while len(med) > 1 and med[-2] > med[-1]:
            h2, c2 = hist.pop(), cnt.pop()
            med.pop()
            hist[-1] += h2
            cnt[-1] += c2
            med[-1] = _wmedian(hist[-1], ts)
    return {"x": xs, "y": np.repeat(med, cnt)}


def fit_isotonic(values, targets, weights=None, loss="l2"):
    """Non-decreasing fit of targets on values by pool-adjacent-violators on the distinct values: least squares
    (loss "l2", block value = weighted mean) or least absolute deviations ("l1", weighted median; the right
    calibration for a map scored by MAE). Returns {"x": distinct values, "y": fitted values} for apply_isotonic."""
    v = np.asarray(values, dtype=np.float64).ravel()
    t = np.asarray(targets, dtype=np.float64).ravel()
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if loss == "l1":
        return _fit_isotonic_l1(v, t, w)
    if loss != "l2":
        raise ValueError(loss)
    xs, inv = np.unique(v, return_inverse=True)
    W = np.bincount(inv, weights=w, minlength=len(xs))
    keep = W > 0
    xs, Y = xs[keep], (np.bincount(inv, weights=w * t, minlength=len(keep)) / np.where(W > 0, W, 1))[keep]
    W = W[keep]
    # PAV: blocks of (mean, weight, count)
    mean, wt, cnt = [], [], []
    for y, ww in zip(Y, W):
        mean.append(y)
        wt.append(ww)
        cnt.append(1)
        while len(mean) > 1 and mean[-2] > mean[-1]:
            m2, w2, c2 = mean.pop(), wt.pop(), cnt.pop()
            wsum = wt[-1] + w2
            mean[-1] = (mean[-1] * wt[-1] + m2 * w2) / wsum
            wt[-1] = wsum
            cnt[-1] += c2
    fitted = np.repeat(mean, cnt)
    return {"x": xs, "y": fitted}


def apply_isotonic(model, values):
    """Piecewise-linear interpolation between the fitted distinct values, constant beyond them."""
    return np.interp(np.asarray(values, dtype=np.float64), model["x"], model["y"])


apply = apply_isotonic
