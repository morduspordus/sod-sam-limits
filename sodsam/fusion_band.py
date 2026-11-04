"""The paper's graph cut at full resolution, restricted to a band around the SOD boundary.

minimize_band solves the energy of fusion.minimize with the labels outside a boolean free mask fixed:
only the free pixels become graph nodes, a pairwise term between a free pixel and a fixed neighbor is a
unary term on the free pixel, and terms between fixed pixels are constants. The labeling equals the one
of fusion.minimize with the fixed pixels clamped by +-BIG terminal capacities (fusion.clamp_band), but a
6K x 4K image needs a graph of the band's pixels only, not of its 24M pixels.

fuse_full is the linear model of fusion2 (unary "linear", pairwise "paper", output "binary") on a full
resolution image with a float SAM count map n in [0, 3] (crops.combined_counts: the crops' mean counts
where covered, the upsampled 1K counts elsewhere):
    u(1) = d - a L(s) - b (n - 1.5),   L = logit of the max-normalized SOD map clipped to [1/255, 254/255]
with the color Potts weights and the repulsive edges where n changes by >= 2 (fusion.pairwise_tables),
the labels fixed to the thresholded SOD map outside the band of radius band_r (crops.boundary_band) and
solved exactly inside. The parameters are the ones of a tune2.py result json (load_linear_params).
"""
import json

import maxflow
import numpy as np

from .crops import boundary_band
from .fusion import color_sqdiff, color_weights, pairwise_tables, sod_binary
from .fusion2 import _logit

LINEAR_KEYS = ("a", "b", "d", "sigma", "lam", "c_neg")
FRAC_UNIFORM = 0.01   # floor of the color weight, as fusion2.fuse2


def load_linear_params(path):
    """The parameters {a, b, d, sigma, lam, c_neg} of a tune2.py result json of the linear/paper model."""
    with open(path) as f:
        res = json.load(f)
    model = res.get("model")
    if model is not None and (model[0].split("+")[0] != "linear" or model[1] != "paper"):
        raise ValueError(f"{path}: model {model} is not linear/paper")
    params = res["params"] if "params" in res else res
    missing = [k for k in LINEAR_KEYS if k not in params]
    if missing:
        raise ValueError(f"{path}: parameters {missing} missing")
    return {k: float(params[k]) for k in LINEAR_KEYS}


def minimize_band(unary1, tables, free, fixed):
    """Exact minimization of the energy of fusion.minimize (unary1 = u_p(1), tables = fusion.pairwise_tables)
    over the pixels of the boolean mask free, the others holding the labels of the boolean array fixed.
    Returns the full boolean labeling (fixed outside free, solved inside)."""
    h, w = unary1.shape
    free = np.asarray(free, bool)
    fixed = np.asarray(fixed, bool)
    x = fixed.copy()
    n_free = int(free.sum())
    if n_free == 0:
        return x
    u = unary1.astype(np.float64).copy()
    (br, cr), (bd, cd) = tables
    idx = np.full((h, w), -1, np.int64)
    idx[free] = np.arange(n_free)

    edges = []
    for b, c, sl_p, sl_q in ((br, cr, np.s_[:, :-1], np.s_[:, 1:]), (bd, cd, np.s_[:-1], np.s_[1:])):
        fp, fq = free[sl_p], free[sl_q]
        both = fp & fq
        p_only, q_only = fp & ~fq, fq & ~fp
        up, uq = u[sl_p], u[sl_q]      # views: in-place updates reach u
        # b (1 - x_p) x_q + c x_p (1 - x_q) with x_q known: a unary on p (plus a constant); same for q
        # (evaluated on the masked pairs only: the temporaries are O(band), not O(image))
        xq = fixed[sl_q][p_only].astype(np.float64)
        up[p_only] += c[p_only] * (1 - xq) - b[p_only] * xq
        xp = fixed[sl_p][q_only].astype(np.float64)
        uq[q_only] += b[q_only] * (1 - xp) - c[q_only] * xp
        # free-free: c x_p - c x_q + (b + c)(1 - x_p) x_q, one directed edge p -> q of capacity b + c
        cb = c[both]
        up[both] += cb
        uq[both] -= cb
        edges.append((idx[sl_p][both], idx[sl_q][both], b[both] + cb))

    g = maxflow.Graph[float](n_free, 2 * n_free)
    nodes = g.add_nodes(n_free)
    uf = u[free]
    # label 1 = sink segment: x_p = 1 cuts the source edge, so it carries u_p(1) > 0 (as fusion.minimize)
    g.add_grid_tedges(nodes, np.maximum(uf, 0), np.maximum(-uf, 0))
    for i, j, cap in edges:
        if len(i):
            g.add_edges(i, j, np.maximum(cap, 0), np.zeros(len(i)))
    g.maxflow()
    x[free] = g.get_grid_segments(nodes)
    return x


def linear_unary(sod_map, counts, params):
    """u(1) = d - a L(s) - b (n - 1.5) of the max-normalized SOD map and a float count map n in [0, 3]."""
    s = sod_map.astype(np.float64)
    s = s / (s.max() + 1e-13)
    return params["d"] - params["a"] * _logit(s) - params["b"] * (counts.astype(np.float64) - 1.5)


def _bbox(mask, margin, shape):
    ys, xs = np.nonzero(mask.any(1)), np.nonzero(mask.any(0))
    y0, y1 = max(0, ys[0][0] - margin), min(shape[0], ys[0][-1] + 1 + margin)
    x0, x1 = max(0, xs[0][0] - margin), min(shape[1], xs[0][-1] + 1 + margin)
    return y0, y1, x0, x1


def fuse_full(img, sod_map, counts, params, band_r, threshold=127.0, info=None):
    """Boolean object mask of the linear/paper model at full resolution (see the module docstring):
    the labels of the thresholded SOD map outside the band of radius band_r, the exact cut inside.
    counts: float map in [0, 3] (crops.combined_counts or crops.global_counts).
    info: an optional dict that receives the number of free pixels (n_free) and their fraction (band_frac)."""
    sod_mask = sod_binary(sod_map, threshold)
    free = boundary_band(sod_map, band_r, threshold)
    n_free = int(free.sum())
    if info is not None:
        info.update(n_free=n_free, band_frac=n_free / free.size)
    if n_free == 0:
        return sod_mask
    # the problem lives in the bounding box of the band plus one fixed pixel of margin; the rest is constant
    y0, y1, x0, x1 = _bbox(free, 1, free.shape)
    box = np.s_[y0:y1, x0:x1]
    u = linear_unary(sod_map[box], counts[box], params)
    w_right, w_down = color_weights(None, params["sigma"], params["lam"], FRAC_UNIFORM, color_sqdiff(img[box]))
    # repulsive edges where the (fractional) count changes by >= 2, as fusion2.fuse2 with k = 3
    tables = pairwise_tables(counts[box].astype(np.float64), w_right, w_down, params["c_neg"], "full", "paper")
    x = sod_mask.copy()
    x[box] = minimize_band(u, tables, free[box], sod_mask[box])
    return x
