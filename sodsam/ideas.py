"""Post-processing ideas on top of the paper's graph cut (SOD map + SAM masks, fusion2.prepare inputs).

nested    two answers, inner (part) and outer (whole), in one graph cut: two binary layers with the paper's
          unary and pairwise terms, inner subset of outer enforced by infinite edges (Delong & Boykov
          multi-region construction), and a coupling mu per pixel in the gap between the two:
            u_in(1) = d_in - a L(s) - b_in (n - 1.5),   u_out(1) = d_out - a L(s) - b_out (n - 1.5)
          9 parameters (a, b_in, b_out, d_in, d_out, mu, sigma, lam, c_neg).
scales    one binary layer per SAM run at another zoom level (whole image, margin 100%, 25%, 0%), the
          same linear unary on each layer's own count n_k, the paper's pairwise terms per layer, and a
          Potts coupling mu between consecutive layers; output = majority of the layers (7 parameters).
decision  one mask chosen for its expected F under a distribution over plausible answers: the SOD mask
          (weight w_sod), the graph cut (w_gc) and the three SAM masks (the rest, softmax(beta * score));
          the pool of candidate outputs is these answers plus the pairwise unions and intersections of the
          SOD mask and the SAM masks; the F matrix is computed once per image (3 parameters).
ordinal   K nested answers (part, object, object with context) in one cut: K binary layers x^1 >= ... >= x^K
          (x^j = [label >= j], Ishikawa / Delong-Boykov), the paper's pairwise terms on every layer and the
          linear unary u_j(1) = d + (j - 1) delta - a L(s) - b (n - 1.5), delta >= 0 (8 parameters); the
          generalization of nested (K = 2) with one b and the offsets ordered by construction.
sod_levels  baseline with the same number of answers: K thresholds on the max-normalized SOD map.
"""
import maxflow
import numpy as np

from .fusion import color_weights, pairwise_tables
from .fusion2 import fuse2

INF = 1e9
RIGHT = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
DOWN = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
RANGES = {
    "nested": {"a": (0, 100), "b_in": (0, 200), "b_out": (0, 200), "d_in": (-200, 200), "d_out": (-200, 200),
               "mu": (0, 100), "sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0)},
    "scales": {"a": (0, 100), "b": (0, 200), "d": (-200, 200), "mu": (0, 100),
               "sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0)},
    "decision": {"w_sod": (0, 1), "w_gc": (0, 1), "beta": (0, 20)},
    "ordinal": {"a": (0, 100), "b": (0, 200), "d": (-200, 200), "delta": (0, 200), "mu": (0, 100),
                "sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0)},
    "sod_levels": {f"t{j}": (0, 1) for j in (1, 2, 3)},
}
MULTI = ("ordinal", "sod_levels")       # models with K answers, tuned by the multi-answer objective
SCALE_PROMPTS = ["mask_soft", "mask_zoom_m100", "mask_zoom", "mask_zoom_m0"]   # coarse to fine


def binary_f(m, g):
    """F_beta (beta^2 = 0.3) of a binary mask; 0 without true positives (also for an empty GT)."""
    tp = np.logical_and(m, g).sum()
    if tp == 0:
        return 0.0
    p, r = tp / m.sum(), tp / g.sum()
    return 1.3 * p * r / (0.3 * p + r)


def _layer(g, nodes, u, tables):
    """t-links and grid edges of one binary layer, exactly as fusion.minimize (label 1 = sink side)."""
    h, w = u.shape
    u = u.astype(np.float64).copy()
    (br, cr), (bd, cd) = tables
    u[:, :-1] += cr
    u[:, 1:] -= cr
    u[:-1] += cd
    u[1:] -= cd
    g.add_grid_tedges(nodes, np.maximum(u, 0), np.maximum(-u, 0))
    wr = np.zeros((h, w))
    wr[:, :-1] = np.maximum(br + cr, 0)
    g.add_grid_edges(nodes, weights=wr, structure=RIGHT, symmetric=False)
    wd = np.zeros((h, w))
    wd[:-1] = np.maximum(bd + cd, 0)
    g.add_grid_edges(nodes, weights=wd, structure=DOWN, symmetric=False)


def _tables(x, n3, th):
    w = color_weights(None, th["sigma"], th["lam"], 0.01, x["sqdiff"])
    return pairwise_tables(n3, *w, th["c_neg"], "full")


def nested_cut(x, th):
    """(inner, outer) boolean masks; x = fusion2.prepare(...) (keys L, n, k, sqdiff)."""
    n3 = x["n"] * 3.0 / x["k"]
    base = -th["a"] * x["L"]
    u_in = th["d_in"] + base - th["b_in"] * (n3 - 1.5)
    u_out = th["d_out"] + base - th["b_out"] * (n3 - 1.5)
    tables = _tables(x, n3, th)
    h, w = u_in.shape
    g = maxflow.Graph[float](2 * h * w, 5 * h * w)
    ni, no = g.add_grid_nodes((h, w)), g.add_grid_nodes((h, w))
    _layer(g, ni, u_in, tables)
    _layer(g, no, u_out, tables)
    # edge outer -> inner is cut when outer = 0 and inner = 1 (forbidden); the reverse edge is the
    # coupling, paid where inner = 0 and outer = 1 (the gap between the two answers)
    g.add_edges(no.ravel(), ni.ravel(), np.full(h * w, INF), np.full(h * w, float(th["mu"])))
    g.maxflow()
    return g.get_grid_segments(ni), g.get_grid_segments(no)


def param_ranges(model, K=3):
    """RANGES of a model; sod_levels has one threshold per answer."""
    if model == "sod_levels":
        return {f"t{j}": (0, 1) for j in range(1, K + 1)}
    return RANGES[model]


def ordinal_cut(x, th, K=3):
    """K nested boolean masks, outermost first: layers x^1 >= x^2 >= ... >= x^K (x^j = [label >= j]), each
    with the paper's pairwise tables and the unary u_j(1) = d + (j - 1) delta - a L(s) - b (n - 1.5); the
    INF edge forbids x^{j+1} = 1 where x^j = 0 and the reverse edge pays mu per pixel of the gap. The
    pairwise tables are identical in every layer, so the pairwise energy summed over the layers is
    w |l_p - l_q| plus a modular (repulsive) part, convex in the label difference: the construction is
    exact (Ishikawa). K = 2 is nested_cut with b_in = b_out, d_out = d, d_in = d + delta."""
    n3 = x["n"] * 3.0 / x["k"]
    base = -th["a"] * x["L"] - th["b"] * (n3 - 1.5)
    tables = _tables(x, n3, th)
    h, w = base.shape
    g = maxflow.Graph[float](K * h * w, (2 + K) * h * w)
    nodes = [g.add_grid_nodes((h, w)) for _ in range(K)]
    for j, nd in enumerate(nodes):
        _layer(g, nd, th["d"] + j * th["delta"] + base, tables)
    for outer, inner in zip(nodes[:-1], nodes[1:]):
        g.add_edges(outer.ravel(), inner.ravel(), np.full(h * w, INF), np.full(h * w, float(th["mu"])))
    g.maxflow()
    return tuple(g.get_grid_segments(nd) for nd in nodes)


def sod_levels(x, th, K=3):
    """K level sets of the max-normalized SOD map (x["s8"], uint8), thresholds sorted, outermost first."""
    ts = sorted(th[f"t{j}"] for j in range(1, K + 1))
    return tuple(x["s8"] >= 255.0 * t for t in ts)


def scales_cut(x, th):
    """Majority mask over the layers and the per-layer masks; x has "ns" = [count map per zoom level]."""
    ns = [n * 3.0 / x["k"] for n in x["ns"]]
    h, w = ns[0].shape
    g = maxflow.Graph[float](len(ns) * h * w, (2 + len(ns)) * h * w)
    nodes = [g.add_grid_nodes((h, w)) for _ in ns]
    for nd, n3 in zip(nodes, ns):
        _layer(g, nd, th["d"] - th["a"] * x["L"] - th["b"] * (n3 - 1.5), _tables(x, n3, th))
    for a, b in zip(nodes[:-1], nodes[1:]):
        g.add_edges(a.ravel(), b.ravel(), np.full(h * w, float(th["mu"])), np.full(h * w, float(th["mu"])))
    g.maxflow()
    segs = [g.get_grid_segments(nd) for nd in nodes]
    return np.mean(segs, axis=0) >= 0.5, segs


def decision_prepare(x, gc, sam, scores):
    """Per-image record of the decision model: candidate pool, answers, F matrix (pool x answers)."""
    sod, sam = x["sod_mask"], [m > 0 for m in sam]
    answers = [sod, gc] + sam                                   # weights: w_sod, w_gc, softmax(beta score)
    pool = list(answers)
    singles = [sod] + sam
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            pool += [singles[i] & singles[j], singles[i] | singles[j]]
    F = np.array([[binary_f(c, a) for a in answers] for c in pool])
    return {"pool": pool, "F": F, "scores": np.asarray(scores, np.float64), "gt": x["gt"]}


def decision(rec, th):
    s = rec["scores"] * th["beta"]
    soft = np.exp(s - s.max())
    rest = max(1.0 - th["w_sod"] - th["w_gc"], 0.0)
    w = np.concatenate([[th["w_sod"], th["w_gc"]], rest * soft / soft.sum()])
    return rec["pool"][int(np.argmax(rec["F"] @ w))]


def gc_mask(x, gc_res):
    """The paper's graph cut (a tune2 result json loaded as a dict) as a boolean mask."""
    return fuse2(x, *gc_res["model"], gc_res["params"]) > 0
