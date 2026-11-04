"""Connectivity prior for the full-resolution band cut: geodesic star convexity, solved exactly.

Thin structures (wires, frame bars) are broken by the Potts term of fusion_band.fuse_full. The prior asks
every foreground pixel to reach a seed through foreground pixels along its geodesic shortest path:
for every free pixel p with parent q (the next pixel on p's geodesic path to the nearest seed) the term
    beta * [x_p = 1, x_q = 0]
is added. It is submodular for any beta >= 0 (a pairwise term with v(1,0) = beta, v(0,1) = 0), so the cut
stays exact; beta = inf is the hard constraint (Gulshan et al.'s geodesic star convexity with the seed
set below as the star centre).

Seeds (star_parents, seed_mode):
  interior  only the fixed foreground: the thresholded SOD mask outside the band (the interior at distance
            > band_r); every foreground band pixel must connect to the object's interior. Only the fixed
            pixels 4-adjacent to the band enter the graph (the rest cannot be on a shortest path into the
            band anyway).
  strict    the interior plus the "strict" band seeds: band pixels with SOD >= T_STRICT (0.9) * max and mean
            SAM count >= C_STRICT (3: every mask of every covering crop agrees), kept only in 4-connected
            seed regions of at least A_min = max(A_MIN_PX, A_MIN_FRAC * band pixels) pixels
            (16 px, 0.01% of the band: about 150-1000 px on DIS5K), so isolated agreeing specks do not
            become roots of detached fragments.
  loose     the interior plus every band pixel with SOD >= t_s * max and count >= c_s (0.9, 2.0); too
            permissive on DIS5K (72-86% of the band's foreground are seeds), kept for comparison.
Roots are decided per 4-connected component of the band (the graph's connectivity; paths cannot leave the
band except into the interior). A component that is 4-adjacent to the interior or holds band seeds of the
mode is constrained to those roots. A component with no root (a thin part with no interior of its own,
e.g. a cable or a bench leg away from the object's body, or every component of an image whose SOD mask
lies entirely within band_r) uses its own strict band seeds (same rule and A_min) in every mode; without
those it is unconstrained (NO_PARENT: the cut is gc_full_crops there). So a free pixel is never forced to
background only for lacking a root. Per image the components are counted by kind (info["components"]).
A band seed is a root (no parent, label free); a pixel's whole path to its root, root included, is
foreground whenever the pixel is. Fixed background pixels are not graph nodes (a path through them could
never be foreground).
SEED_RULE_VERSION 3 = this per-component rule (1: loose seeds only, 2: modes with a global fallback).

Geodesic parents: scipy.sparse.csgraph.dijkstra(min_only=True, return_predecessors=True, indices=seeds) on
the 4-grid of the band pixels plus the adjacent fixed seeds (not of the band's bounding box: the pixels
outside the band are fixed, so they need no parent, and the graph is O(band) instead of O(bbox)). Edge
cost between 4-neighbours p, q: 1 + gamma * (1 - (s_p + s_q) / 2), s = SOD map / max(SOD map).
Memory: ~ 4 directed entries per node (12 bytes each in CSR) plus the COO arrays during construction
(~16 bytes per entry) and Dijkstra's per-node state (distance, predecessor, source, heap node):
roughly 150-200 bytes per band pixel at peak.

minimize_band_star: fusion_band.minimize_band (same folding of the fixed neighbours, same label convention:
label 1 = sink segment) plus the star edges. With beta = HARD the number of violated hard constraints is
counted (info["violations"], 0 in theory) instead of asserted, so a long run is not stopped.
"""
import time

import maxflow
import numpy as np
from scipy.ndimage import label as cc_label
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .crops import boundary_band
from .fusion import BIG, color_sqdiff, color_weights, pairwise_tables, sod_binary
from .fusion_band import FRAC_UNIFORM, _bbox, linear_unary

NO_PARENT = -1      # root (seed) or no constraint
UNREACHABLE = -2    # no path to any seed: the hard constraint forces background
HARD = np.inf       # beta of the hard constraint (replaced by a finite bound, see minimize_band_star)
T_SEED = 0.9        # loose band seeds
C_SEED = 2.0
T_STRICT = 0.9      # strict band seeds
C_STRICT = 3.0
A_MIN_PX = 16       # strict seed regions: at least max(A_MIN_PX, A_MIN_FRAC * band pixels) pixels
A_MIN_FRAC = 1e-4
SEED_MODES = ("interior", "strict", "loose")
SEED_RULE_VERSION = 3


def _fold(u, tables, free, fixed, idx):
    """As fusion_band.minimize_band: the pairwise terms with a fixed neighbour become unary terms (u is
    updated in place); returns the free-free edges (i, j, capacity of the directed edge i -> j)."""
    (br, cr), (bd, cd) = tables
    edges = []
    for b, c, sl_p, sl_q in ((br, cr, np.s_[:, :-1], np.s_[:, 1:]), (bd, cd, np.s_[:-1], np.s_[1:])):
        fp, fq = free[sl_p], free[sl_q]
        both = fp & fq
        p_only, q_only = fp & ~fq, fq & ~fp
        up, uq = u[sl_p], u[sl_q]
        xq = fixed[sl_q][p_only].astype(np.float64)
        up[p_only] += c[p_only] * (1 - xq) - b[p_only] * xq
        xp = fixed[sl_p][q_only].astype(np.float64)
        uq[q_only] += b[q_only] * (1 - xp) - c[q_only] * xp
        cb = c[both]
        up[both] += cb
        uq[both] -= cb
        edges.append((idx[sl_p][both], idx[sl_q][both], b[both] + cb))
    return edges


def hard_beta(u_folded, free):
    """A finite beta that makes the star constraint hard: every labeling that violates a star term costs
    more than the all-background labeling of the free pixels (which satisfies the constraint), because the
    folded energy is at least (constant - sum of the negative unaries) and the all-background labeling costs
    the constant. So beta > sum_p max(-u'_p, 0) suffices; never below fusion.BIG."""
    return max(BIG, 1.0 + float(np.maximum(-u_folded[free], 0).sum()))


def star_terms(parent, free, fixed):
    """Classify the star terms of the free pixels: (p, q) flat indices of free pixels with a free parent,
    and the flat indices of free pixels whose parent is fixed background or unreachable (beta on u(1))."""
    pf = parent.ravel()
    freef = free.ravel()
    fixf = fixed.ravel()
    p = np.flatnonzero(freef & (pf >= 0))
    q = pf[p].astype(np.int64)
    q_free = freef[q]
    to_bg = q[~q_free]
    penal = np.concatenate([p[~q_free][~fixf[to_bg]], np.flatnonzero(freef & (pf == UNREACHABLE))])
    return p[q_free], q[q_free], penal


def violations(x, parent, free):
    """Number of free foreground pixels whose parent is background or that are unreachable."""
    pf, xf = parent.ravel(), x.ravel()
    fg = free.ravel() & xf
    has = fg & (pf >= 0)
    return int((~xf[pf[has].astype(np.int64)]).sum() + (fg & (pf == UNREACHABLE)).sum())


def minimize_band_star(unary1, tables, free, fixed, parent, beta, info=None):
    """fusion_band.minimize_band plus the star terms beta * [x_p = 1, x_parent(p) = 0] of the free pixels.
    parent: int array of the shape of unary1 with the row-major flat index of each pixel's parent,
    NO_PARENT (-1) or UNREACHABLE (-2); only the entries of free pixels are read. beta = HARD (inf) is the
    hard constraint (a finite bound, hard_beta); info then receives the number of violations (0 in theory).
    Returns the full boolean labeling (fixed outside free)."""
    h, w = unary1.shape
    free = np.asarray(free, bool)
    fixed = np.asarray(fixed, bool)
    parent = np.asarray(parent)
    x = fixed.copy()
    n_free = int(free.sum())
    if n_free == 0:
        return x
    u = unary1.astype(np.float64).copy()
    idx = np.full((h, w), -1, np.int64)
    idx[free] = np.arange(n_free)
    edges = _fold(u, tables, free, fixed, idx)
    hard = np.isinf(beta)
    b = hard_beta(u, free) if hard else float(beta)
    if info is not None:
        info["beta_used"] = b
    sp, sq, penal = star_terms(parent, free, fixed)
    if b > 0:
        u.ravel()[penal] += b

    n_edges = sum(len(e[0]) for e in edges) + len(sp)
    g = maxflow.Graph[float](n_free, max(n_edges, 1))
    nodes = g.add_nodes(n_free)
    uf = u[free]
    # label 1 = sink segment: x_p = 1 cuts the source edge, so it carries u_p(1) > 0 (as fusion.minimize)
    g.add_grid_tedges(nodes, np.maximum(uf, 0), np.maximum(-uf, 0))
    for i, j, cap in edges:
        if len(i):
            g.add_edges(i, j, np.maximum(cap, 0), np.zeros(len(i)))
    if b > 0 and len(sp):
        # [x_p = 1, x_q = 0]: cut when q is on the source side and p on the sink side, i.e. the edge q -> p;
        # added as the reverse capacity of the edge p -> q
        ip, iq = idx.ravel()[sp], idx.ravel()[sq]
        g.add_edges(ip, iq, np.zeros(len(ip)), np.full(len(ip), b))
    g.maxflow()
    x[free] = g.get_grid_segments(nodes)
    if hard and info is not None:
        info["violations"] = violations(x, parent, free)
    return x


def strict_seeds(s, counts, free, t_s=T_STRICT, c_s=C_STRICT):
    """Band pixels with s >= t_s and count >= c_s, in 4-connected regions of at least
    max(A_MIN_PX, A_MIN_FRAC * band pixels) pixels."""
    cand = free & (s >= t_s) & (counts >= c_s)
    if not cand.any():
        return cand
    a_min = max(A_MIN_PX, A_MIN_FRAC * int(free.sum()))
    lab, n = cc_label(cand)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    keep = area >= a_min
    keep[0] = False
    return keep[lab]


def star_parents(s, counts, free, fixed, gamma=0.0, t_s=T_SEED, c_s=C_SEED, info=None, seed_mode="strict"):
    """Geodesic parents of the free pixels (see the module docstring).
    s: SOD map divided by its maximum over the whole image (float, in [0, 1]); counts: the mean SAM counts;
    free: the band; fixed: the labels outside it; seed_mode: interior | strict | loose (t_s, c_s are the
    thresholds of loose). Returns an int32 array of flat parent indices (NO_PARENT for seeds, non-free pixels
    and the pixels of band components without any root; star_parents never returns UNREACHABLE).
    info receives the number of band components by kind (components: interior = 4-adjacent to the
    interior, band_seeds = only the mode's band seeds, own_strict = rootless in the mode, given their own
    strict seeds, unconstrained = no root at all), seed_source (the same as a string), n_unconstrained_px
    and n_unreachable (nodes without a path to a root; 0 by construction)."""
    if seed_mode not in SEED_MODES:
        raise ValueError(seed_mode)
    t0 = time.time()
    h, w = free.shape
    if h * w >= 2 ** 31:
        raise ValueError("box too large for int32 indices")
    free = np.asarray(free, bool)
    fixed_fg = np.asarray(fixed, bool) & ~free
    near = np.zeros_like(free)       # 4-neighbours of the band
    near[:, 1:] |= free[:, :-1]
    near[:, :-1] |= free[:, 1:]
    near[1:] |= free[:-1]
    near[:-1] |= free[1:]
    seed_fixed = fixed_fg & near
    strict = None
    if seed_mode == "interior":
        seed_band = np.zeros_like(free)
    elif seed_mode == "strict":
        seed_band = strict = strict_seeds(s, counts, free)
    else:
        seed_band = free & (s >= t_s) & (counts >= c_s)
    # per 4-connected component of the band (the graph's connectivity): does it reach a root?
    lab, n_comp = cc_label(free)
    touch = np.zeros_like(free)      # band pixels 4-adjacent to the interior seeds
    touch[:, 1:] |= seed_fixed[:, :-1]
    touch[:, :-1] |= seed_fixed[:, 1:]
    touch[1:] |= seed_fixed[:-1]
    touch[:-1] |= seed_fixed[1:]
    has_int = np.zeros(n_comp + 1, bool)
    has_int[lab[free & touch]] = True
    has_band = np.zeros(n_comp + 1, bool)
    has_band[lab[seed_band]] = True
    rootless = ~(has_int | has_band)
    rootless[0] = False
    has_own = np.zeros(n_comp + 1, bool)
    if rootless.any():
        # a component without a root (a thin part with no interior of its own): its own strict seeds
        if strict is None:
            strict = strict_seeds(s, counts, free)
        own = strict & rootless[lab]
        has_own[lab[own]] = True
        seed_band = seed_band | own
    uncon = rootless & ~has_own      # no root at all: no constraint (NO_PARENT), never UNREACHABLE
    del touch, strict
    active = free & ~uncon[lab]
    nodes = active | seed_fixed
    seeds = seed_band | seed_fixed
    parent = np.full((h, w), NO_PARENT, np.int32)
    n_nodes, n_seeds = int(nodes.sum()), int(seeds.sum())
    comps = {"interior": int(has_int[1:].sum()), "band_seeds": int((has_band & ~has_int)[1:].sum()),
             "own_strict": int(has_own[1:].sum()), "unconstrained": int(uncon[1:].sum())}
    source = ",".join(f"{k}={v}" for k, v in comps.items() if v) or "no_band"
    stats = {"n_nodes": n_nodes, "n_seeds": n_seeds, "n_seed_band": int(seed_band.sum()),
             "n_unreachable": 0, "dijkstra_s": 0.0, "seed_source": source, "components": comps,
             "n_unconstrained_px": int(free.sum() - active.sum())}
    del lab
    if n_seeds == 0 or not active.any():
        if info is not None:
            info.update(stats)
        return parent
    idx = np.full((h, w), -1, np.int32)
    idx[nodes] = np.arange(n_nodes, dtype=np.int32)
    rows, cols, data = [], [], []
    for sl_p, sl_q in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1], np.s_[1:])):
        both = nodes[sl_p] & nodes[sl_q]
        both &= active[sl_p] | active[sl_q]      # no edges between two fixed seeds
        i, j = idx[sl_p][both], idx[sl_q][both]
        c = 1.0 + gamma * (1.0 - 0.5 * (s[sl_p][both] + s[sl_q][both]))
        rows += [i, j]
        cols += [j, i]
        data += [c, c]
        del both
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    data = np.concatenate(data)
    graph = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    del rows, cols, data
    stats["n_entries"] = int(graph.nnz)
    t1 = time.time()
    # min_only with predecessors returns (dist, predecessors, sources)
    pred = dijkstra(graph, directed=True, indices=idx[seeds], return_predecessors=True, min_only=True)[1]
    stats["dijkstra_s"] = time.time() - t1
    del graph
    node_flat = np.flatnonzero(nodes.ravel())
    free_nodes = active.ravel()[node_flat]
    pn = pred[free_nodes]
    # every active component holds a root, so a node without a predecessor is a seed; anything else would
    # be a bug: counted (n_unreachable) and left unconstrained (NO_PARENT), never forced to background
    out = np.where(pn >= 0, node_flat[np.maximum(pn, 0)], NO_PARENT)
    is_seed = seeds.ravel()[node_flat][free_nodes]
    stats["n_unreachable"] = int(((pn < 0) & ~is_seed).sum())
    parent.ravel()[node_flat[free_nodes]] = out
    stats["parents_s"] = time.time() - t0
    if info is not None:
        info.update(stats)
    return parent


def prepare_band(img, sod_map, counts, params, band_r, threshold=127.0):
    """The band problem of fusion_band.fuse_full: the bounding box of the band plus one pixel of margin,
    its unary, pairwise tables, free mask and fixed labels. None when the band is empty."""
    sod_mask = sod_binary(sod_map, threshold)
    free = boundary_band(sod_map, band_r, threshold)
    if not free.any():
        return {"sod_mask": sod_mask, "box": None, "n_free": 0}
    y0, y1, x0, x1 = _bbox(free, 1, free.shape)
    box = np.s_[y0:y1, x0:x1]
    u = linear_unary(sod_map[box], counts[box], params)
    w_right, w_down = color_weights(None, params["sigma"], params["lam"], FRAC_UNIFORM, color_sqdiff(img[box]))
    tables = pairwise_tables(counts[box].astype(np.float64), w_right, w_down, params["c_neg"], "full", "paper")
    fb = free[box]
    return {"sod_mask": sod_mask, "box": box, "u": u, "tables": tables, "free": fb,
            "fixed": sod_mask[box], "n_free": int(fb.sum())}


def fuse_full_star(img, sod_map, counts, params, band_r, beta, gamma=0.0, t_s=T_SEED, c_s=C_SEED,
                   threshold=127.0, prep=None, parent=None, info=None, seed_mode="strict"):
    """fusion_band.fuse_full with the star terms (beta = HARD for the hard constraint). prep (prepare_band)
    and parent (star_parents on the box) can be passed to share them between rows."""
    if prep is None:
        prep = prepare_band(img, sod_map, counts, params, band_r, threshold)
    if prep["box"] is None:
        return prep["sod_mask"]
    box = prep["box"]
    if parent is None:
        s = sod_map[box].astype(np.float64) / (float(sod_map.max()) + 1e-13)
        parent = star_parents(s, counts[box], prep["free"], prep["fixed"], gamma, t_s, c_s, info, seed_mode)
    x = prep["sod_mask"].copy()
    x[box] = minimize_band_star(prep["u"], prep["tables"], prep["free"], prep["fixed"], parent, beta, info)
    return x
