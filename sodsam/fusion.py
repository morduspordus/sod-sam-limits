"""Fusion of the SOD mask and three SAM masks by graph-cut energy minimization.

Energy (labels x_p in {0,1}, 1 = object, 4-connected grid N):
    E(x) = sum_p u_p(x_p) + sum_{(p,q) in N} v_pq(x_p, x_q),
    u_p(0) = 0,  u_p(1) = f(M_SOD(p), M_SAM(p)),  M_SAM = M1 + M2 + M3 in {0..3}.

For a neighbor pair (p, q) (q to the right of / below p) with d = M_SAM(p) - M_SAM(q):
    d <= -2  (p background side, q object side, pair in E^01):
        v(0,1) = c_neg,  v(1,0) = -c_neg   ("paper", modular, exactly submodular)
                          v(1,0) = w_pq     ("legacy", as in the MATLAB code)
    d >= 2   (pair in E^10): the same with the roles of (0,1) and (1,0) swapped,
    otherwise Potts:  v(0,1) = v(1,0) = w_pq,
    w_pq = lambda * (u + (1 - u) * exp(-||C_p - C_q||^2 / (2 sigma^2))),  u = frac_uniform.

Pairwise weight variants (fuse(pairwise=...)): "paper" (color, above), "feat" (SAM feature distance
d_pq = 1 - cos(f_p, f_q) in [0, 1], exp(-d^2 / (2 sigma_f^2)); scripts/run_sam_extra.py --prompt feat),
"both" (exp of the sum of the two exponents); all three keep the repulsive term of "paper".
Band constraint (fuse(band=w)): pixels farther than w pixels from the boundary of the thresholded
SOD map are clamped to the SOD label by a terminal capacity of BIG (safety variant).

The "legacy" pairwise variant is not submodular when w_pq < -c_neg. The MATLAB code passed
it to the BK maxflow with the regularity assert compiled out, so the graph got negative
capacities. Here negative capacities are clipped to 0 and counted, which gives a minimizer
of a slightly different energy; it is kept only for comparison with the old results.
"""
from dataclasses import asdict, dataclass, field

import cv2
import maxflow
import numpy as np

BIG = 1e6   # terminal capacity of a clamped pixel (finite, far above any sum of edge weights)
PAIRWISE = ("none", "potts", "paper", "feat", "both")


@dataclass
class FusionParams:
    # unary table f[sod][sam], sod in {0,1}, sam in {0,1,2,3}; u_p(0) = 0
    unary: list = field(default_factory=lambda: [[90.0, 60.0, 20.0, -10.0],
                                                 [1.0, -1.0, -10.0, -70.0]])
    sigma: float = 20.0          # sigma_col
    sigma_f: float = 0.2         # sigma of the SAM feature distance (pairwise "feat" / "both" only)
    lam: float = 20.0            # lambda (Potts weight)
    c_neg: float = -10.0         # repulsive weight (< 0)
    frac_uniform: float = 0.01   # floor of the Gaussian color weight, as in take_triple_diff.m
    sod_threshold: float = 127.0  # on the max-normalized map, pixel >= threshold is object

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# values printed in the paper (Sect. 5.6, Table 4)
PAPER_PARAMS = FusionParams()

# Param/optimal_most_general_3_masks.m: the parameters actually used for the paper's tables
MATLAB_OPTIMAL_PARAMS = FusionParams(
    unary=[[903.9937, 638.4027, 181.9473, -109.7902],
           [9.5557, -0.7349, -329.1187, -737.8861]],
    sigma=23.9928, lam=14.7898, c_neg=-25.6846)


def sod_binary(sod_map, threshold=127.0):
    """Threshold the soft SOD map after rescaling it to max 255 (processDir_3_masks.m)."""
    s = sod_map.astype(np.float64)
    s = 255.0 * s / (s.max() + 1e-13)
    return s >= threshold


def color_sqdiff(img):
    """Squared RGB differences of right (H, W-1) and down (H-1, W) neighbor pairs."""
    im = img.astype(np.float64)
    return ((im[:, 1:] - im[:, :-1]) ** 2).sum(-1), ((im[1:] - im[:-1]) ** 2).sum(-1)


def color_weights(img, sigma, lam, frac_uniform, sqdiff=None):
    """Gaussian color weights for right (H, W-1) and down (H-1, W) neighbor pairs."""
    d_right, d_down = color_sqdiff(img) if sqdiff is None else sqdiff

    def w(d):
        return lam * (frac_uniform + (1 - frac_uniform) * np.exp(-d / (2 * sigma ** 2)))
    return w(d_right), w(d_down)


def feat_weights(d_right, d_down, sigma_f, lam, frac_uniform, sqdiff=None, sigma=None):
    """Gaussian weights of the SAM feature distances d in [0, 1] of the right (H, W-1) and down
    (H-1, W) pairs; with sqdiff and sigma the color exponent is added ("both")."""
    def w(d, sq):
        e = -d.astype(np.float64) ** 2 / (2 * sigma_f ** 2)
        if sq is not None:
            e = e - sq / (2 * sigma ** 2)
        return lam * (frac_uniform + (1 - frac_uniform) * np.exp(e))
    sq_r, sq_d = (None, None) if sqdiff is None else sqdiff
    return w(d_right, sq_r), w(d_down, sq_d)


def pair_weights(pairwise, sigma, sigma_f, lam, frac_uniform, sqdiff, feat):
    """Weights of a pairwise mode; sqdiff = color_sqdiff(img), feat = (d_right, d_down) feature
    distances cropped to the pair shapes (pipeline.load_feat)."""
    if pairwise in ("potts", "paper", "legacy"):
        return color_weights(None, sigma, lam, frac_uniform, sqdiff)
    if feat is None:
        raise ValueError(f"pairwise {pairwise} needs the SAM feature distances (sam_feat)")
    if pairwise == "feat":
        return feat_weights(*feat, sigma_f, lam, frac_uniform)
    if pairwise == "both":
        return feat_weights(*feat, sigma_f, lam, frac_uniform, sqdiff, sigma)
    raise ValueError(pairwise)


def boundary_dist(mask):
    """Euclidean distance (pixels) of every pixel to the boundary of a boolean mask: 1 for pixels
    adjacent to the other label, inf when the mask has no boundary."""
    if mask.all() or not mask.any():
        return np.full(mask.shape, np.inf)
    m = mask.astype(np.uint8)
    return (cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            + cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)).astype(np.float64)


def clamp_band(unary1, sod_mask, width, dist=None):
    """Unary with the pixels farther than width from the SOD boundary clamped to the SOD label."""
    far = (boundary_dist(sod_mask) if dist is None else dist) > width
    return np.where(far, np.where(sod_mask, -BIG, BIG), unary1)


def pairwise_tables(m_sam, w_right, w_down, c_neg, variant="full", pairwise="paper"):
    """Costs b = v(0,1) and c = v(1,0) for the right and down neighbor pairs.

    variant: "full" (Potts + repulsive), "potts" (no repulsive terms).
    """
    out = []
    for w, d in ((w_right, m_sam[:, :-1] - m_sam[:, 1:]),
                 (w_down, m_sam[:-1] - m_sam[1:])):
        b, c = w.copy(), w.copy()
        if variant == "full":
            e01, e10 = d <= -2, d >= 2
            b[e01] = c_neg
            c[e10] = c_neg
            if pairwise == "paper":
                c[e01] = -c_neg
                b[e10] = -c_neg
            elif pairwise != "legacy":
                raise ValueError(pairwise)
        elif variant != "potts":
            raise ValueError(variant)
        out.append((b, c))
    return out


def energy(x, unary1, tables):
    """Energy of a labeling (for tests and diagnostics)."""
    x = x.astype(np.int64)
    e = (unary1 * x).sum()
    (br, cr), (bd, cd) = tables
    for (b, c), xp, xq in (((br, cr), x[:, :-1], x[:, 1:]), ((bd, cd), x[:-1], x[1:])):
        e += (b * ((1 - xp) * xq)).sum() + (c * (xp * (1 - xq))).sum()
    return float(e)


def minimize(unary1, tables):
    """Exact minimization with a graph cut (for submodular tables).

    Each pairwise term b*(1-x_p)*x_q + c*x_p*(1-x_q) is rewritten as
    c*x_p - c*x_q + (b + c)*(1-x_p)*x_q, so the unary terms absorb c and a single directed
    edge p -> q with capacity b + c remains. Returns (labels, number of clipped edges).
    """
    h, w = unary1.shape
    u = unary1.astype(np.float64).copy()
    (br, cr), (bd, cd) = tables
    u[:, :-1] += cr
    u[:, 1:] -= cr
    u[:-1] += cd
    u[1:] -= cd
    cap_r, cap_d = br + cr, bd + cd
    n_clipped = int((cap_r < 0).sum() + (cap_d < 0).sum())

    g = maxflow.Graph[float](h * w, 2 * h * w)  # preallocation makes construction ~2x faster
    nodes = g.add_grid_nodes((h, w))
    # label 1 = sink segment: x_p = 1 cuts the source edge, so it carries u_p(1) > 0
    g.add_grid_tedges(nodes, np.maximum(u, 0), np.maximum(-u, 0))
    wr = np.zeros((h, w))
    wr[:, :-1] = np.maximum(cap_r, 0)
    g.add_grid_edges(nodes, weights=wr, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
                     symmetric=False)
    wd = np.zeros((h, w))
    wd[:-1] = np.maximum(cap_d, 0)
    g.add_grid_edges(nodes, weights=wd, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
                     symmetric=False)
    g.maxflow()
    return g.get_grid_segments(nodes), n_clipped


def unary_term(sod_mask, sam_masks, params):
    m_sam = sum((m > 0).astype(np.int64) for m in sam_masks)
    table = np.asarray(params.unary, dtype=np.float64)
    return table[sod_mask.astype(np.int64), m_sam], m_sam


def fuse(img, sod_map, sam_masks, params, variant="full", pairwise="paper", sqdiff=None, feat=None,
         band=None):
    """Returns a boolean object mask.

    variant: "full", "potts" (no repulsive terms), or "unary" (no pairwise terms).
    pairwise: "paper" | "legacy" (repulsive form, color weights) | "feat" | "both" (see the module
    docstring; feat = (d_right, d_down) feature distances cropped to the pair shapes).
    sqdiff: optional cached color_sqdiff(img).  band: width in pixels of the band around the SOD
    boundary outside which pixels keep the SOD label (None = off).
    """
    sod_mask = sod_binary(sod_map, params.sod_threshold)
    unary1, m_sam = unary_term(sod_mask, sam_masks, params)
    if band is not None:
        unary1 = clamp_band(unary1, sod_mask, band)
    if variant == "unary":
        return unary1 < 0
    if sqdiff is None:
        sqdiff = color_sqdiff(img)
    w_right, w_down = pair_weights(pairwise, params.sigma, params.sigma_f, params.lam,
                                   params.frac_uniform, sqdiff, feat)
    tables = pairwise_tables(m_sam, w_right, w_down, params.c_neg, variant,
                             "legacy" if pairwise == "legacy" else "paper")
    x, _ = minimize(unary1, tables)
    return x
