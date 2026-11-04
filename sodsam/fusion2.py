"""Generalized fusion: alternative unary terms, pairwise terms and outputs.

A model is (unary, pairwise, output):
  unary (a base, optionally followed by "+modifier"s, e.g. "table_lin+veto+band")
    table        u(1) = f[sod >= 127][n],  n = number of SAM masks containing p (the paper; 8 params)
    table_lin    u(1) = f[sod >= 127][n] - a * L(s): the table plus the SOD confidence within each
                 cell (9 params)
    bins4        u(1) = f[saliency bin][n], 4 saliency bins split at 64/128/192 (16 params)
    linear       u(1) = d - a * L(s) - b * (n - 1.5),  L = logit of the saliency in [0,1] clipped to
                 [1/255, 254/255] (3 params)
    linear_conf  as linear with n replaced by 3 * sum_k c_k m_k / sum_k c_k (SAM scores c_k)
    linear_gate  as linear with b scaled per image by sigmoid(20 * (IoU - t)), IoU between the SOD
                 mask and the majority SAM mask (4 params): trust SAM only when it agrees with SOD
    linear_prob  as linear with n replaced by the sum of the three SAM probabilities (needs probs)
    tree         u(1) = g[sod >= 127][U] - a * L(s), where U is the union of the level-set components
                 {n >= j} (j = 1..3, nested, a containment tree) selected by dynamic programming:
                 a component is taken whole when its gain sum_p (s_p - tau) + beta_j * area beats the
                 best selection among its children (10 params)
    duo          two SOD models (sod2 = second map, its SAM masks appended to sam):
                 u(1) = d - a * L(s) - a2 * L(s2) - 3 b (1 + g |s - s2|) (n / k - 1/2): SAM's weight
                 grows where the SOD models disagree, SAM as the referee (5 params)
    duo_plain    the same without the disagreement term (4 params; the plain ensemble)
    logodds      u(1) = d - a * lo, lo = learned per-pixel log-odds log P(GT = 1 | features) of
                 unary_learn.py (table or gbt model, loaded as x["lo"] from output/<sod>/<ds>/lo_<model>/)
                 (2 params)
  modifiers
    +veto        + v * n_bg, n_bg = masks of the complement (inverted-map) prompt containing p, on the
                 0..3 scale: SAM's background segmentation votes against saliency (1 param)
    +band        w * u + (1 - w) * (d_far - a_far * L(s)), w = exp(-kappa * dist), dist = distance to
                 the boundary of the thresholded SOD map in units of max(H, W): SAM decides near the
                 SOD boundary, SOD alone decides far from it (3 params; kappa = 0 is the base model)
    +thin        (linear base only) thin-structure region T = the parts of the SOD mask thinner than
                 2 r + 1 pixels (mask minus its opening by a disk of radius r = round(r_t)), dilated by
                 the same disk so the gaps between thin parts are included; inside T the SAM term is
                 scaled by 1 - g_b and every Potts weight of a pair touching T by 1 - g_p: SAM (which
                 works at ~1024 px) and the boundary-length penalty both erase thin parts, so there SOD
                 decides (3 params g_b, g_p, r_t; g_b = g_p = 0, their lower bounds, is the base model)
    +sel         region selection among SAM's masks (one per granularity): mask k scores
                 sum_{p in m_k} (s_p - tau_s) / sum_p s_p + beta_s * c_k (c_k = SAM's predicted IoU);
                 the best mask U, if its score is > 0, adds -e * (2 [p in U] - 1) (nothing otherwise):
                 a vote for one whole SAM region instead of per-pixel counts (3 params e, tau_s, beta_s;
                 e = 0 is the base model; needs the SAM scores)
  pairwise   none | potts (sigma, lam) | paper (sigma, lam, c_neg; see fusion.py)
             | feat (sigma_f, lam, c_neg: SAM feature distances instead of color, needs x["feat"])
             | both (sigma, sigma_f, lam, c_neg: the product of the color and feature Gaussians)
             | paper_sb (paper + a SAM boundary band; sigma, lam, c_neg, rho, r_sb): every Potts weight
               is scaled by 1 - rho * exp(-D^2 / (2 r_sb^2)), D = distance (pixels) of the pair to the
               SAM boundary B (both pixels of every 4-neighbour pair across the boundary of the majority
               mask, count >= 1.5 on the 3-mask scale; this contains the endpoints of every repulsive
               pair, and unlike them it also covers where the masks are offset by a few pixels and the
               count steps by 1), D = min over the pair's two pixels; no B (no majority boundary)
               leaves the weights unchanged. The cut becomes cheaper within
               a few r_sb of SAM's boundary, and the color contrast picks where in the band it goes.
               Still submodular (weights stay >= 0 for rho <= 1); rho = 0 is exactly paper.
  output     binary | soft (the SOD map in 0..255, raised to >= 128 inside the cut and capped at <= 127
             outside it, so thresholding at 128 gives exactly the cut)
  band       fuse2(..., band=True) clamps the pixels farther than th["band_w"] pixels from the SOD
             boundary to the SOD label (fusion.clamp_band); tune2.py --band, range BAND_RANGE
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label

from .fusion import clamp_band, color_sqdiff, minimize, pair_weights, pairwise_tables

PAIRWISE_RANGES = {"potts": {"sigma": (1, 50), "lam": (0, 50)},
                   "paper": {"sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0)},
                   "feat": {"sigma_f": (0.02, 1.0), "lam": (0, 50), "c_neg": (-50, 0)},
                   "both": {"sigma": (1, 50), "sigma_f": (0.02, 1.0), "lam": (0, 50), "c_neg": (-50, 0)},
                   # rho at its lower bound 0 switches the band off (exact paper; --init from a paper result)
                   "paper_sb": {"sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0), "rho": (0, 1),
                                "r_sb": (0.5, 16)}}
BAND_RANGE = {"band_w": (2, 64)}
LIN = {"a": (0, 100), "b": (0, 200), "d": (-200, 200)}
UNARY_RANGES = {
    "table": {"f00": (0, 1000), "f01": (0, 1000), "f02": (-300, 300), "f03": (-300, 100),
              "f10": (-50, 50), "f11": (-100, 50), "f12": (-500, 0), "f13": (-1000, 0)},
    "bins4": {f"g{b}{n}": (-400, 400) for b in range(4) for n in range(4)},
    # wide cells: the tuned table and linear models (see as_table_lin) both lie inside
    "table_lin": {**{f"f{s}{n}": (-1000, 1000) for s in range(2) for n in range(4)}, "a": (0, 100)},
    "linear": LIN,
    "linear_conf": LIN,
    "linear_gate": {**LIN, "t": (0, 1)},
    "linear_prob": LIN,
    "tree": {**{f"g{s}{u}": (-1000, 1000) for s in range(2) for u in range(2)}, "a": (0, 100),
             "tau": (0, 1), "beta1": (-0.5, 0.5), "beta2": (-0.5, 0.5), "beta3": (-0.5, 0.5)},
    "duo": {**LIN, "a2": (0, 100), "g": (0, 10)},
    "duo_plain": {**LIN, "a2": (0, 100)},
    "logodds": {"a": (0, 300), "d": (-300, 300)},
}
# the lower bound of the first parameter of each modifier switches it off (exact base model)
MODIFIER_RANGES = {
    "veto": {"v": (0, 300)},
    "band": {"kappa": (0, 500), "a_far": (0, 100), "d_far": (-300, 300)},
    "thin": {"g_b": (0, 1), "g_p": (0, 1), "r_t": (1, 6)},
    "sel": {"e": (0, 300), "tau_s": (0, 1), "beta_s": (-1, 1)},
}
NEEDS_THREE = ("table", "bins4", "table_lin")


def param_ranges(unary, pairwise, band=False):
    base, *mods = unary.split("+")
    r = dict(UNARY_RANGES[base])
    for m in mods:
        r.update(MODIFIER_RANGES[m])
    r.update(PAIRWISE_RANGES.get(pairwise, {}))
    if band:
        r.update(BAND_RANGE)
    return r


def as_table_lin(unary, th):
    """Parameters of a table or linear model (3 masks) as the identical table_lin model."""
    out = {k: v for k, v in th.items() if k in PAIRWISE_RANGES["paper"]}
    if unary == "table":
        out.update({f"f{s}{n}": th[f"f{s}{n}"] for s in range(2) for n in range(4)}, a=0.0)
    elif unary == "linear":
        out.update({f"f{s}{n}": th["d"] - th["b"] * (n - 1.5) for s in range(2) for n in range(4)}, a=th["a"])
    elif unary == "table_lin":
        out.update(th)
    else:
        raise ValueError(f"no table_lin form of unary {unary}")
    return out


def _logit(s):
    s = np.clip(s, 1 / 255, 254 / 255)
    return np.log(s / (1 - s))


def _level_tree(n3, s):
    """Containment tree of the 8-connected components of the level sets {n3 >= j}, j = 1..3.
    Per level: label map (0 = outside) and arrays area, sum_s, parent (index at the level above)."""
    labs, levels = [], []
    for j in (1, 2, 3):
        lab, cnt = label(n3 >= j - 0.5, structure=np.ones((3, 3)))
        flat = lab.ravel()
        area = np.bincount(flat, minlength=cnt + 1)[1:].astype(np.float64)
        sum_s = np.bincount(flat, weights=s.ravel(), minlength=cnt + 1)[1:]
        if j == 1 or cnt == 0:
            parent = np.zeros(cnt, np.int64)
        else:
            # the level sets are nested, so the first pixel of a component names its parent
            order = np.argsort(flat, kind="stable")
            first = order[np.searchsorted(flat[order], np.arange(1, cnt + 1))]
            parent = labs[-1].ravel()[first] - 1
        labs.append(lab)
        levels.append((area, sum_s, parent))
    return labs, levels


def tree_select(tree, tau, betas):
    """Union of the components chosen by the tree DP: a node is taken whole (gain) or left to the
    best selection among its children; ties go to the whole node."""
    labs, levels = tree
    gain = [sum_s - tau * area + b * area for (area, sum_s, _), b in zip(levels, betas)]
    childsum = [np.zeros_like(g) for g in gain]
    for j in (2, 1):
        best = np.maximum(gain[j], childsum[j])
        childsum[j - 1] = np.bincount(levels[j][2], weights=best, minlength=len(gain[j - 1]))[:len(gain[j - 1])]
    U = np.zeros(labs[0].shape, bool)
    covered = np.zeros(len(gain[0]), bool)
    for j in range(3):
        sel = ~covered & (gain[j] >= childsum[j])
        U |= np.concatenate([[False], sel])[labs[j]]
        if j < 2:
            covered = (covered | sel)[levels[j + 1][2]]
    return U


def prepare(inp):
    """Per-image quantities that do not depend on the parameters."""
    s = inp["sod"].astype(np.float64)
    s = s / (s.max() + 1e-13)                       # max-normalized as in the MATLAB code
    m = np.stack(inp["sam"]).astype(np.float64)     # k x H x W (3 per prompt)
    k = len(m)
    sod_mask = s >= 127 / 255
    major = 2 * m.sum(0) >= k                       # majority of the masks (2 of 3 for k = 3)
    union = np.logical_or(sod_mask, major).sum()
    n3 = m.sum(0) * 3.0 / k                          # count on the 3-mask scale
    out = {"s": s, "L": _logit(s), "n": m.sum(0), "k": k, "sod_mask": sod_mask,
           "bin": np.minimum((s * 255).astype(np.int64) // 64, 3),
           "iou": np.logical_and(sod_mask, major).sum() / union if union else 1.0,
           "gt": inp["gt"], "img": inp["img"], "tree": _level_tree(n3, s)}
    if inp.get("scores") is not None and len(inp["scores"]) == k:
        c = np.asarray(inp["scores"], dtype=np.float64)
        out["n_conf"] = k * np.tensordot(c, m, axes=1) / max(c.sum(), 1e-6)   # weighted count in [0, k]
        # +sel: SOD mass and area of every mask, the scores, the masks, the total SOD mass
        out["sel"] = (np.tensordot(m, s, axes=2), m.sum((1, 2)), c, m.astype(bool), max(s.sum(), 1e-6))
    if inp.get("sod2") is not None:
        s2 = inp["sod2"].astype(np.float64)
        s2 = s2 / (s2.max() + 1e-13)
        out["L2"], out["dis"] = _logit(s2), np.abs(s - s2)
    if inp.get("probs") is not None:
        out["n_prob"] = np.stack(inp["probs"]).astype(np.float64).sum(0) / 255.0
    if inp.get("sam_neg") is not None:
        mb = np.stack(inp["sam_neg"]).astype(np.float64)
        out["n_bg"] = mb.sum(0) * 3.0 / len(mb)
    if sod_mask.all() or not sod_mask.any():
        out["dist"] = np.ones(s.shape)
    else:
        out["dist"] = (distance_transform_edt(sod_mask) + distance_transform_edt(~sod_mask)) / max(s.shape)
    out["sqdiff"] = color_sqdiff(inp["img"])
    if inp.get("feat") is not None:                 # (d_right, d_down) float32 in [0, 1], pair shapes
        out["feat"] = inp["feat"]
    if inp.get("lo") is not None:                   # learned log-odds map (unary logodds)
        out["lo"] = np.asarray(inp["lo"], dtype=np.float64)
    return out


def _base_unary(unary, th, x):
    if unary in NEEDS_THREE and x["k"] != 3:
        raise ValueError(f"unary {unary} needs exactly 3 SAM masks, got {x['k']}")
    if unary in ("table", "table_lin"):
        t = np.array([[th[f"f{a}{b}"] for b in range(4)] for a in range(2)])
        u = t[x["sod_mask"].astype(np.int64), x["n"].astype(np.int64)]
        return u - th["a"] * x["L"] if unary == "table_lin" else u
    if unary == "bins4":
        t = np.array([[th[f"g{a}{b}"] for b in range(4)] for a in range(4)])
        return t[x["bin"], x["n"].astype(np.int64)]
    if unary == "tree":
        U = tree_select(x["tree"], th["tau"], (th["beta1"], th["beta2"], th["beta3"]))
        g = np.array([[th["g00"], th["g01"]], [th["g10"], th["g11"]]])
        return g[x["sod_mask"].astype(np.int64), U.astype(np.int64)] - th["a"] * x["L"]
    if unary == "logodds":
        if "lo" not in x:
            raise ValueError("unary logodds needs the learned log-odds map (lo)")
        return th["d"] - th["a"] * x["lo"]
    if unary in ("duo", "duo_plain"):
        if "L2" not in x:
            raise ValueError(f"unary {unary} needs a second SOD map (sod2)")
        b = th["b"] * (1 + th["g"] * x["dis"]) if unary == "duo" else th["b"]
        return th["d"] - th["a"] * x["L"] - th["a2"] * x["L2"] - 3 * b * (x["n"] / x["k"] - 0.5)
    n = {"linear": "n", "linear_gate": "n", "linear_conf": "n_conf", "linear_prob": "n_prob"}[unary]
    b = th["b"]
    if unary == "linear_gate":
        b = b / (1 + np.exp(-20 * (x["iou"] - th["t"])))
    # SAM evidence as a fraction in [0, 1] (any number of masks), centered at 1/2
    return th["d"] - th["a"] * x["L"] - 3 * b * (x[n] / x["k"] - 0.5)


def unary1(unary, th, x):
    base, *mods = unary.split("+")
    u = _base_unary(base, th, x)
    for m in mods:
        if m == "veto":
            if "n_bg" not in x:
                raise ValueError("+veto needs the complement-prompt masks (sam_neg)")
            u = u + th["v"] * x["n_bg"]
        elif m == "band":
            w = np.exp(-th["kappa"] * x["dist"])
            u = w * u + (1 - w) * (th["d_far"] - th["a_far"] * x["L"])
        elif m == "thin":
            if base != "linear" or mods[0] != "thin":
                # the add-back below assumes u is still the plain linear unary
                raise ValueError("+thin needs the linear unary and must be its first modifier")
            if th["g_b"] > 0:
                # add back g_b of the SAM term -3 b (n / k - 1/2) inside T
                u = u + th["g_b"] * thin_region(x, th["r_t"]) * 3 * th["b"] * (x["n"] / x["k"] - 0.5)
        elif m == "sel":
            if "sel" not in x:
                raise ValueError("+sel needs the SAM scores")
            if th["e"] > 0:
                U = sel_region(x["sel"], th["tau_s"], th["beta_s"])
                if U is not None:
                    u = u - th["e"] * (2.0 * U - 1.0)
        else:
            raise ValueError(f"unknown modifier {m}")
    return u


def thin_mask(sod_mask, r):
    """Thin-structure region of a boolean mask: the mask minus its opening by a disk of radius r (the
    parts thinner than 2 r + 1 pixels), dilated by the same disk; r = 0 gives the empty region."""
    if r <= 0 or not sod_mask.any():
        return np.zeros(sod_mask.shape, bool)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(sod_mask.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
    thin = sod_mask & ~opened
    return cv2.dilate(thin.astype(np.uint8), disk).astype(bool)


def thin_region(x, r_t):
    """+thin region T (float32 0/1) of a prepared image for r = round(r_t), cached in x per r (float32:
    up to 6 radii per 1024 px image while tuning)."""
    r = int(round(r_t))
    cache = x.setdefault("thin", {})
    if r not in cache:
        cache[r] = thin_mask(x["sod_mask"], r).astype(np.float32)
    return cache[r]


def sel_region(sel, tau, beta):
    """+sel: the SAM mask with the best score sum_{p in m_k} (s_p - tau) / sum_p s_p + beta * c_k
    (float 0/1), or None when no score is > 0."""
    mass, area, c, masks, total = sel
    g = (mass - tau * area) / total + beta * c
    k = int(np.argmax(g))
    return masks[k].astype(np.float64) if g[k] > 0 else None


def sam_boundary_dist(n3):
    """Euclidean distance (pixels, float32) of every pixel to the SAM boundary B of a count map on the
    3-mask scale (B: both pixels of every 4-neighbour pair across the boundary of the majority mask
    n3 >= 1.5, the same majority as prepare's); None when B is empty. A count change by >= 2 always
    crosses 1.5, so B contains the endpoints of every repulsive pair."""
    maj = n3 >= 1.5
    b = np.zeros(n3.shape, bool)
    jr = maj[:, 1:] != maj[:, :-1]
    jd = maj[1:] != maj[:-1]
    b[:, 1:] |= jr
    b[:, :-1] |= jr
    b[1:] |= jd
    b[:-1] |= jd
    if not b.any():
        return None
    return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def band_factors(dist, rho, r):
    """Potts scale factors (right (H, W-1), down (H-1, W)) of paper_sb from sam_boundary_dist."""
    def f(d):
        return 1.0 - rho * np.exp(-d.astype(np.float64) ** 2 / (2 * r ** 2))
    return f(np.minimum(dist[:, 1:], dist[:, :-1])), f(np.minimum(dist[1:], dist[:-1]))


def fuse2(x, unary, pairwise, output, th, band=False):
    """uint8 map of one prepared image; band: clamp outside the th["band_w"] band (see above)."""
    u = unary1(unary, th, x)
    if band:
        # x["dist"] is the exact Euclidean distance in units of max(H, W) (1.0 when there is no boundary)
        u = clamp_band(u, x["sod_mask"], th["band_w"], x["dist"] * max(x["s"].shape))
    if pairwise == "none":
        lab = u < 0
    else:
        w = pair_weights("paper" if pairwise == "paper_sb" else pairwise, th.get("sigma"), th.get("sigma_f"),
                         th["lam"], 0.01, x["sqdiff"], x.get("feat"))
        if pairwise == "paper_sb" and th["rho"] > 0:
            if "dist_sb" not in x:      # cached in the prepared image (computed once per worker)
                x["dist_sb"] = sam_boundary_dist(x["n"] * 3.0 / x["k"])
            if x["dist_sb"] is not None:
                fr, fd = band_factors(x["dist_sb"], th["rho"], th["r_sb"])
                w = (w[0] * fr, w[1] * fd)
        if "thin" in unary.split("+")[1:] and th["g_p"] > 0:
            T = thin_region(x, th["r_t"])
            # a pair touches T when either pixel is in it
            w = (w[0] * (1 - th["g_p"] * np.maximum(T[:, 1:], T[:, :-1])),
                 w[1] * (1 - th["g_p"] * np.maximum(T[1:], T[:-1])))
        # repulsive edges where the count, on the 3-mask scale, changes by >= 2 (2 of 3 masks agree)
        tables = pairwise_tables(x["n"] * 3.0 / x["k"], *w, th.get("c_neg", 0.0),
                                 "potts" if pairwise == "potts" else "full")
        lab, _ = minimize(u, tables)
    if output == "binary":
        return lab.astype(np.uint8) * 255
    v = np.round(x["s"] * 255)
    # inside >= 128 > 127 >= outside: the cut survives every rounding (min/max at 0.5 made both 128)
    return np.where(lab, np.maximum(v, 128), np.minimum(v, 127)).astype(np.uint8)
