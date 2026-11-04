"""Salient object = the cheapest small set of SAM candidate masks that explains the SOD map
(alpha-expansion with label costs, Delong, Osokin, Isack and Boykov, IJCV 2012).

Labels: 0 = background, k = candidate mask k (the three prompted SAM masks and SAM's automatic masks,
pruned by `candidates`). Energy over the half-resolution grid (4-connected):
    E(f) = sum_p D_p(f_p) + sum_{(p,q)} w_pq [f_p != f_q] + sum_k h_k [some p has f_p = k],
    D_p(0) = 0,  D_p(k) = u_p if p in mask_k else +inf,  u_p = d - a * L(s_p)  (fusion2's linear unary
    without the SAM count: the candidates carry SAM's evidence), w_pq the Gaussian color weights of
    fusion.color_weights (Potts), h_k = lam_L * (1 - iou_k) + h0 the cost of using candidate k.
The object is the union of the chosen candidates (f > 0). Pixels outside every candidate are background
in every labeling, so only the pixels inside the union of the candidates ("active") are graph nodes.

Expansion on alpha, one max-flow per move, label 1 = sink side = switch to alpha (as fusion.minimize):
    t-links: cost(1) = D_p(alpha), cost(0) = D_p(f_p); pixels with f_p = alpha are fixed at 1;
    pairs with f_p = f_q: symmetric edge w; pairs with f_p != f_q: w [not both switch]
        = w (1 - y_p) + w y_p (1 - y_q): sink-link w on p and a directed edge q -> p of capacity w;
    pairs with an inactive q (background forever): w [alpha != 0] on y_p = 1, w [f_p != 0] on y_p = 0;
    label cost of an unused alpha: auxiliary node a, source link h_alpha, edges a -> p of capacity inf
        for the admissible p (a is sink, h_alpha paid, as soon as one pixel takes alpha);
    label cost of a used beta != alpha: auxiliary node b, sink link h_beta, edges p -> b of capacity inf
        for the pixels of beta (b may be sink, h_beta saved, only when all of them switch).
"""
import os

import cv2
import maxflow
import numpy as np

from . import data
from .fusion import color_weights

INF = 1e9
LABELCOST_RANGES = {"a": (0, 100), "d": (-200, 200), "lam_L": (0, 2000), "h0": (0, 2000),
                    "sigma": (1, 50), "lam": (0, 50)}


AUTO_DIR = os.environ.get("SODSAM_AUTO_DIR", "sam_auto")   # sam_auto_loose: run_sam_extra.py --prompt auto_loose


def load_auto(ds, id_):
    """SAM's automatic masks of an image at half resolution (K x h2 x w2 bool), predicted IoU, stability."""
    z = np.load(data.OUT_ROOT / AUTO_DIR / ds / f"{id_}.npz")
    h2, w2 = (int(v) for v in z["shape"])
    masks = np.unpackbits(z["masks"], axis=-1, count=w2).astype(bool).reshape(-1, h2, w2)
    return masks, z["iou"].astype(np.float64), z["stability"].astype(np.float64)


def _iou_matrix(a, b):
    """IoU between every mask of a (K x N flat) and every mask of b (M x N flat)."""
    a, b = a.reshape(len(a), -1).astype(np.float64), b.reshape(len(b), -1).astype(np.float64)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1)


def candidates(sod_mask, auto_masks, auto_iou, prompted, prompted_iou, min_inside=0.3, max_labels=40,
               dilate_frac=0.03, same=0.95):
    """Label set: the prompted masks, then the automatic masks that lie >= min_inside inside the dilated
    SOD mask or overlap a prompted mask (IoU >= 0.5), without near duplicates (IoU >= same), at most
    max_labels by predicted IoU. Returns (masks K x h2 x w2 bool, iou K)."""
    h, w = sod_mask.shape
    r = max(1, int(dilate_frac * max(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    region = cv2.dilate(sod_mask.astype(np.uint8), kernel) > 0
    masks = [m for m in prompted if m.any()]
    ious = [float(c) for m, c in zip(prompted, prompted_iou) if m.any()]
    if len(auto_masks):
        area = auto_masks.reshape(len(auto_masks), -1).sum(1)
        inside = (auto_masks & region).reshape(len(auto_masks), -1).sum(1) / np.maximum(area, 1)
        near = _iou_matrix(auto_masks, np.stack(masks)).max(1) if masks else np.zeros(len(auto_masks))
        keep = np.nonzero((area > 0) & ((inside >= min_inside) | (near >= 0.5)))[0]
        for k in keep[np.argsort(-auto_iou[keep], kind="stable")]:
            masks.append(auto_masks[k])
            ious.append(float(auto_iou[k]))
    if not masks:
        return np.zeros((0, h, w), bool), np.zeros(0)
    M = np.stack(masks)
    sim = _iou_matrix(M, M)
    chosen = []
    for k in range(len(M)):                    # prompted first, then by predicted IoU
        if all(sim[k, j] < same for j in chosen):
            chosen.append(k)
        if len(chosen) == max_labels:
            break
    return M[chosen], np.asarray(ious)[chosen]


def energy(labels, unary_obj, cands, costs, w_right, w_down):
    """E(f) of a label map (0 = background, k = candidate k); inf for an inadmissible labeling."""
    obj = labels > 0
    inside = np.ones(labels.shape, bool)
    for k in range(len(cands)):
        sel = labels == k + 1
        inside[sel] = cands[k][sel]
    if not inside.all():
        return np.inf
    e = unary_obj[obj].sum()
    e += (w_right * (labels[:, :-1] != labels[:, 1:])).sum() + (w_down * (labels[:-1] != labels[1:])).sum()
    e += sum(costs[k] for k in range(len(cands)) if (labels == k + 1).any())
    return float(e)


def _expand(labels, alpha, unary_obj, cands, costs, w_right, w_down, active, idx):
    """One expansion move on alpha; returns the new label map (None if alpha has no admissible pixel)."""
    h, w = labels.shape
    K = len(cands)
    adm = cands[alpha - 1] if alpha else active            # admissible pixels of alpha (active ones)
    adm = adm & active
    if not adm.any() or (labels[adm] == alpha).all():
        return None
    n = int(active.sum())
    g = maxflow.Graph[float](n + K + 1, 4 * n)
    nodes = g.add_nodes(n)
    # t-links: cost(1) = D(alpha), cost(0) = D(current); f_p = alpha fixed at 1
    cur_obj = labels > 0
    d_alpha = np.where(adm, unary_obj if alpha else 0.0, INF)
    d_cur = np.where(cur_obj, unary_obj, 0.0)
    cs, ct = d_alpha.copy(), d_cur.copy()
    fixed = labels == alpha
    cs[fixed], ct[fixed] = 0.0, INF
    # pairs with an inactive neighbor (background forever)
    for (sl_p, sl_q, wgt) in (((slice(None), slice(None, -1)), (slice(None), slice(1, None)), w_right),
                              ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), w_down)):
        for a_sl, b_sl in ((sl_p, sl_q), (sl_q, sl_p)):      # p = a_sl side active, q = b_sl side inactive
            m = np.zeros((h, w), bool)
            m[a_sl] = active[a_sl] & ~active[b_sl]
            ww = np.zeros((h, w))
            ww[a_sl] = wgt
            if alpha:
                cs[m] += ww[m]
            ct[m] += ww[m] * cur_obj[m]
    # pairs of active pixels
    ei, ej, ecap, ercap = [], [], [], []
    for (sl_p, sl_q, wgt) in (((slice(None), slice(None, -1)), (slice(None), slice(1, None)), w_right),
                              ((slice(None, -1), slice(None)), (slice(1, None), slice(None)), w_down)):
        both = active[sl_p] & active[sl_q]
        same = both & (labels[sl_p] == labels[sl_q])
        diff = both & ~same
        p, q = idx[sl_p], idx[sl_q]
        ei += [p[same], q[diff]]
        ej += [q[same], p[diff]]
        ecap += [wgt[same], wgt[diff]]
        ercap += [wgt[same], np.zeros(int(diff.sum()))]
        extra = np.zeros((h, w))                               # the w (1 - y_p) part: a sink link on p
        extra[sl_p] = wgt * diff
        ct += extra
    g.add_grid_tedges(nodes, cs[active], ct[active])
    if ei:
        g.add_edges(np.concatenate(ei), np.concatenate(ej), np.concatenate(ecap), np.concatenate(ercap))
    # label costs
    used = [k for k in range(1, K + 1) if (labels == k).any()]
    if alpha and alpha not in used:
        a = int(g.add_nodes(1)[0])
        g.add_tedge(a, costs[alpha - 1], 0.0)
        dst = idx[adm]                                     # a -> p: p sink forces a sink
        g.add_edges(np.full(len(dst), a), dst, np.full(len(dst), INF), np.zeros(len(dst)))
    for beta in used:
        if beta == alpha:
            continue
        b = int(g.add_nodes(1)[0])
        g.add_tedge(b, 0.0, costs[beta - 1])
        src = idx[labels == beta]
        g.add_edges(src, np.full(len(src), b), np.full(len(src), INF), np.zeros(len(src)))
    g.maxflow()
    switch = np.zeros((h, w), bool)
    switch[active] = g.get_grid_segments(nodes)
    new = labels.copy()
    new[switch] = alpha
    return new


def expand_fit(unary_obj, cands, costs, w_right, w_down, sweeps=2, tol=1e-6):
    """Alpha-expansion with label costs from the all-background labeling; (label map, object mask)."""
    h, w = unary_obj.shape
    labels = np.zeros((h, w), np.int64)
    if not len(cands):
        return labels, labels > 0
    active = cands.any(0)
    idx = np.full((h, w), -1, np.int64)
    idx[active] = np.arange(int(active.sum()))
    e = energy(labels, unary_obj, cands, costs, w_right, w_down)
    for _ in range(sweeps):
        improved = False
        for alpha in range(len(cands) + 1):
            new = _expand(labels, alpha, unary_obj, cands, costs, w_right, w_down, active, idx)
            if new is None:
                continue
            e_new = energy(new, unary_obj, cands, costs, w_right, w_down)
            if e_new < e - tol:
                labels, e, improved = new, e_new, True
        if not improved:
            break
    return labels, labels > 0


def prepare_half(img, sod, gt, prompted, prompted_iou, auto_masks, auto_iou, **cand_kw):
    """Per-image quantities at half resolution: L (logit of the max-normalized SOD map), GT, candidate
    masks (packed) and their IoU, color squared differences of the half-resolution image."""
    s = sod.astype(np.float64)[::2, ::2]
    s = s / (s.max() + 1e-13)
    sod_mask = s >= 127 / 255
    L = np.log(np.clip(s, 1 / 255, 254 / 255) / (1 - np.clip(s, 1 / 255, 254 / 255)))
    im = img[::2, ::2].astype(np.float64)
    sqdiff = (((im[:, 1:] - im[:, :-1]) ** 2).sum(-1), ((im[1:] - im[:-1]) ** 2).sum(-1))
    cands, iou = candidates(sod_mask, auto_masks, auto_iou, [m[::2, ::2] for m in prompted], prompted_iou,
                            **cand_kw)
    return {"L": L, "gt": gt[::2, ::2], "shape": s.shape, "full_shape": sod.shape,
            "cands": np.packbits(cands, axis=-1), "n_cands": len(cands), "iou": iou, "sqdiff": sqdiff}


def load_prepared(sod_model, ds, id_, scores, prompt="mask_zoom", **cand_kw):
    """prepare_half of one image from the saved SOD map, prompted masks (scores: {id: [3 scores]}) and
    automatic masks; without the npz of the automatic masks (not computed) only the prompted ones are
    used and "no_auto" is set."""
    import cv2
    img, gt = data.load_image(ds, id_), data.load_gt(ds, id_)
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    if gt.shape != sod.shape:
        gt = cv2.resize(gt.astype(np.uint8), (sod.shape[1], sod.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    d = data.run_dir(sod_model, ds, f"sam_{prompt}")
    prompted = [data.read_png(d / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
    try:
        auto, aiou, _ = load_auto(ds, id_)
        no_auto = False
    except FileNotFoundError:
        h2, w2 = (sod.shape[0] + 1) // 2, (sod.shape[1] + 1) // 2
        auto, aiou, no_auto = np.zeros((0, h2, w2), bool), np.zeros(0), True
    x = prepare_half(img, sod, gt, prompted, scores[id_], auto, aiou, **cand_kw)
    x["no_auto"] = no_auto
    return x


def fit(x, th, sweeps=2):
    """Object mask (half resolution, bool) of a prepared image for parameters th."""
    h, w = x["shape"]
    cands = np.unpackbits(x["cands"], axis=-1, count=w).astype(bool).reshape(x["n_cands"], h, w)
    u = th["d"] - th["a"] * x["L"]
    costs = th["lam_L"] * (1 - x["iou"]) + th["h0"]
    w_r, w_d = color_weights(None, th["sigma"], th["lam"], 0.01, x["sqdiff"])
    return expand_fit(u, cands, costs, w_r, w_d, sweeps)[1]


def upsample(mask_half, full_shape):
    """Nearest-neighbour upsampling of a half-resolution mask to the full size."""
    m = np.repeat(np.repeat(mask_half, 2, 0), 2, 1)
    return m[:full_shape[0], :full_shape[1]]
