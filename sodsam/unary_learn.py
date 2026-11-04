"""Learned per-pixel log-odds lo(p) = log P(GT = 1 | features) for the fusion2 unary "logodds",
u(1) = d - a * lo. Two models, both fitted on images unseen by the SOD models (MSRA-B4K):

  table  cells = SAM pattern (which of the 3 masks contain p: 8) x SOD bin (16 equal bins of the
         max-normalized map) x distance bin (distance to the nearest boundary of any mask or of the
         thresholded SOD map, bins [0,2) [2,6) [6,16) [16,inf)) = 512 cells;
         lo = log((n1 + 1) / (n0 + 1)) from the pixel counts n1 / n0 of GT = 1 / 0 in the cell.
  gbt    sklearn HistGradientBoostingClassifier on the per-pixel FEATURES below (its decision function
         is the log-odds).

Log-odds maps are clipped to [-LO_CLIP, LO_CLIP] and written as float16 .npy to
output/<sod>/<ds>/lo_<model>/<id>.npy (pipeline.lo_dir / load_lo); scripts/learn_unary.py drives it.
"""
import numpy as np
import cv2

from . import pipeline

N_SBIN = 16
D_EDGES = np.array([0.0, 2.0, 6.0, 16.0, np.inf])      # distance bins (pixels)
N_CELLS = 8 * N_SBIN * (len(D_EDGES) - 1)
LO_CLIP = 8.0
D_CLIP = 512.0          # signed distances are clipped to +-D_CLIP in the GBT features (no boundary: +-inf)
NEAR_PX = 16.0          # "near a boundary" for the boundary-dense sampling of the GBT training pixels
LAYOUT = "cell = (pattern * 16 + sbin) * 4 + dbin; pattern = m1 + 2 m2 + 4 m3; sbin = min(floor(16 s), 15)"
FEATURES = ["m1", "m2", "m3", "n", "s", "s_mean9", "s_std9", "d_m1", "d_m2", "d_m3", "d_sod",
            "score1", "score2", "score3", "area1", "area2", "area3", "grad", "sod_conf"]


def signed_dist(mask):
    """Signed Euclidean distance (pixels) to the boundary of a boolean mask, negative inside
    (1 / -1 for the pixels adjacent to the other label); +-inf when the mask has no boundary."""
    if mask.all():
        return np.full(mask.shape, -np.inf)
    if not mask.any():
        return np.full(mask.shape, np.inf)
    m = mask.astype(np.uint8)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(mask, -inside, outside).astype(np.float64)


def features(inp):
    """Per-pixel feature maps of one pipeline.load_inputs record (3 masks of the prompt): dict of
    H x W arrays (keys FEATURES) plus "pattern" (0..7) and "d_min" (distance to the nearest boundary)."""
    s = inp["sod"].astype(np.float32)
    s = s / (s.max() + 1e-13)
    m = [np.asarray(x, bool) for x in inp["sam"][:3]]
    if len(m) != 3:
        raise ValueError(f"the learned unary needs exactly 3 SAM masks, got {len(m)}")
    sod_mask = s >= 0.5
    f = {"m1": m[0], "m2": m[1], "m3": m[2], "n": m[0].astype(np.int64) + m[1] + m[2], "s": s}
    f["s_mean9"] = cv2.blur(s, (9, 9))
    f["s_std9"] = np.sqrt(np.maximum(cv2.blur(s * s, (9, 9)) - f["s_mean9"] ** 2, 0))
    for k in range(3):
        f[f"d_m{k + 1}"] = signed_dist(m[k])
    f["d_sod"] = signed_dist(sod_mask)
    scores = inp.get("scores") or [0.0, 0.0, 0.0]
    for k in range(3):
        f[f"score{k + 1}"] = float(scores[k])
        f[f"area{k + 1}"] = float(m[k].mean())
    gray = cv2.cvtColor(inp["img"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    f["grad"] = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    f["sod_conf"] = float(((s >= 0.2) & (s <= 0.8)).mean())
    f["pattern"] = m[0].astype(np.int64) + 2 * m[1] + 4 * m[2]
    f["d_min"] = np.min(np.abs(np.stack([f["d_m1"], f["d_m2"], f["d_m3"], f["d_sod"]])), axis=0)
    return f


def feature_matrix(f, idx=None):
    """(N, len(FEATURES)) float32 matrix of the pixels idx (flat indices; all pixels when None)."""
    shape = f["s"].shape
    n = shape[0] * shape[1] if idx is None else len(idx)
    X = np.empty((n, len(FEATURES)), np.float32)
    for j, name in enumerate(FEATURES):
        v = f[name]
        if np.isscalar(v):
            X[:, j] = v
        else:
            v = v.ravel() if idx is None else v.ravel()[idx]
            X[:, j] = np.clip(v, -D_CLIP, D_CLIP) if name.startswith("d_") else v
    return X


def cell_index(f):
    """H x W cell indices (0..N_CELLS - 1) of the table model (LAYOUT)."""
    sbin = np.minimum((f["s"] * N_SBIN).astype(np.int64), N_SBIN - 1)
    dbin = np.searchsorted(D_EDGES[1:-1], f["d_min"], side="right")
    return (f["pattern"] * N_SBIN + sbin) * (len(D_EDGES) - 1) + dbin


def table_counts(f, gt):
    """(n1, n0): pixel counts of GT = 1 / 0 per cell of one image."""
    c = cell_index(f).ravel()
    g = np.asarray(gt, bool).ravel()
    return np.bincount(c[g], minlength=N_CELLS), np.bincount(c[~g], minlength=N_CELLS)


def table_logodds(n1, n0):
    """Clipped log((n1 + 1) / (n0 + 1)) per cell, float32."""
    lo = np.log((n1 + 1.0) / (n0 + 1.0))
    return np.clip(lo, -LO_CLIP, LO_CLIP).astype(np.float32)


def save_table(path, n1, n0, meta=None):
    np.savez(path, n1=n1, n0=n0, lo=table_logodds(n1, n0), s_edges=np.linspace(0, 1, N_SBIN + 1),
             d_edges=D_EDGES, layout=LAYOUT, **(meta or {}))


def load_table(path):
    """Log-odds per cell (N_CELLS,) of a saved table."""
    with np.load(path) as z:
        return table_logodds(z["n1"], z["n0"])


def table_predict(lo_cells, f):
    return lo_cells[cell_index(f)]


def gbt_predict(clf, f):
    """Clipped log-odds map of a fitted HistGradientBoostingClassifier (binary log loss: the
    decision function is the log-odds)."""
    lo = clf.decision_function(feature_matrix(f)).reshape(f["s"].shape)
    return np.clip(lo, -LO_CLIP, LO_CLIP).astype(np.float32)


def sample_pixels(f, rng, n, near_frac=0.5, near_px=NEAR_PX):
    """Flat indices of a stratified sample of n pixels: a share near_frac within near_px of any
    boundary (d_min <= near_px), the rest from the other pixels (fewer when a stratum is smaller)."""
    near = np.flatnonzero(f["d_min"].ravel() <= near_px)
    far = np.flatnonzero(f["d_min"].ravel() > near_px)
    k_near = min(int(round(n * near_frac)), len(near))
    k_far = min(n - k_near, len(far))
    return np.concatenate([rng.choice(near, k_near, replace=False), rng.choice(far, k_far, replace=False)])


def write_lo(sod_model, ds, model, id_, lo):
    d = pipeline.lo_dir(sod_model, ds, model)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{id_}.npy", np.clip(lo, -LO_CLIP, LO_CLIP).astype(np.float16))
