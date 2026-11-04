"""Stitching overlapping SAM windows into one full-resolution label map (pure stitching, no SOD unary).

Every crop c contributes SAM's best-scoring mask probability p_c (argmax of the three scores). Over the
area covered by the crops:
    avg     uniform mean of p_c over the covering crops >= 0.5
    gauss   Gaussian-weighted mean >= 0.5, w_c a 2D Gaussian centred on crop c with sigma = (crop height / 4,
            crop width / 4), max 1 (nnU-Net's sliding-window blending, with sigma = size / 4)
    seam    a graph cut only where the crops disagree: pixels covered by >= 2 crops whose labels p_c >= 0.5
            are not all equal, dilated by r_s pixels (and kept inside the covered area); there
                u_p(1) = - sum_c w_c(p) logit(clip(p_c(p), 1e-4, 1 - 1e-4)) / sum_c w_c(p)
            with the colour Potts pairwise term (fusion.color_weights, fusion.pairwise_tables "potts"),
            solved exactly by fusion_band.minimize_band with every other pixel fixed to the gauss label
            (the fixed neighbours couple the seam to its surroundings).
Outside the covered area every variant takes the thresholded SOD map (fusion.sod_binary at 127).

Stitcher accumulates the crops online (sums over the bounding box of the crops), so no per-crop map is
kept at image size: ~18 bytes per pixel of the crops' bounding box.
"""
import cv2
import numpy as np

from .fusion import color_sqdiff, color_weights, pairwise_tables, sod_binary
from .fusion_band import FRAC_UNIFORM, _bbox, minimize_band

P_CLIP = 1e-4
R_SEAM = 8
LAMS = (2.0, 8.0, 32.0)
SIGMAS = (10.0, 30.0)


def gaussian_weight(h, w):
    """2D Gaussian of an h x w crop centred on it, sigma = (h / 4, w / 4), max 1, float32."""
    gy = np.exp(-0.5 * ((np.arange(h) - (h - 1) / 2) / (h / 4)) ** 2)
    gx = np.exp(-0.5 * ((np.arange(w) - (w - 1) / 2) / (w / 4)) ** 2)
    g = np.outer(gy, gx)
    return (g / g.max()).astype(np.float32)


def seam_name(lam, sigma):
    return f"seam_l{lam:g}_s{sigma:g}"


class Stitcher:
    """Online sums of the crops' probabilities over the bounding box of the crops."""

    def __init__(self, shape, boxes):
        self.shape = tuple(shape[:2])
        self.boxes = [tuple(int(v) for v in b) for b in boxes]
        if self.boxes:
            b = np.array(self.boxes)
            self.oy, self.ox = int(b[:, 0].min()), int(b[:, 1].min())
            hh, ww = int(b[:, 2].max()) - self.oy, int(b[:, 3].max()) - self.ox
        else:
            self.oy = self.ox = hh = ww = 0
        self.sum_p = np.zeros((hh, ww), np.float32)    # sum of p_c (uniform average)
        self.sum_w = np.zeros((hh, ww), np.float32)    # sum of w_c
        self.sum_wp = np.zeros((hh, ww), np.float32)   # sum of w_c p_c
        self.sum_wl = np.zeros((hh, ww), np.float32)   # sum of w_c logit(p_c)
        self.n_cov = np.zeros((hh, ww), np.uint8)      # number of covering crops
        self.n_pos = np.zeros((hh, ww), np.uint8)      # number of covering crops with p_c >= 0.5
        self._wcache = {}

    def add(self, box, prob):
        """Add the probability map (crop size) of the crop box = (y0, x0, y1, x1)."""
        y0, x0, y1, x1 = (int(v) for v in box)
        p = np.asarray(prob, np.float32)
        assert p.shape == (y1 - y0, x1 - x0), (p.shape, box)
        key = p.shape
        if key not in self._wcache:
            self._wcache[key] = gaussian_weight(*key)
        w = self._wcache[key]
        sl = np.s_[y0 - self.oy:y1 - self.oy, x0 - self.ox:x1 - self.ox]
        pc = np.clip(p, P_CLIP, 1 - P_CLIP)
        self.sum_p[sl] += p
        self.sum_w[sl] += w
        self.sum_wp[sl] += w * p
        self.sum_wl[sl] += w * np.log(pc / (1 - pc))
        self.n_cov[sl] += 1
        self.n_pos[sl] += (p >= 0.5).astype(np.uint8)

    @property
    def box(self):
        return np.s_[self.oy:self.oy + self.n_cov.shape[0], self.ox:self.ox + self.n_cov.shape[1]]

    def covered(self):
        return self.n_cov > 0

    def avg(self):
        cov = self.covered()
        return cov & (self.sum_p >= 0.5 * np.maximum(self.n_cov, 1))

    def gauss(self):
        cov = self.covered()
        return cov & (self.sum_wp >= 0.5 * self.sum_w)

    def disagree(self):
        return (self.n_cov >= 2) & (self.n_pos > 0) & (self.n_pos < self.n_cov)

    def unary(self):
        """u(1) = - sum_c w_c logit(p_c) / sum_c w_c (0 where uncovered)."""
        return -np.divide(self.sum_wl, self.sum_w, out=np.zeros_like(self.sum_wl), where=self.sum_w > 0)


def stitch_all(img, sod_map, st, lams=LAMS, sigmas=SIGMAS, r_s=R_SEAM, threshold=127.0, info=None):
    """Full-resolution boolean maps {"avg", "gauss", seam_name(lam, sigma)...} of a filled Stitcher st.
    info receives the seam size (n_disagree, n_seam) and the seconds of the seam cuts."""
    import time
    sod_mask = sod_binary(sod_map, threshold)
    names = ["avg", "gauss"] + [seam_name(l, s) for l in lams for s in sigmas]
    if not st.boxes:
        if info is not None:
            info.update(n_disagree=0, n_seam=0, seam_s={})
        return {k: sod_mask for k in names}
    box = st.box
    cov = st.covered()
    out = {}
    for name, lab in (("avg", st.avg()), ("gauss", st.gauss())):
        x = sod_mask.copy()
        x[box] = np.where(cov, lab, sod_mask[box])
        out[name] = x
    gauss = out["gauss"]
    dis = st.disagree()
    n_dis = int(dis.sum())
    seam = np.zeros(sod_mask.shape, bool)
    if n_dis and r_s > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_s + 1, 2 * r_s + 1))
        seam[box] = (cv2.dilate(dis.astype(np.uint8), k) > 0) & cov
    elif n_dis:
        seam[box] = dis
    n_seam = int(seam.sum())
    times = {}
    if n_seam == 0:
        for l in lams:
            for s in sigmas:
                out[seam_name(l, s)] = gauss
    else:
        # the seam's bounding box plus one fixed pixel of margin, in image coordinates
        y0, y1, x0, x1 = _bbox(seam, 1, seam.shape)
        sb = np.s_[y0:y1, x0:x1]
        u = np.zeros((y1 - y0, x1 - x0))
        uy0, ux0 = y0 - st.oy, x0 - st.ox            # the seam lies inside the crops' box
        un = st.unary()
        cy0, cx0 = max(uy0, 0), max(ux0, 0)
        cy1, cx1 = min(y1 - st.oy, un.shape[0]), min(x1 - st.ox, un.shape[1])
        u[cy0 - uy0:cy1 - uy0, cx0 - ux0:cx1 - ux0] = un[cy0:cy1, cx0:cx1]
        sq = color_sqdiff(img[sb])
        zeros = np.zeros(u.shape)
        for l in lams:
            for s in sigmas:
                t0 = time.time()
                w_r, w_d = color_weights(None, s, l, FRAC_UNIFORM, sq)
                tables = pairwise_tables(zeros, w_r, w_d, 0.0, "potts")
                x = gauss.copy()
                x[sb] = minimize_band(u, tables, seam[sb], gauss[sb])
                out[seam_name(l, s)] = x
                times[seam_name(l, s)] = time.time() - t0
    if info is not None:
        info.update(n_disagree=n_dis, n_seam=n_seam, seam_s=times,
                    covered_frac=float(cov.sum()) / sod_mask.size)
    return out
