"""SAM2 with the prompts used in the paper.

Prompts:
  mask_soft    the soft SOD map (0..255), resized to 256x256 with nearest neighbour and passed
               as mask_input logits, multimask output. This is what the original script
               apply_sam_with_mask_all_masks.py did.
  mask_binary  the thresholded SOD mask as +-logit mask_input (what the paper describes).
  box_sod      bounding box of the SOD pixels with saliency > 0 (apply_sam_with_box.py).
  box_margin   whole image shrunk by a margin in pixels, single-mask output.
  center       center point, single-mask output.
SAM2 resizes the image to 1024x1024 without padding, so the 256x256 mask_input covers
the whole image.
"""
import os

import numpy as np
import torch
from PIL import Image

SAM2_ROOT = os.environ.get("SAM2_ROOT", "third_party/sam2")
SAM2_CKPT = os.environ.get("SAM2_CKPT", os.path.join(SAM2_ROOT, "checkpoints", "sam2.1_hiera_large.pt"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_l.yaml")
BINARY_LOGIT = 10.0


class SamRunner:
    def __init__(self, checkpoint=SAM2_CKPT, cfg=SAM2_CFG, device="cuda"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = device
        self.predictor = SAM2ImagePredictor(build_sam2(cfg, checkpoint, device=device))

    def set_image(self, img):
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(img)

    def _predict(self, **kw):
        """Masks (bool), scores and mask probabilities (sigmoid of the full-resolution logits)."""
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            logits, scores, _ = self.predictor.predict(return_logits=True, **kw)
        logits = np.asarray(logits, dtype=np.float32)
        # the predictor thresholds logits at mask_threshold = 0 when return_logits=False
        return logits > 0, np.asarray(scores, dtype=np.float64), 1 / (1 + np.exp(-logits))

    def from_mask(self, sod_map, mode="soft", threshold=127.0):
        """Three masks (3,H,W bool), scores and probabilities for an SOD map prompt; call set_image first.

        mode: soft (the 0..255 map as logits, original script), binary (thresholded map as +-10),
        logit (calibrated: 2 * logit of the max-normalized map clipped to [0.01, 0.99]).
        """
        m = np.array(Image.fromarray(sod_map.astype(np.uint8)).resize((256, 256), Image.NEAREST))
        s = m.astype(np.float64) / (sod_map.max() + 1e-13)   # max-normalized, in [0, 1]
        if mode == "soft":
            prompt = m.astype(np.float32)
        elif mode == "binary":
            prompt = np.where(255.0 * s >= threshold, BINARY_LOGIT, -BINARY_LOGIT).astype(np.float32)
        elif mode == "logit":
            s = np.clip(s, 0.01, 0.99)
            prompt = (2 * np.log(s / (1 - s))).astype(np.float32)
        else:
            raise ValueError(mode)
        return self._predict(mask_input=prompt[None], multimask_output=True)

    def from_mask_and_box(self, sod_map, threshold=127.0):
        """Soft mask prompt together with the box of the thresholded SOD mask."""
        m = np.array(Image.fromarray(sod_map.astype(np.uint8)).resize((256, 256), Image.NEAREST))
        return self._predict(mask_input=m.astype(np.float32)[None],
                             box=np.asarray(sod_box(sod_map, threshold), dtype=np.float32),
                             multimask_output=True)

    def from_components(self, sod_map, threshold=127.0, min_frac=0.01, max_comp=4):
        """Per-component prompting: the soft map restricted to each connected component of the
        thresholded SOD mask (dilated by 2% of the image size) is a separate prompt; mask k of the
        result is the union of mask k over the components, the probability the maximum.
        With a single component this equals the soft mask prompt."""
        import cv2
        h, w = sod_map.shape
        binary = (255.0 * sod_map / (sod_map.max() + 1e-13) >= threshold).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        comps = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= min_frac * h * w]
        comps = sorted(comps, key=lambda c: -stats[c, cv2.CC_STAT_AREA])[:max_comp]
        if len(comps) <= 1:
            return self.from_mask(sod_map, "soft")
        r = max(1, int(0.02 * max(h, w)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        masks, probs, scores = None, None, []
        for c in comps:
            region = cv2.dilate((lab == c).astype(np.uint8), kernel) > 0
            mk, sc, pr = self.from_mask(np.where(region, sod_map, 0).astype(np.uint8), "soft")
            masks = mk if masks is None else masks | mk
            probs = pr if probs is None else np.maximum(probs, pr)
            scores.append(sc)
        return masks, np.mean(scores, axis=0), probs

    def from_morph(self, sod_map, op, frac=0.02):
        """Soft mask prompt after grayscale erosion or dilation by 2% of the image size."""
        import cv2
        r = max(1, int(frac * max(sod_map.shape)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        f = cv2.erode if op == "erode" else cv2.dilate
        return self.from_mask(f(sod_map.astype(np.uint8), kernel), "soft")

    def from_zoom(self, img, sod_map, threshold=127.0, margin=0.25, max_frac=0.3):
        """Zoom-in prompting: the image is cropped to the box of the thresholded SOD map enlarged by
        `margin` of its size, encoded again, prompted with the soft map of the crop, and the masks are
        pasted back (background outside the crop). Falls back to the soft prompt on the whole image
        when the box covers more than max_frac of it. Leaves the predictor on the crop."""
        h, w = sod_map.shape
        x0, y0, x1, y1 = sod_box(sod_map, threshold)
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw * bh > max_frac * h * w or min(bw, bh) < 8:
            self.set_image(img)
            return self.from_mask(sod_map, "soft")
        mx, my = int(margin * bw), int(margin * bh)
        X0, Y0, X1, Y1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + 1 + mx), min(h, y1 + 1 + my)
        self.set_image(np.ascontiguousarray(img[Y0:Y1, X0:X1]))
        mk, sc, pr = self.from_mask(sod_map[Y0:Y1, X0:X1], "soft")
        masks, probs = np.zeros((3, h, w), bool), np.zeros((3, h, w), np.float32)
        masks[:, Y0:Y1, X0:X1], probs[:, Y0:Y1, X0:X1] = mk, pr
        return masks, sc, probs

    def from_box(self, box, multimask=False):
        return self._predict(box=np.asarray(box, dtype=np.float32), multimask_output=multimask)

    def box_logits(self, box, multimask=False):
        """Full-resolution logits (K,H,W float32, K = 3 or 1) and scores for an xyxy box prompt."""
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            logits, scores, _ = self.predictor.predict(box=np.asarray(box, dtype=np.float32),
                                                       multimask_output=multimask, return_logits=True)
        return np.asarray(logits, dtype=np.float32), np.asarray(scores, dtype=np.float64)

    def from_point(self, xy, multimask=False):
        return self._predict(point_coords=np.array([xy], dtype=np.float32),
                             point_labels=np.array([1]), multimask_output=multimask)


def sod_box(sod_map, threshold=None):
    """xyxy box around pixels with saliency > 0 (or, with a threshold, of the max-normalized map
    >= threshold); the whole image if there are none."""
    if threshold is None:
        ys, xs = np.nonzero(sod_map > 0)
    else:
        ys, xs = np.nonzero(255.0 * sod_map / (sod_map.max() + 1e-13) >= threshold)
    h, w = sod_map.shape
    if xs.size == 0:
        return [0, 0, w, h]
    return [xs.min(), ys.min(), xs.max(), ys.max()]


def margin_box(h, w, margin):
    return [margin, margin, w - margin, h - margin]
