"""Guided filter (He, Sun & Tang 2013), gray guide, for a matting-like boundary on a fused map."""
import cv2
import numpy as np


def guided_filter(img, src, radius_frac=0.02, eps=1e-3):
    """src: float map in [0, 1]; img: RGB uint8 guide. Window radius = radius_frac * max(H, W)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = src.astype(np.float32)
    r = max(1, int(round(radius_frac * max(src.shape))))
    box = lambda a: cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)
    mean_g, mean_p = box(g), box(p)
    var_g = box(g * g) - mean_g * mean_g
    cov_gp = box(g * p) - mean_g * mean_p
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    return np.clip(box(a) * g + box(b), 0.0, 1.0)
