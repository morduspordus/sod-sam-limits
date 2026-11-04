"""Native-resolution SAM crops along the boundary band of the SOD map, and the stitching of their masks
into per-pixel mask counts for the full-resolution graph cut (fusion_band.fuse_full).

On the high-resolution benchmarks SAM at 1K cannot resolve the thin structures. Instead, square crops of
CROP_SIZE pixels are placed on a grid along the band of radius band_radius(shape) around the boundary of
the thresholded SOD map (scripts/run_sam_crops.py prompts SAM with the SOD map of each crop), and the
three masks of every crop are reduced to a per-pixel count in [0, 3] (the mean over the crops covering
the pixel); outside the crops the count of the global 1K masks (upsampled) is used. The graph cut at full
resolution, not a hand rule, decides where crops overlap or disagree.
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .fusion import sod_binary

CROP_SIZE = 1024
CROP_STRIDE = 768
MIN_FRAC = 0.01       # a crop is kept when at least this fraction of its area is band
MAX_CROPS = 24
SUB = "sam_crops"     # output/<model>/<ds>/sam_crops/<id>_n.png (round(COUNT_SCALE * count)), <id>_cov.png, <id>.json
COUNT_SCALE = 64


def band_radius(shape):
    """Radius of the boundary band in pixels: max(32, 1.5% of the longest side)."""
    return max(32.0, 0.015 * max(shape[:2]))


def boundary_dist(mask):
    """Euclidean distance of every pixel to the boundary of a boolean mask (>= 1, as fusion2.prepare:
    pixels adjacent to the other label are at distance 1); inf without a boundary."""
    if mask.all() or not mask.any():
        return np.full(mask.shape, np.inf)
    return distance_transform_edt(mask) + distance_transform_edt(~mask)


def boundary_band(sod_map, r, threshold=127.0):
    """Boolean band of the pixels within distance r of the boundary of the thresholded SOD map
    (fusion.sod_binary: >= threshold on the max-normalized map); empty when the mask has no boundary."""
    return boundary_dist(sod_binary(sod_map, threshold)) <= r


def _starts(length, size, stride):
    """Crop starts along one axis: a grid of the given stride, the last crop shifted inward so that it is
    flush with the border; a single crop at 0 when the axis is shorter than the crop."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size, stride))
    starts.append(length - size)
    return sorted(set(starts))


def place_crops(band, size=CROP_SIZE, stride=CROP_STRIDE, min_frac=MIN_FRAC, max_crops=MAX_CROPS):
    """Axis-aligned square crops (clipped at the image border) on a grid of the given stride, keeping the
    ones whose band coverage is at least min_frac of their area; the max_crops best-covered ones if more
    qualify. Returns a list of (y0, x0, y1, x1) in row-major order; an empty band gives no crops."""
    h, w = band.shape
    if not band.any():
        return []
    # integral image of the band: the coverage of every crop in O(1)
    integral = np.zeros((h + 1, w + 1), np.int64)
    integral[1:, 1:] = band.cumsum(0).cumsum(1)
    cands = []
    for y0 in _starts(h, size, stride):
        y1 = min(h, y0 + size)
        for x0 in _starts(w, size, stride):
            x1 = min(w, x0 + size)
            n = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
            frac = n / ((y1 - y0) * (x1 - x0))
            if frac >= min_frac:
                cands.append((frac, (y0, x0, y1, x1)))
    if len(cands) > max_crops:
        cands = sorted(cands, key=lambda c: -c[0])[:max_crops]
    return sorted(box for _, box in cands)


def stitch_counts(shape, crops, crop_masks):
    """Per-pixel mean over the covering crops of the number of SAM masks containing the pixel
    (crop_masks[i]: (3, h_i, w_i) bool of crop i), as float32 in [0, 3], and the boolean covered map."""
    h, w = shape[:2]
    total = np.zeros((h, w), np.float32)
    cover = np.zeros((h, w), np.int32)
    for (y0, x0, y1, x1), m in zip(crops, crop_masks):
        total[y0:y1, x0:x1] += np.asarray(m).astype(np.float32).sum(0)
        cover[y0:y1, x0:x1] += 1
    covered = cover > 0
    count = np.divide(total, cover, out=np.zeros_like(total), where=covered)
    return count, covered


def global_counts(masks_1k, shape):
    """Count of the (3, h, w) 1K masks upsampled (nearest) to the full-resolution shape, float32."""
    n = np.asarray(masks_1k).astype(np.uint8).sum(0).astype(np.uint8)
    h, w = shape[:2]
    if n.shape != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_NEAREST)
    return n.astype(np.float32)


def combined_counts(crop_count, covered, global_count):
    """The crops' counts where covered, the global counts elsewhere."""
    return np.where(covered, crop_count, global_count).astype(np.float32)
