"""Boundary band, crop placement and the stitching of crop masks (sodsam.crops) on synthetic data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam.crops import (band_radius, boundary_band, combined_counts, global_counts, place_crops,
                          stitch_counts)


def square_map(h, w, y0, y1, x0, x1, value=255):
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = value
    return m


def test_band_is_around_the_boundary():
    sod = square_map(200, 300, 50, 150, 100, 220)
    band = boundary_band(sod, 10)
    assert band.dtype == bool and band.shape == sod.shape
    assert band[50, 160] and band[149, 160] and band[100, 100] and band[100, 219]   # first rows/cols inside
    assert band[40, 160] and band[159, 160] and band[100, 90] and band[100, 229]   # 10 outside
    assert not band[39, 160] and not band[100, 230]                                   # 11 outside
    assert band[59, 160] and not band[60, 160]                                         # 10 / 11 inside
    assert not band[100, 160] and not band[0, 0]
    # a soft map is thresholded at 127/255 of its max (sod_binary): 90 of max 200 (= 115) is background
    sod[:, :] = np.where(sod > 0, 200, 90)
    assert np.array_equal(boundary_band(sod, 10), band)
    assert not boundary_band(np.zeros((50, 60), np.uint8), 10).any()
    assert not boundary_band(np.full((50, 60), 255, np.uint8), 10).any()
    assert band_radius((6000, 4000)) == 90.0 and band_radius((1000, 800)) == 32.0


def test_crops_cover_the_band():
    rng = np.random.default_rng(0)
    for _ in range(10):
        h, w = rng.integers(1200, 4000, 2)
        sod = np.zeros((h, w), np.uint8)
        y0, x0 = rng.integers(0, h // 2), rng.integers(0, w // 2)
        sod[y0:y0 + rng.integers(100, h // 2), x0:x0 + rng.integers(100, w // 2)] = 255
        band = boundary_band(sod, 40)
        crops = place_crops(band, 1024, 768, 0.01, 1000)
        covered = np.zeros((h, w), bool)
        for cy0, cx0, cy1, cx1 in crops:
            assert 0 <= cy0 < cy1 <= h and 0 <= cx0 < cx1 <= w
            assert cy1 - cy0 == min(1024, h) and cx1 - cx0 == min(1024, w)   # square, shifted inward
            covered[cy0:cy1, cx0:cx1] = True
        # every band pixel lies in some crop of the stride-768 grid; those crops hold >= 1% band, except
        # crops that contain only a few band pixels, so check the bulk of the band
        assert (band & covered).sum() >= 0.98 * band.sum()
        # a crop with less band than min_frac is dropped
        for cy0, cx0, cy1, cx1 in crops:
            assert band[cy0:cy1, cx0:cx1].mean() >= 0.01
    # the max_crops best-covered ones are kept, in row-major order
    sod = square_map(3000, 3000, 500, 2500, 500, 2500)
    band = boundary_band(sod, 40)
    many = place_crops(band, 1024, 768, 0.01, 1000)
    few = place_crops(band, 1024, 768, 0.01, 4)
    assert len(many) > 4 and len(few) == 4 and few == sorted(few)
    cov = {b: band[b[0]:b[2], b[1]:b[3]].mean() for b in many}
    assert min(cov[b] for b in few) >= max(cov[b] for b in many if b not in few)
    assert place_crops(np.zeros((3000, 3000), bool)) == []


def test_small_image_gives_one_whole_crop():
    sod = square_map(600, 900, 100, 400, 200, 700)
    band = boundary_band(sod, 20)
    assert place_crops(band, 1024, 768, 0.01, 24) == [(0, 0, 600, 900)]
    # exactly the crop size: one crop too
    sod = square_map(1024, 1024, 100, 400, 200, 700)
    assert place_crops(boundary_band(sod, 20)) == [(0, 0, 1024, 1024)]
    # one side longer: the last crop is flush with the border
    sod = square_map(600, 1500, 100, 400, 200, 1400)
    crops = place_crops(boundary_band(sod, 20), 1024, 768, 0.01, 24)
    assert crops == [(0, 0, 600, 1024), (0, 476, 600, 1500)]


def test_overlapping_crops_average():
    shape = (40, 60)
    crops = [(0, 0, 30, 30), (10, 10, 40, 40)]
    m1 = np.zeros((3, 30, 30), bool)
    m1[:, :, :] = True                   # all three masks everywhere: count 3
    m2 = np.zeros((3, 30, 30), bool)
    m2[0] = True                         # one mask: count 1
    count, covered = stitch_counts(shape, crops, [m1, m2])
    assert count.dtype == np.float32 and covered.dtype == bool
    assert count[5, 5] == 3 and count[35, 35] == 1 and count[20, 20] == 2   # overlap: mean of 3 and 1
    assert covered[5, 5] and covered[35, 35] and not covered[5, 50] and count[5, 50] == 0
    assert count.min() >= 0 and count.max() <= 3
    masks_1k = np.zeros((3, 4, 6), bool)
    masks_1k[0, :, 3:] = True
    masks_1k[1, :, 3:] = True
    g = global_counts(masks_1k, shape)
    assert g.shape == shape and g[0, 0] == 0 and g[0, 59] == 2 and g[20, 29] == 0 and g[20, 30] == 2
    comb = combined_counts(count, covered, g)
    assert comb[5, 5] == 3 and comb[5, 50] == 2 and comb[35, 35] == 1 and comb.dtype == np.float32
    # no crops: nothing covered, the global counts everywhere
    count0, covered0 = stitch_counts(shape, [], [])
    assert not covered0.any() and np.array_equal(combined_counts(count0, covered0, g), g)


if __name__ == "__main__":
    test_band_is_around_the_boundary()
    test_crops_cover_the_band()
    test_small_image_gives_one_whole_crop()
    test_overlapping_crops_average()
    print("all tests passed")
