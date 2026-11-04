"""Graph-cut result equals the brute-force minimum on small random problems."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam.fusion import FusionParams, color_weights, energy, fuse, minimize, pairwise_tables, unary_term


def brute_force(unary1, tables):
    h, w = unary1.shape
    best = None
    for bits in itertools.product((0, 1), repeat=h * w):
        x = np.array(bits).reshape(h, w)
        e = energy(x, unary1, tables)
        best = e if best is None or e < best else best
    return best


def random_problem(rng, h=3, w=4):
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    sod = rng.integers(0, 256, (h, w)).astype(np.uint8)
    sam = [rng.random((h, w)) < 0.5 for _ in range(3)]
    p = FusionParams(unary=rng.uniform(-50, 50, (2, 4)).tolist(), sigma=rng.uniform(5, 50),
                     lam=rng.uniform(0, 30), c_neg=rng.uniform(-30, 0))
    return img, sod, sam, p


def test_graph_cut_is_exact():
    rng = np.random.default_rng(0)
    for _ in range(200):
        img, sod, sam, p = random_problem(rng)
        sod_mask = sod >= 127
        unary1, m_sam = unary_term(sod_mask, sam, p)
        for variant in ("full", "potts"):
            tables = pairwise_tables(m_sam, *color_weights(img, p.sigma, p.lam, p.frac_uniform),
                                     p.c_neg, variant, "paper")
            x, n_clipped = minimize(unary1, tables)
            assert n_clipped == 0
            assert abs(energy(x, unary1, tables) - brute_force(unary1, tables)) < 1e-6


def test_repulsive_term_is_modular():
    # v(0,0) + v(1,1) == v(0,1) + v(1,0) for every repulsive pair
    m_sam = np.array([[0, 3, 1, 3, 0]])
    w = np.full((1, 4), 7.0)
    (b, c), _ = pairwise_tables(np.vstack([m_sam, m_sam]), np.vstack([w, w]), np.full((1, 5), 7.0),
                                -10.0, "full", "paper")
    rep = np.abs(m_sam[:, :-1] - m_sam[:, 1:]) >= 2
    assert np.allclose((b + c)[rep[[0, 0]]], 0)


def test_unary_variant_matches_sign():
    rng = np.random.default_rng(1)
    img, sod, sam, p = random_problem(rng, 5, 6)
    x = fuse(img, sod, sam, p, "unary")
    s = sod.astype(np.float64)
    unary1, _ = unary_term(255 * s / (s.max() + 1e-13) >= p.sod_threshold, sam, p)
    assert np.array_equal(x, unary1 < 0)


if __name__ == "__main__":
    test_graph_cut_is_exact()
    test_repulsive_term_is_modular()
    test_unary_variant_matches_sign()
    print("all tests passed")
