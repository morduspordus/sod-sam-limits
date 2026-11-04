"""The star-convexity cut (sodsam.connect): exact against brute force, the hard constraint holds, beta = 0 is
fusion_band.minimize_band, the geodesic parents form a forest rooted at the seeds, and a thin line broken
by a strong Potts term is restored by the hard constraint."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam.connect import (HARD, NO_PARENT, UNREACHABLE, fuse_full_star, minimize_band_star, star_parents,
                            strict_seeds, violations)
from sodsam.fusion import color_weights, energy, pairwise_tables
from sodsam.fusion_band import fuse_full, minimize_band

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fusion_band import random_problem


def random_parent(rng, h, w):
    """Arbitrary parents (any pixel, possibly fixed, possibly cyclic among free pixels), roots and
    unreachable pixels: the star term is submodular for any pairs."""
    parent = rng.integers(0, h * w, (h, w))
    r = rng.random((h, w))
    parent[r < 0.15] = NO_PARENT
    parent[(r >= 0.15) & (r < 0.25)] = UNREACHABLE
    same = parent.ravel() == np.arange(h * w)
    parent.ravel()[same] = NO_PARENT
    return parent


def star_cost(x, parent, free, beta):
    """beta * number of violated star terms (beta inf: inf if any)."""
    nv = violations(x, parent, free)
    if np.isinf(beta):
        return np.inf if nv else 0.0
    return beta * nv


def brute_force(unary1, tables, free, fixed, parent, beta):
    best, best_x = np.inf, None
    cells = np.flatnonzero(free)
    for bits in itertools.product((False, True), repeat=len(cells)):
        x = fixed.copy()
        x.ravel()[cells] = bits
        e = energy(x, unary1, tables) + star_cost(x, parent, free, beta)
        if e < best - 1e-9:
            best, best_x = e, x
    return best, best_x


def test_exact_vs_brute_force():
    rng = np.random.default_rng(0)
    n_active = 0
    for t in range(60):
        h, w = (3, 4) if t % 2 == 0 else tuple(rng.integers(2, 4, 2))
        unary1, tables = random_problem(rng, h, w)
        free = np.ones((h, w), bool) if t % 3 == 0 else rng.random((h, w)) < 0.8
        fixed = rng.random((h, w)) < 0.5
        parent = random_parent(rng, h, w)
        for beta in (0.0, rng.uniform(5, 80), HARD):
            x = minimize_band_star(unary1, tables, free, fixed, parent, beta)
            assert x.dtype == bool and np.array_equal(x[~free], fixed[~free])
            e = energy(x, unary1, tables) + star_cost(x, parent, free, beta)
            best, bx = brute_force(unary1, tables, free, fixed, parent, beta)
            assert abs(e - best) < 1e-6, (t, beta, e, best)
            if beta == HARD:
                assert violations(x, parent, free) == 0
                n_active += violations(minimize_band_star(unary1, tables, free, fixed, parent, 0.0),
                                       parent, free) > 0
    assert n_active > 5   # the constraint changed the solution on several problems


def test_beta_zero_is_minimize_band():
    rng = np.random.default_rng(1)
    for _ in range(50):
        h, w = rng.integers(2, 30, 2)
        unary1, tables = random_problem(rng, h, w)
        free = rng.random((h, w)) < rng.uniform(0.1, 1.0)
        fixed = rng.random((h, w)) < 0.5
        parent = random_parent(rng, h, w)
        assert np.array_equal(minimize_band_star(unary1, tables, free, fixed, parent, 0.0),
                              minimize_band(unary1, tables, free, fixed))


def test_hard_constraint_on_medium_problems():
    rng = np.random.default_rng(2)
    for _ in range(20):
        h, w = rng.integers(40, 80, 2)
        unary1, tables = random_problem(rng, h, w)
        unary1 *= 20
        free = rng.random((h, w)) < 0.9
        fixed = rng.random((h, w)) < 0.5
        s = rng.random((h, w))
        counts = rng.uniform(0, 3, (h, w))
        mode = ("loose", "interior", "strict")[_ % 3]
        parent = star_parents(s, counts, free, fixed, gamma=rng.uniform(0, 5), t_s=0.8, c_s=1.5, seed_mode=mode)
        info = {}
        x = minimize_band_star(unary1, tables, free, fixed, parent, HARD, info)
        assert violations(x, parent, free) == 0 and info["violations"] == 0
        # every foreground free pixel reaches a seed through foreground pixels (follow the parents)
        pf = parent.ravel()
        for p in np.flatnonzero(x.ravel() & free.ravel())[:200]:
            steps = 0
            while pf[p] >= 0 and free.ravel()[p]:
                p = pf[p]
                assert x.ravel()[p]
                steps += 1
                assert steps <= h * w
        # huge unaries: the bound of hard_beta, not BIG, keeps it hard
        x = minimize_band_star(unary1 * 1e5, tables, free, fixed, parent, HARD)
        assert violations(x, parent, free) == 0


def test_geodesic_parents():
    rng = np.random.default_rng(3)
    h, w = 30, 40
    free = np.zeros((h, w), bool)
    free[5:25, 5:35] = True
    fixed = np.zeros((h, w), bool)
    fixed[10:20, :5] = True                  # fixed foreground left of the band: seeds next to it
    s = np.full((h, w), 0.2)
    counts = np.zeros((h, w))
    s[22, 30], counts[22, 30] = 1.0, 3.0     # one band seed
    info = {}
    parent = star_parents(s, counts, free, fixed, gamma=0.0, info=info, seed_mode="loose")
    assert info["n_seed_band"] == 1 and info["n_seeds"] == 11 and info["n_unreachable"] == 0
    assert parent[22, 30] == NO_PARENT and (parent[~free] == NO_PARENT).all()
    pf = parent.ravel()
    for p in np.flatnonzero(free.ravel()):
        q = pf[p]
        assert q >= 0 or p == 22 * w + 30
        if q >= 0:   # a 4-neighbour
            assert abs(q // w - p // w) + abs(q % w - p % w) == 1
    # with gamma = 0 the path length is the L1 distance to the nearest seed: (15, 6) goes left
    assert pf[15 * w + 6] == 15 * w + 5 and pf[15 * w + 5] == 15 * w + 4
    # a band component cut off from all roots (no strict seeds of its own) is unconstrained, not forced
    free2 = free.copy()
    free2[:, 20] = False
    info = {}
    parent2 = star_parents(s, counts * 0, free2, fixed, info=info, seed_mode="loose")
    assert (parent2[free2 & (np.arange(w) > 20)[None]] == NO_PARENT).all() and (parent2 != UNREACHABLE).all()
    assert info["components"] == {"interior": 1, "band_seeds": 0, "own_strict": 0, "unconstrained": 1}
    assert info["n_unconstrained_px"] == (free2 & (np.arange(w) > 20)[None]).sum() and info["n_unreachable"] == 0
    # no seeds: no constraint
    for mode in ("loose", "interior", "strict"):
        info = {}
        assert (star_parents(s, counts * 0, free, fixed * False, info=info, seed_mode=mode) == NO_PARENT).all()
        assert info["seed_source"] == "unconstrained=1"


def detached_problem():
    """A fixed interior block on the left, a free band to its right holding a high-SOD blob (all crops agree,
    count 3) that the unary wants, separated from the interior by a strongly background gap. As on real data
    the fixed labels inside the band are the SOD mask: the blob and a low-confidence SOD-foreground patch
    (rows 2-3, cols 20-23) are True there; neither may be a root of the interior mode."""
    h, w = 30, 60
    free = np.zeros((h, w), bool)
    free[:, 10:] = True
    fixed = np.zeros((h, w), bool)
    fixed[5:25, :10] = True
    fixed[10:20, 40:50] = True
    fixed[2:4, 20:24] = True
    s = np.full((h, w), 0.05)
    counts = np.zeros((h, w))
    unary1 = np.full((h, w), 500.0)
    s[10:20, 40:50], counts[10:20, 40:50], unary1[10:20, 40:50] = 0.95, 3.0, -60.0
    img = np.zeros((h, w, 3), np.uint8)
    w_r, w_d = color_weights(img, 10.0, 5.0, 0.01)
    tables = pairwise_tables(np.zeros((h, w)), w_r, w_d, 0.0, "potts")
    return unary1, tables, free, fixed, s, counts


def test_interior_seeds_remove_detached_fragment():
    unary1, tables, free, fixed, s, counts = detached_problem()
    x0 = minimize_band(unary1, tables, free, fixed)
    assert x0[15, 45]                                          # the plain cut keeps the fragment
    for mode, kept in (("loose", True), ("strict", True), ("interior", False)):
        info = {}
        parent = star_parents(s, counts, free, fixed, info=info, seed_mode=mode)
        assert info["components"]["interior"] == 1 and info["components"]["unconstrained"] == 0
        assert info["n_seed_band"] == (0 if mode == "interior" else 100)
        # SOD-foreground band pixels are not roots (only the fixed interior outside the band is)
        assert parent[2, 21] >= 0 and parent[15, 45] == (NO_PARENT if kept else parent[15, 45])
        if mode == "interior":
            assert parent[15, 45] >= 0
        x = minimize_band_star(unary1, tables, free, fixed, parent, HARD, info)
        assert info["violations"] == 0
        assert x[15, 45] == kept, mode                         # interior: the detached fragment is removed
        assert np.array_equal(x[~free], fixed[~free])


def test_strict_area_and_empty_interior_fallback():
    unary1, tables, free, fixed, s, counts = detached_problem()
    # a tiny agreeing speck (< A_MIN_PX pixels) is not a strict seed; the 100 px blob is
    s[2, 30], counts[2, 30] = 1.0, 3.0
    seeds = strict_seeds(s, counts, free)
    assert seeds[15, 45] and seeds.sum() == 100 and not seeds[2, 30]
    # empty interior (the fixed labels are the SOD mask in the band only): the strict band seeds
    no_interior = fixed & free
    for mode, kind in (("interior", "own_strict"), ("strict", "band_seeds"), ("loose", "band_seeds")):
        info = {}
        parent = star_parents(s, counts, free, no_interior, info=info, seed_mode=mode)
        assert info["seed_source"] == f"{kind}=1", (mode, info["seed_source"])
        assert info["n_seeds"] == (101 if mode == "loose" else 100)   # loose: the speck is a seed too
        assert parent[15, 45] == NO_PARENT and parent[15, 51] >= 0 and parent[2, 21] >= 0
        assert (parent[2, 30] >= 0) == (mode != "loose")
        x = minimize_band_star(unary1, tables, free, no_interior, parent, HARD, info)
        assert x[15, 45] and info["violations"] == 0
    # ... and without strict seeds there is no constraint: the plain cut
    info = {}
    parent = star_parents(s, counts * 0, free, no_interior, info=info, seed_mode="interior")
    assert info["seed_source"] == "unconstrained=1" and (parent == NO_PARENT).all()
    assert np.array_equal(minimize_band_star(unary1, tables, free, no_interior, parent, HARD),
                          minimize_band(unary1, tables, free, no_interior))


def test_strict_area_threshold_and_connectivity():
    """A_min = max(16, 1e-4 * 1500) = 16 px: a region of exactly 16 px is kept, 15 px dropped; regions are
    4-connected, so two 9-px specks touching only diagonally (18 px as one 8-connected region) are dropped."""
    h, w = 30, 60
    free = np.zeros((h, w), bool)
    free[:, 10:] = True
    s = np.zeros((h, w))
    counts = np.zeros((h, w))

    def put(y0, y1, x0, x1):
        s[y0:y1, x0:x1], counts[y0:y1, x0:x1] = 1.0, 3.0
    put(2, 6, 12, 16)          # 4 x 4 = 16: kept
    put(10, 13, 12, 17)        # 3 x 5 = 15: dropped
    put(20, 23, 20, 23)        # 3 x 3 ...
    put(23, 26, 23, 26)        # ... and 3 x 3 touching it at one corner: dropped
    seeds = strict_seeds(s, counts, free)
    assert seeds[2:6, 12:16].all() and seeds.sum() == 16


def test_thin_part_without_interior_per_component():
    """An object with an interior on the left and a thin SOD part (a separate band component, no interior of
    its own) on the right. (i) without strict seeds of its own the part is unconstrained: the hard cut equals
    the plain cut there (the old global fallback forced it to background); (ii) with its own strict seeds it
    is constrained to them: a fragment of it detached by a strongly background gap is removed."""
    h, w = 40, 80
    free = np.zeros((h, w), bool)
    free[:, 10:30] = True                    # the band around the object with an interior
    free[15:25, 45:75] = True                # the band of the thin part: a separate component
    fixed = np.zeros((h, w), bool)
    fixed[5:35, :10] = True                  # the interior
    s = np.full((h, w), 0.05)
    counts = np.zeros((h, w))
    unary1 = np.full((h, w), 500.0)
    unary1[19:21, 47:73] = -40.0             # the thin part: a 2-px line the unary wants
    fixed[19:21, 47:73] = True               # SOD foreground in the band
    unary1[19:21, 58:62] = 500.0             # ... broken by a strongly background gap
    img = np.zeros((h, w, 3), np.uint8)
    w_r, w_d = color_weights(img, 10.0, 2.0, 0.01)
    tables = pairwise_tables(np.zeros((h, w)), w_r, w_d, 0.0, "potts")
    x0 = minimize_band(unary1, tables, free, fixed)
    assert x0[19:21, 47:58].all() and x0[19:21, 62:73].all()
    part = np.zeros((h, w), bool)
    part[15:25, 45:75] = True
    for mode in ("interior", "strict", "loose"):
        # (i) no strict seeds in the thin part
        info = {}
        parent = star_parents(s, counts, free, fixed, info=info, seed_mode=mode)
        assert info["components"] == {"interior": 1, "band_seeds": 0, "own_strict": 0, "unconstrained": 1}
        assert (parent[part] == NO_PARENT).all() and info["n_unconstrained_px"] == part.sum()
        x = minimize_band_star(unary1, tables, free, fixed, parent, HARD, info)
        assert np.array_equal(x[part], x0[part]) and info["violations"] == 0
        # (ii) strict seeds (4 x 4) on the left half of the line: the right half, beyond the gap, is removed
        s2, c2 = s.copy(), counts.copy()
        s2[18:22, 48:52], c2[18:22, 48:52] = 1.0, 3.0
        u2 = unary1.copy()
        u2[18:22, 48:52] = -40.0
        info = {}
        parent = star_parents(s2, c2, free, fixed, info=info, seed_mode=mode)
        kind = "own_strict" if mode == "interior" else "band_seeds"
        assert info["components"][kind] == 1 and info["components"]["unconstrained"] == 0
        assert parent[18, 48] == NO_PARENT and parent[19, 70] >= 0
        x = minimize_band_star(u2, tables, free, fixed, parent, HARD, info)
        assert info["violations"] == 0
        assert x[19:21, 48:58].all() and not x[19:21, 62:73].any(), mode


def test_own_strict_seeds_only_in_rootless_components():
    """Interior mode: a strict blob inside an interior-touching component is NOT a root (own strict seeds are
    given only to rootless components), so the hard cut removes the blob, which is cut off from the interior
    by a strongly background gap; a seedless thin part elsewhere stays unconstrained. In strict mode the same
    blob is a band seed and is kept."""
    h, w = 40, 80
    free = np.zeros((h, w), bool)
    free[:, 10:30] = True                    # band of the object with an interior (one component)
    free[15:25, 45:75] = True                # a rootless thin part (separate component), no strict seeds
    fixed = np.zeros((h, w), bool)
    fixed[5:35, :10] = True                  # the interior
    s = np.full((h, w), 0.05)
    counts = np.zeros((h, w))
    unary1 = np.full((h, w), 500.0)
    unary1[15:20, 22:27] = -40.0             # the blob the unary wants, 12+ px of background away from col 10
    s[15:20, 22:27], counts[15:20, 22:27] = 1.0, 3.0      # 25 px of strict seed (>= A_min = 16)
    fixed[15:20, 22:27] = True               # SOD foreground in the band
    img = np.zeros((h, w, 3), np.uint8)
    w_r, w_d = color_weights(img, 10.0, 2.0, 0.01)
    tables = pairwise_tables(np.zeros((h, w)), w_r, w_d, 0.0, "potts")
    x0 = minimize_band(unary1, tables, free, fixed)
    assert x0[15:20, 22:27].all()
    info = {}
    parent = star_parents(s, counts, free, fixed, info=info, seed_mode="interior")
    assert info["components"] == {"interior": 1, "band_seeds": 0, "own_strict": 0, "unconstrained": 1}
    assert parent[17, 24] >= 0                       # the blob is not a root in interior mode
    x = minimize_band_star(unary1, tables, free, fixed, parent, HARD, info)
    assert info["violations"] == 0 and not x[15:20, 22:27].any()
    info = {}
    parent = star_parents(s, counts, free, fixed, info=info, seed_mode="strict")
    x = minimize_band_star(unary1, tables, free, fixed, parent, HARD, info)
    assert info["violations"] == 0 and x[15:20, 22:27].all()


def test_own_strict_area_threshold_uses_the_whole_band():
    """A_min for own strict seeds is max(16, 1e-4 * the whole band): with a 200k-pixel band it is 20, so an
    18-pixel own-seed region in a small rootless component is dropped (the component stays unconstrained),
    and a 24-pixel one is kept."""
    h, w = 520, 440
    free = np.zeros((h, w), bool)
    free[:500, 10:410] = True                # a 200,000-pixel band touching the interior
    free[505:515, 415:435] = True            # a small rootless component (200 px)
    fixed = np.zeros((h, w), bool)
    fixed[:500, :10] = True                  # the interior
    s = np.full((h, w), 0.05)
    counts = np.zeros((h, w))
    s[507:510, 420:426], counts[507:510, 420:426] = 1.0, 3.0   # 18 px: >= 16 but < 20
    info = {}
    star_parents(s, counts, free, fixed, info=info, seed_mode="interior")
    assert info["components"]["own_strict"] == 0 and info["components"]["unconstrained"] == 1
    s[507:511, 420:426], counts[507:511, 420:426] = 1.0, 3.0   # 24 px: kept
    info = {}
    star_parents(s, counts, free, fixed, info=info, seed_mode="interior")
    assert info["components"]["own_strict"] == 1 and info["components"]["unconstrained"] == 0


def test_thin_line_restored():
    """A fixed foreground block on the left, a free thin line (1 px) to a free blob on the right that the
    unary wants. The Potts term breaks the line (its two long sides cost more than its unary gain), so the
    unconstrained cut keeps a disconnected blob; with the block as the seed and the geodesic parents along
    the (high SOD) line, the hard constraint keeps the blob connected by restoring the line."""
    h, w = 21, 60
    free = np.zeros((h, w), bool)
    free[:, 10:] = True
    fixed = np.zeros((h, w), bool)
    fixed[5:16, :10] = True
    s = np.full((h, w), 0.05)
    s[10, 10:50] = 0.95
    s[7:14, 50:57] = 0.95
    unary1 = np.full((h, w), 30.0)
    unary1[10, 10:50] = -2.0                # weakly foreground line
    unary1[7:14, 50:57] = -60.0             # strongly foreground blob
    img = np.zeros((h, w, 3), np.uint8)
    w_r, w_d = color_weights(img, 10.0, 10.0, 0.01)
    tables = pairwise_tables(np.zeros((h, w)), w_r, w_d, 0.0, "potts")
    x0 = minimize_band(unary1, tables, free, fixed)
    assert x0[10, 52] and not x0[10, 30:45].any()            # blob kept, line broken
    counts = np.zeros((h, w))                                  # no band seeds: only the fixed block
    parent = star_parents(s, counts, free, fixed, gamma=20.0)
    x = minimize_band_star(unary1, tables, free, fixed, parent, HARD)
    assert x[10, 10:50].all() and x[10, 52]                    # line restored, blob kept
    assert violations(x, parent, free) == 0
    from scipy.ndimage import label
    lab, _ = label(x)
    assert lab[10, 52] == lab[10, 0]                           # connected to the seed
    # a small finite beta does not restore it, a large one does
    assert not minimize_band_star(unary1, tables, free, fixed, parent, 0.01)[10, 30:45].any()
    assert minimize_band_star(unary1, tables, free, fixed, parent, 1000.0)[10, 10:50].all()


def test_fuse_full_star_beta_zero_is_fuse_full():
    rng = np.random.default_rng(4)
    th = {"a": 30.0, "b": 80.0, "d": 40.0, "sigma": 15.0, "lam": 25.0, "c_neg": -5.0}
    for _ in range(5):
        h, w = 60, 80
        img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
        sod = np.zeros((h, w), np.uint8)
        sod[15:45, 20:60] = rng.integers(100, 256, (30, 40))
        sod[22:38, 28:52] = 255                     # a solid interior (distance > 6 from the boundary)
        counts = rng.uniform(0, 3, (h, w)).astype(np.float32)
        ref = fuse_full(img, sod, counts, th, 6.0)
        assert np.array_equal(fuse_full_star(img, sod, counts, th, 6.0, 0.0, gamma=4.0), ref)
        info = {}
        hard = fuse_full_star(img, sod, counts, th, 6.0, HARD, gamma=4.0, info=info)
        assert hard.dtype == bool and info["n_seeds"] > 0


if __name__ == "__main__":
    test_exact_vs_brute_force()
    test_beta_zero_is_minimize_band()
    test_hard_constraint_on_medium_problems()
    test_geodesic_parents()
    test_thin_line_restored()
    test_fuse_full_star_beta_zero_is_fuse_full()
    test_interior_seeds_remove_detached_fragment()
    test_strict_area_and_empty_interior_fallback()
    test_strict_area_threshold_and_connectivity()
    test_thin_part_without_interior_per_component()
    print("all tests passed")
