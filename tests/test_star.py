"""Exactness of the star-prior graph cut (sodsam.star) against brute force, and the star-shape property."""
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import star


def _problem(rng, h, w):
    logit = rng.normal(0, 4, (h, w))
    img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
    x0, y0 = int(rng.integers(0, 2)), int(rng.integers(0, 2))
    x1, y1 = int(rng.integers(w - 2, w)), int(rng.integers(h - 2, h))
    box = (x0, y0, x1, y1)
    centre = (int(rng.integers(x0, x1 + 1)), int(rng.integers(y0, y1 + 1)))
    th = {"L": float(rng.uniform(1, 8)), "lam": float(rng.uniform(0, 4)), "sigma": float(rng.uniform(5, 60)),
          "frac_uniform": 0.01}
    return logit, img, box, centre, th


def _brute(logit, img, box, th, centre, beta):
    free = star.free_mask(logit.shape, box)
    idx = np.flatnonzero(free)
    best = math.inf
    for bits in itertools.product((0, 1), repeat=idx.size):
        x = np.zeros(logit.size, bool)
        x[idx] = bits
        best = min(best, star.energy(x.reshape(logit.shape), logit, img, box, th, centre, beta))
    return best


def test_brute_force_exact():
    rng = np.random.default_rng(0)
    for trial in range(60):
        h, w = (3, 4) if trial % 2 else (4, 3)
        logit, img, box, centre, th = _problem(rng, h, w)
        for beta in (0.0, float(rng.uniform(0.1, 6)), star.HARD):
            x = star.cut_star(logit, img, box, th, centre, beta)
            e = star.energy(x, logit, img, box, th, centre, beta)
            b = _brute(logit, img, box, th, centre, beta)
            assert math.isfinite(e) and abs(e - b) < 1e-6, (trial, beta, e, b)


def test_cached_inputs_match():
    rng = np.random.default_rng(1)
    for _ in range(10):
        logit, img, box, centre, th = _problem(rng, 9, 11)
        sq = star.color_sqdiff(img)
        steps = star.star_steps(logit.shape, centre)
        for beta in (0.0, 2.0, star.HARD):
            a = star.cut_star(logit, img, box, th, centre, beta)
            b = star.cut_star(logit, img, box, th, centre, beta, sqdiff=sq, steps=steps)
            assert (a == b).all()


def test_steps_closer_and_in_bbox():
    rng = np.random.default_rng(2)
    for _ in range(20):
        h, w = int(rng.integers(5, 60)), int(rng.integers(5, 60))
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        dy, dx = star.star_steps((h, w), c)
        yy, xx = np.mgrid[:h, :w]
        not_c = (yy != c[1]) | (xx != c[0])
        assert ((dy == 0) & (dx == 0) == ~not_c).all()
        d0 = (yy - c[1]) ** 2 + (xx - c[0]) ** 2
        d1 = (yy + dy - c[1]) ** 2 + (xx + dx - c[0]) ** 2
        assert (d1[not_c] < d0[not_c]).all()
        # the step stays in the bounding box of p and c
        qy, qx = yy + dy, xx + dx
        assert ((qy - yy) * (qy - c[1]) <= 0).all() and ((qx - xx) * (qx - c[0]) <= 0).all()


def test_chain_follows_segment():
    """The discrete segment (chain of steps) stays within 0.25 |p - c| (+1 pixel) of the continuous segment
    p-c (measured: 0.2 |p - c|, ~6 pixels at |p - c| ~ 50; inherent to a chain where each pixel re-aims at c)."""
    worst, c = 0.0, (40, 33)
    dy, dx = star.star_steps((80, 90), c)
    for y in range(0, 80, 3):
        for x in range(0, 90, 3):
            vx, vy = c[0] - x, c[1] - y
            n = math.hypot(vx, vy)
            for (qy, qx) in star.segment_chain((dy, dx), (y, x)):
                if n:
                    worst = max(worst, abs(vx * (qy - y) - vy * (qx - x)) / n - 0.25 * n)
    assert worst < 1.0, worst


def test_hard_prior_star_shaped():
    rng = np.random.default_rng(3)
    import cv2
    n_nonempty = n_active = 0
    for trial in range(40):
        h, w = int(rng.integers(15, 50)), int(rng.integers(15, 50))
        logit = cv2.GaussianBlur(rng.normal(0, 6, (h, w)), (0, 0), 2.0) * 4 + 1.0
        img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
        box = (1, 2, w - 2, h - 1)
        centre = (int(rng.integers(1, w - 1)), int(rng.integers(2, h)))
        th = {"L": 8.0, "lam": float(rng.uniform(0, 2)), "sigma": 20.0}
        x = star.cut_star(logit, img, box, th, centre, star.HARD)
        steps = star.star_steps(x.shape, centre)
        ys, xs = np.nonzero(x)
        n_nonempty += ys.size > 0
        for p in zip(ys, xs):
            assert all(x[q] for q in star.segment_chain(steps, (int(p[0]), int(p[1]))))
        # without the prior the solution is no worse in the prior-free energy, and it differs in many trials
        x0 = star.cut_star(logit, img, box, th, centre, 0.0)
        assert star.energy(x0, logit, img, box, th, centre, 0.0) <= star.energy(x, logit, img, box, th, centre, 0.0) + 1e-9
        n_active += bool((x != x0).any())
    assert n_nonempty >= 20
    assert n_active >= 10, n_active


def _multi_energy(x, logit, img, box, th, steps, beta):
    """Energy with the multi-centre star term, from the definition (prior-free part via star.energy)."""
    e = star.energy(x, logit, img, box, th, (0, 0), 0.0)
    if math.isinf(e) or beta == 0:
        return e
    dy, dx = steps
    free = star.free_mask(x.shape, box)
    ys, xs = np.nonzero(free & ((dy != 0) | (dx != 0)))
    viol = int((x[ys, xs] & ~x[ys + dy[ys, xs], xs + dx[ys, xs]]).sum())
    return e if not viol else (math.inf if math.isinf(beta) else e + beta * viol)


def test_multi_steps_terminate():
    rng = np.random.default_rng(4)
    for _ in range(20):
        h, w = int(rng.integers(5, 50)), int(rng.integers(5, 50))
        cs = [(int(rng.integers(0, w)), int(rng.integers(0, h))) for _ in range(int(rng.integers(1, 5)))]
        dy, dx = star.star_steps_multi((h, w), cs)
        yy, xx = np.mgrid[:h, :w]
        dmin = lambda y, x: np.min([(y - cy) ** 2 + (x - cx) ** 2 for cx, cy in cs], axis=0)
        moving = (dy != 0) | (dx != 0)
        assert (dmin(yy, xx)[~moving] == 0).all()                       # only the centres stay
        assert (dmin(yy + dy, xx + dx)[moving] < dmin(yy, xx)[moving]).all()
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                q = star.segment_chain((dy, dx), (y, x))[-1]
                assert (q[1], q[0]) in cs


def test_multi_brute_force_and_hard():
    rng = np.random.default_rng(5)
    for trial in range(40):
        h, w = (3, 4) if trial % 2 else (4, 3)
        logit, img, box, centre, th = _problem(rng, h, w)
        x0, y0, x1, y1 = box
        cs = [centre, (int(rng.integers(x0, x1 + 1)), int(rng.integers(y0, y1 + 1)))]
        steps = star.star_steps_multi(logit.shape, cs)
        for beta in (0.0, float(rng.uniform(0.1, 6)), star.HARD):
            x = star.cut_star(logit, img, box, th, cs[0], beta, steps=steps)
            e = _multi_energy(x, logit, img, box, th, steps, beta)
            free = star.free_mask(logit.shape, box)
            idx = np.flatnonzero(free)
            best = math.inf
            for bits in itertools.product((0, 1), repeat=idx.size):
                z = np.zeros(logit.size, bool)
                z[idx] = bits
                best = min(best, _multi_energy(z.reshape(logit.shape), logit, img, box, th, steps, beta))
            assert math.isfinite(e) and abs(e - best) < 1e-6, (trial, beta, e, best)


def test_component_centres():
    m = np.zeros((40, 60), bool)
    m[5:20, 5:25] = True          # large component
    m[25:35, 40:55] = True        # second component
    m[38, 0] = True               # speck (< 5%)
    cs = star.component_centres(m)
    assert len(cs) == 2 and all(m[c[1], c[0]] for c in cs)
    assert 5 <= cs[0][1] < 20 and 5 <= cs[0][0] < 25          # largest component first
    assert star.component_centres(np.zeros((5, 5), bool)) == []
    # a fragmented mask (every component < 5%) still gives its largest component
    f = np.zeros((40, 60), bool)
    f[::4, ::4] = True
    f[0:2, 0:2] = True
    cs = star.component_centres(f)
    assert len(cs) == 1 and f[cs[0][1], cs[0][0]] and cs[0][0] < 2 and cs[0][1] < 2


def test_multi_steps_override_centre():
    """With steps given, the centre argument of cut_star is unused (polyp_star passes a placeholder)."""
    rng = np.random.default_rng(6)
    for _ in range(20):
        logit, img, box, centre, th = _problem(rng, 12, 14)
        x0, y0, x1, y1 = box
        cs = [centre, (int(rng.integers(x0, x1 + 1)), int(rng.integers(y0, y1 + 1)))]
        steps = star.star_steps_multi(logit.shape, cs)
        for beta in (2.0, star.HARD):
            a = star.cut_star(logit, img, box, th, cs[0], beta, steps=steps)
            b = star.cut_star(logit, img, box, th, (x0, y0), beta, steps=steps)
            assert (a == b).all()
            # the hard result is closed along the multi-centre chains
            if math.isinf(beta):
                for p in zip(*np.nonzero(a)):
                    assert all(a[q] for q in star.segment_chain(steps, (int(p[0]), int(p[1]))))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(name, "ok")
