"""Graph cut on a foundation model's logits with a colour Potts term and a star-shape prior
(Veksler, "Star shape prior for graph-cut image segmentation", ECCV 2008).

Labels x_p in {0, 1}, 1 = object (the sink side of the cut, as in fusion.minimize). On the array given
(usually a crop around the box), pixels outside `box` (xyxy, inclusive) are fixed to 0; the energy over
the free pixels F is
    E(x) = sum_{p in F} u_p x_p + sum_{(p,q) 4-neighbours} w_pq [x_p != x_q] + sum_{p in F, p != c} beta [x_p = 1, x_s(p) = 0]
    u_p = -clip(logit_p, -L, L),
    w_pq = lam * (fu + (1 - fu) * exp(-||C_p - C_q||^2 / (2 sigma^2)))     (fusion.color_weights, fu = frac_uniform)
Pairs with one pixel outside the box count with the outside pixel at 0 (so a free pixel next to the fixed
region pays w_pq for x_p = 1).

Star step s(p) (the discretization of "the next pixel on the segment from p to the centre c"): the first pixel
after p of the DDA / Bresenham rasterization of the line p -> c. With v = c - p and the major axis the one with
the larger |v| (x on ties), s(p) = p + d with d_major = sign(v_major) and d_minor = round(v_minor / |v_major|)
(in {-1, 0, 1}; halves rounded away from zero, i.e. towards the diagonal). Properties (tests/test_star.py):
  * s(p) is strictly closer to c than p, so the chain p, s(p), s(s(p)), ... reaches c; this chain is the
    discrete segment from p to c that the hard prior keeps in the object;
  * every nonzero component of d has the sign of the matching component of v, so s(p) lies in the bounding
    box of p and c; with c inside the box, star edges never leave the box;
  * since every pixel of the chain re-aims at c, the chain drifts from the continuous segment p-c by up to
    ~0.2 |p - c| (6 pixels at |p - c| ~ 50). Every memoryless next-pixel rule does this (measured: the
    8-neighbour closest in angle, the one closest to the line, the DDA step: 6.0 / 6.0 / 5.7 pixels on an
    80x90 grid); the shape class is "star-shaped along these chains", as in Veksler's pairwise construction.
    Exact convexity along the continuous segments would need an edge from p to every pixel of its
    rasterized segment (O(N * diameter) edges).
beta = inf (HARD) is the hard prior: every object pixel's discrete segment to c is object (the object is
star-shaped w.r.t. c, or empty; a non-empty object therefore contains c, so a centre outside the true object
forces errors, and a second object far from c is lost).
Several centres (star_steps_multi): every pixel steps towards its nearest centre (Euclidean); the step is
strictly closer to that centre, so the distance to the nearest centre strictly decreases along every chain and
each chain ends at some centre: the object is a union of regions star-shaped w.r.t. the centres. The term beta [x_p = 1, x_q = 0] is a
single directed edge q -> p (q on the source side, p on the sink side), so the energy is submodular for any
beta >= 0 and the cut is exact; inf is the finite capacity HARD_CAP, far above any sum of other terms.
"""
import math

import cv2
import maxflow
import numpy as np

from .fusion import color_sqdiff, color_weights

HARD = math.inf
HARD_CAP = 1e9   # capacity standing for beta = inf (and for a fixed pixel); all other capacities are O(100)
FIXED_CAP = 1e9
DEFAULTS = {"L": 8.0, "lam": 1.0, "sigma": 10.0, "frac_uniform": 0.01}


def star_steps(shape, centre):
    """(dy, dx) int8 arrays of the star step of every pixel of an HxW array towards centre = (cx, cy);
    (0, 0) at the centre."""
    h, w = shape
    cx, cy = centre
    yy, xx = np.mgrid[:h, :w]
    vy, vx = cy - yy, cx - xx
    ay, ax = np.abs(vy), np.abs(vx)
    major = np.maximum(np.maximum(ay, ax), 1)

    def minor(v):   # round(|v| / major) half away from zero, with the sign of v
        return (np.sign(v) * np.floor(np.abs(v) / major + 0.5)).astype(np.int8)
    x_major = ax >= ay
    dx = np.where(x_major, np.sign(vx), minor(vx)).astype(np.int8)
    dy = np.where(x_major, minor(vy), np.sign(vy)).astype(np.int8)
    return dy, dx


def star_steps_multi(shape, centres):
    """Star steps of every pixel towards its nearest centre (squared Euclidean distance; the first centre in
    the list on ties); centres = [(cx, cy), ...]. (0, 0) exactly at the centres."""
    if not 1 <= len(centres) <= 31:     # np.choose takes at most 32 choices on numpy 1.x
        raise ValueError(f"{len(centres)} centres (1..31 supported)")
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    d2 = np.stack([(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy in centres])
    k = np.argmin(d2, axis=0)
    steps = [star_steps(shape, c) for c in centres]
    dy = np.choose(k, [s[0] for s in steps]).astype(np.int8)
    dx = np.choose(k, [s[1] for s in steps]).astype(np.int8)
    return dy, dx


def component_centres(mask, min_frac=0.05):
    """dt_centre of every 8-connected component of a boolean mask that holds at least min_frac of the mask's
    pixels, largest component first; the largest component is always kept (a fragmented non-empty mask gives at
    least one centre); [] only for an empty mask. min_frac = 0.05 gives at most 20 centres."""
    if not mask.any():
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-area, kind="stable")
    keep = [i + 1 for r, i in enumerate(order) if r == 0 or area[i] >= min_frac * mask.sum()]
    return [dt_centre(lab == i) for i in keep]


def box_centre(box):
    """Integer centre (x, y) of an xyxy box."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) // 2, (y0 + y1) // 2)


def dt_centre(mask):
    """(x, y) of the maximum of the Euclidean distance transform of a boolean mask (the mask's pixel farthest
    from its boundary, counting the array border as boundary; the first one in raster order on ties);
    None for an empty mask."""
    if not mask.any():
        return None
    m = np.pad(mask.astype(np.uint8), 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return (int(x), int(y))


def _params(th):
    p = dict(DEFAULTS)
    p.update(th or {})
    return p


def free_mask(shape, box):
    x0, y0, x1, y1 = box
    f = np.zeros(shape, bool)
    f[y0:y1 + 1, x0:x1 + 1] = True
    return f


def terms(logit, img, box, th, sqdiff=None):
    """Unary u (HxW, x = 1 cost; meaningful on the free pixels), Potts weights (w_right, w_down), free mask."""
    p = _params(th)
    u = -np.clip(logit.astype(np.float64), -p["L"], p["L"])
    if sqdiff is None:
        sqdiff = color_sqdiff(img)
    wr, wd = color_weights(None, p["sigma"], p["lam"], p["frac_uniform"], sqdiff)
    return u, (wr, wd), free_mask(logit.shape, box)


def energy(x, logit, img, box, th, centre, beta, sqdiff=None):
    """Energy of a labeling (inf if it violates the hard prior or labels a fixed pixel 1); for tests."""
    x = np.asarray(x, bool)
    u, (wr, wd), free = terms(logit, img, box, th, sqdiff)
    if (x & ~free).any():
        return math.inf
    xi = x.astype(np.int64)
    e = float((u * xi)[free].sum())
    e += float((wr * (xi[:, 1:] != xi[:, :-1])).sum() + (wd * (xi[1:] != xi[:-1])).sum())
    if beta > 0:
        dy, dx = star_steps(x.shape, centre)
        ys, xs = np.nonzero(free & ((dy != 0) | (dx != 0)))
        viol = int((x[ys, xs] & ~x[ys + dy[ys, xs], xs + dx[ys, xs]]).sum())
        if viol:
            e = math.inf if math.isinf(beta) else e + beta * viol
    return e


def cut_star(logit, img, box, th, centre, beta, sqdiff=None, steps=None):
    """Exact minimizer of E (module docstring) as a boolean mask of logit's shape.

    logit: HxW model logits; img: HxWx3 colour image of the same array; box: xyxy (inclusive) free region,
    pixels outside are 0; th: {"L", "lam", "sigma", "frac_uniform"} (missing keys from DEFAULTS);
    centre: (x, y) inside the box (ignored when beta == 0); beta: 0 (no prior), > 0, or HARD (= inf).
    sqdiff: optional cached fusion.color_sqdiff(img); steps: optional cached star_steps(shape, centre).
    The graph is built on the box padded by one pixel (the ring fixed to 0), so pairs across the box
    border are exact.
    """
    h, w = logit.shape
    x0, y0, x1, y1 = box
    X0, Y0, X1, Y1 = max(0, x0 - 1), max(0, y0 - 1), min(w - 1, x1 + 1), min(h - 1, y1 + 1)
    sl = (slice(Y0, Y1 + 1), slice(X0, X1 + 1))
    hp, wp = Y1 - Y0 + 1, X1 - X0 + 1
    if sqdiff is None:
        sq = color_sqdiff(img[sl])
    else:
        sq = (sqdiff[0][Y0:Y1 + 1, X0:X1], sqdiff[1][Y0:Y1, X0:X1 + 1])
    box_p = (x0 - X0, y0 - Y0, x1 - X0, y1 - Y0)
    u, (wr, wd), free = terms(logit[sl], None, box_p, th, sq)
    u = np.where(free, u, FIXED_CAP)

    g = maxflow.Graph[float](hp * wp, 4 * hp * wp)
    nodes = g.add_grid_nodes((hp, wp))
    g.add_grid_tedges(nodes, np.maximum(u, 0), np.maximum(-u, 0))
    w_right = np.zeros((hp, wp))
    w_right[:, :-1] = wr
    g.add_grid_edges(nodes, weights=w_right, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), symmetric=True)
    w_down = np.zeros((hp, wp))
    w_down[:-1] = wd
    g.add_grid_edges(nodes, weights=w_down, structure=np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]), symmetric=True)
    if beta > 0:
        if steps is None:
            dy, dx = star_steps((hp, wp), (centre[0] - X0, centre[1] - Y0))
        else:
            dy, dx = steps[0][sl], steps[1][sl]
        ys, xs = np.nonzero(free & ((dy != 0) | (dx != 0)))
        qy, qx = ys + dy[ys, xs], xs + dx[ys, xs]
        if not (free[qy, qx].all()):
            raise ValueError("a star edge leaves the box: the centre must be inside the box")
        cap = HARD_CAP if math.isinf(beta) else float(beta)
        p_id, q_id = nodes[ys, xs].ravel(), nodes[qy, qx].ravel()
        # beta [x_p = 1, x_q = 0]: p on the sink side, q on the source side cuts the edge q -> p
        g.add_edges(q_id, p_id, np.full(p_id.size, cap), np.zeros(p_id.size))
    g.maxflow()
    out = np.zeros((h, w), bool)
    out[sl] = g.get_grid_segments(nodes) & free
    return out


def segment_chain(steps, p):
    """Pixels (y, x) of the discrete segment from p = (y, x) to the centre, following the star steps."""
    dy, dx = steps
    chain = [p]
    y, x = p
    while dy[y, x] or dx[y, x]:
        y, x = y + int(dy[y, x]), x + int(dx[y, x])
        chain.append((y, x))
        if len(chain) > dy.size:
            raise RuntimeError("star chain does not terminate")
    return chain
