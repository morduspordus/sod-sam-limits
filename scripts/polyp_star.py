"""Star-shape prior graph cut on SAM's box-prompt logits (sodsam.star), tuned on the train images of a polyp
dataset and evaluated on its test images (PraNet split, sodsam.polyp).

  python scripts/polyp_star.py --datasets Kvasir-SEG CVC-ClinicDB --workers 16
  python scripts/polyp_star.py --datasets Kvasir-SEG --limit 4 --n-random 4 --rounds 1 --n-refine 2 --workers 0

Inputs: the SAM outputs of scripts/run_sam_polyp.py (output/polyp/<ds>/sam_box/). Rows:
  sam_multi       SAM mask with the best predicted IoU among the three of multimask output
  sam_single      SAM single-mask output
  cut             graph cut on the source logits (below), colour Potts only (beta = 0)
  star_box_hard   + hard star prior, centre = centre of the GT box
  star_dt_hard    + hard star prior, centre = maximum of the distance transform of the source SAM mask
                  (restricted to the dilated box; the box centre when that mask is empty)
  star_multi_hard + hard prior with one centre per component of the source SAM mask (>= 5% of its area;
                  sodsam.star.component_centres), every pixel stepping towards its nearest centre
  star_beta       + finite beta, tuned, with the centre(s) of the best hard-star row on the train images
  sam_*_box       the SAM rows with the mask restricted to the dilated box, the region every cut row uses
                  (so the cut rows are compared with SAM on equal terms)
The json also records, for the test images, how often the box centre lies outside the GT and how many GTs
have >= 2 components (cases the single-centre hard prior cannot represent).
Source logits: --source multi | single | auto (default: the SAM row with the higher train Dice).
All cuts: unary -clip(logit, -L, L), pixels outside the dilated box fixed to background; L, lam, sigma
(and beta) are tuned by random search on the train images (objective: mean Dice; --n-random candidates,
then --rounds rounds of --n-refine around the best in a range shrunk by --shrink; lam, sigma, beta are
searched in log scale). Metrics: per-image Dice and IoU, averaged over the images.
--limit N: only the first N train and the first N test ids (file name gets "_lim<N>").
--workers 0 runs in this process; otherwise each spawned worker holds a 1/workers shard of the images.
Output: results/polyp/<dataset>_src-<source>_seed<seed>[_lim<N>].json (params, train / test means, per-image
test scores, cut times, diagnostics).
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, polyp, star
from sodsam.fusion import color_sqdiff

RESULTS = Path(__file__).resolve().parents[1] / "results" / "polyp"
SAM_ROWS = ("sam_multi", "sam_single", "sam_multi_box", "sam_single_box")   # *_box: SAM mask within the dilated box
CUT_ROWS = ("cut", "star_box_hard", "star_dt_hard", "star_multi_hard", "star_beta")
HARD_ROWS = (("star_box_hard", "box"), ("star_dt_hard", "dt"), ("star_multi_hard", "multi"))
# name -> (low, high, log scale)
RANGES = {"L": (1.0, 16.0, False), "lam": (0.02, 20.0, True), "sigma": (2.0, 80.0, True),
          "beta": (0.01, 20.0, True)}
FIRST = {"L": 8.0, "lam": 1.0, "sigma": 10.0, "beta": 1.0}   # first candidate of every search


def sam_dir(ds):
    return data.OUT_ROOT / "polyp" / ds / "sam_box"


def dice_iou(pred, gt):
    inter = float((pred & gt).sum())
    s, u = float(pred.sum() + gt.sum()), float((pred | gt).sum())
    return (2 * inter / s if s else 1.0), (inter / u if u else 1.0)


def prepare(ds, id_, source):
    """Everything a worker needs for one image, in crop coordinates."""
    d = sam_dir(ds)
    meta = json.loads((d / f"{id_}.json").read_text())
    x0, y0, x1, y1 = meta["crop"]
    sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
    img = polyp.load_image(ds, id_)
    gt = polyp.load_mask(ds, id_)
    in_dbox = star.free_mask(gt.shape, tuple(meta["dbox"]))
    base = {}
    for name in ("multi", "single"):
        m = data.read_png(d / f"{id_}_{name}.png") > 127
        base[f"sam_{name}"] = dice_iou(m, gt)
        base[f"sam_{name}_box"] = dice_iou(m & in_dbox, gt)
    if gt[sl].sum() != gt.sum():
        raise ValueError(f"{ds}/{id_}: GT outside the crop")
    logit = np.load(d / f"{id_}_{source}.npy").astype(np.float32)
    shift = np.array([x0, y0, x0, y0])
    dbox = tuple(int(v) for v in np.array(meta["dbox"]) - shift)
    bx = tuple(int(v) for v in np.array(meta["box"]) - shift)
    sam_mask = (logit > 0) & star.free_mask(logit.shape, dbox)
    c_box = star.box_centre(bx)
    c_dt = star.dt_centre(sam_mask)
    c_multi = star.component_centres(sam_mask) or [c_box]
    sq = color_sqdiff(img[sl])
    # diagnostics of the star assumption (GT used only for reporting, never by a prediction)
    g = gt[sl]
    n_gt, _, st, _ = cv2.connectedComponentsWithStats(g.astype(np.uint8), connectivity=8)
    diag = {"gt_components_5pct": int((st[1:, cv2.CC_STAT_AREA] >= 0.05 * max(g.sum(), 1)).sum()),
            "box_centre_in_gt": bool(g[c_box[1], c_box[0]]), "n_multi_centres": len(c_multi)}
    return {"logit": logit, "sqdiff": (sq[0].astype(np.float32), sq[1].astype(np.float32)), "gt": g,
            "dbox": dbox, "centres": {"box": c_box, "dt": c_dt or c_box, "multi": c_multi[0]},
            "dt_fallback": c_dt is None, "diag": diag,
            "steps": {"box": star.star_steps(logit.shape, c_box),
                      "dt": star.star_steps(logit.shape, c_dt or c_box),
                      "multi": star.star_steps_multi(logit.shape, c_multi)},
            "base": base, "shape": logit.shape}


def run_cut(P, spec):
    """spec = (row, th, centre_name): mask of the crop."""
    row, th, cname = spec
    if row == "cut":
        beta, cname = 0.0, "box"
    elif row.endswith("_hard"):
        beta = star.HARD
    else:
        beta = th["beta"]
    sq = tuple(a.astype(np.float64) for a in P["sqdiff"])
    return star.cut_star(P["logit"], None, P["dbox"], th, P["centres"][cname], beta, sqdiff=sq,
                         steps=P["steps"][cname])


def evaluate(X, specs, keys, per_image):
    """Sums of (Dice, IoU) per spec over keys, or with per_image {key: [(dice, iou, seconds) per spec]}."""
    if per_image:
        out = {}
        for k in keys:
            res = []
            for spec in specs:
                if spec[0] in SAM_ROWS:
                    res.append((*X[k]["base"][spec[0]], 0.0))
                else:
                    t0 = time.perf_counter()
                    m = run_cut(X[k], spec)
                    res.append((*dice_iou(m, X[k]["gt"]), time.perf_counter() - t0))
            out[k] = res
        return out
    sums = np.zeros((len(specs), 2))
    for k in keys:
        for j, spec in enumerate(specs):
            sums[j] += X[k]["base"][spec[0]] if spec[0] in SAM_ROWS else dice_iou(run_cut(X[k], spec), X[k]["gt"])
    return sums


def _worker(conn, keys, source):
    X = {k: prepare(*k, source) for k in keys}
    conn.send(("ready", {k: (X[k]["dt_fallback"], X[k]["diag"]) for k in keys}))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        specs, ks, per_image = msg
        conn.send(evaluate(X, specs, [k for k in ks if k in X], per_image))


class Workers:
    def __init__(self, n, keys, source):
        self.n, self.fallback, self.diag = n, {}, {}
        if n == 0:
            self.X = {k: prepare(*k, source) for k in keys}
            self.fallback = {k: v["dt_fallback"] for k, v in self.X.items()}
            self.diag = {k: v["diag"] for k, v in self.X.items()}
            return
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, keys[w::n], source), daemon=True)
            p.start()
            b.close()   # the parent's copy of the child end: a dead worker then gives EOF instead of a hang
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            tag, fb = self._recv(c)
            if tag != "ready":
                raise RuntimeError("worker failed to start")
            self.fallback.update({k: v[0] for k, v in fb.items()})
            self.diag.update({k: v[1] for k, v in fb.items()})

    def _recv(self, c):
        """recv that fails (instead of blocking) when a worker died; on Windows poll itself raises
        BrokenPipeError for a dead peer, on Linux recv raises EOFError."""
        try:
            while not c.poll(30):
                dead = [p.exitcode for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f"a worker died (exit codes {dead})")
            return c.recv()
        except (EOFError, OSError):
            raise RuntimeError("a worker died (pipe closed)") from None

    def means(self, specs, keys):
        """(mean Dice, mean IoU) per spec over keys."""
        if self.n == 0:
            sums = evaluate(self.X, specs, keys, False)
        else:
            for c in self.conns:
                c.send((specs, keys, False))
            sums = sum(self._recv(c) for c in self.conns)
        return sums / len(keys)

    def per_image(self, specs, keys):
        if self.n == 0:
            return evaluate(self.X, specs, keys, True)
        for c in self.conns:
            c.send((specs, keys, True))
        out = {}
        for c in self.conns:
            out.update(self._recv(c))
        return out

    def close(self):
        """Stop the workers; never raises (it runs in a finally that may already be handling a dead worker)."""
        if self.n:
            for c in self.conns:
                try:
                    c.send(None)
                except OSError:
                    pass
            for p in self.procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.terminate()


def search(W, row, cname, names, keys, args, seed):
    """Random search of the parameters `names` of a row on keys; returns (th, mean train Dice)."""
    rng = np.random.default_rng(seed)
    lo0 = np.array([math.log(RANGES[k][0]) if RANGES[k][2] else RANGES[k][0] for k in names])
    hi0 = np.array([math.log(RANGES[k][1]) if RANGES[k][2] else RANGES[k][1] for k in names])
    to_th = lambda v: {k: float(math.exp(x) if RANGES[k][2] else x) for k, x in zip(names, v)}
    first = np.array([math.log(FIRST[k]) if RANGES[k][2] else FIRST[k] for k in names])
    lo, hi, best_v, best_f = lo0.copy(), hi0.copy(), None, -1.0
    for rnd in range(args.rounds + 1):
        n = args.n_random if rnd == 0 else args.n_refine
        cand = lo + (hi - lo) * rng.random((n, len(names)))
        cand[0] = first if best_v is None else best_v
        f = W.means([(row, to_th(v), cname) for v in cand], keys)[:, 0]
        j = int(np.argmax(f))
        if f[j] > best_f:
            best_v, best_f = cand[j], float(f[j])
        width = (hi - lo) * args.shrink
        lo, hi = np.maximum(best_v - width / 2, lo0), np.minimum(best_v + width / 2, hi0)
    return to_th(best_v), best_f


def run_dataset(ds, args):
    t_start = time.time()
    train = [(ds, i) for i in polyp.list_ids(ds, "train")[:args.limit]]
    test = [(ds, i) for i in polyp.list_ids(ds, "test")[:args.limit]]
    missing = [i for _, i in train + test if not (sam_dir(ds) / f"{i}.json").exists()]
    if missing:
        sys.exit(f"{ds}: {len(missing)} ids without SAM outputs (run_sam_polyp.py), e.g. {missing[0]}")
    fakes = sum(json.loads((sam_dir(ds) / f"{i}.json").read_text()).get("fake", False) for _, i in train + test)
    source = args.source
    if source == "auto":   # the SAM row with the higher mean train Dice
        m = np.mean([[dice_iou(data.read_png(sam_dir(ds) / f"{i}_{s}.png") > 127, polyp.load_mask(ds, i))[0]
                      for s in ("multi", "single")] for _, i in train], axis=0)
        source = "multi" if m[0] >= m[1] else "single"
    W = Workers(args.workers, train + test, source)
    rows = {}
    try:
        for r in SAM_ROWS:
            rows[r] = {"params": {}, "centre": None, "train_dice": float(W.means([(r, {}, None)], train)[0, 0])}
        base_names = ["L", "lam", "sigma"]
        for j, (row, cname) in enumerate((("cut", None),) + HARD_ROWS):
            th, f = search(W, row, cname, base_names, train, args, args.seed + j)
            rows[row] = {"params": th, "centre": cname, "train_dice": f}
            print(f"  {ds} {row}: train Dice {f:.4f} {th}", flush=True)
        # the finite-beta row uses the centre of the best hard row on the TRAIN images
        cname = max(HARD_ROWS, key=lambda rc: rows[rc[0]]["train_dice"])[1]
        th, f = search(W, "star_beta", cname, base_names + ["beta"], train, args, args.seed + len(HARD_ROWS) + 1)
        rows["star_beta"] = {"params": th, "centre": cname, "train_dice": f}
        print(f"  {ds} star_beta ({cname} centre): train Dice {f:.4f} {th}", flush=True)
        names = list(SAM_ROWS) + list(CUT_ROWS)
        specs = [(r, rows[r]["params"], rows[r]["centre"]) for r in names]
        per = W.per_image(specs, test)
        n_fallback = sum(W.fallback[k] for k in train + test)
    finally:
        W.close()
    for j, r in enumerate(names):
        v = np.array([per[k][j] for k in test])
        rows[r].update({"test_dice": float(v[:, 0].mean()), "test_iou": float(v[:, 1].mean()),
                        "seconds_per_image": float(v[:, 2].mean()),
                        "per_image": {k[1]: [float(v[i, 0]), float(v[i, 1])] for i, k in enumerate(test)}})
    dg = [W.diag[k] for k in test]
    diag = {"test_box_centre_outside_gt": int(sum(not d["box_centre_in_gt"] for d in dg)),
            "test_gt_2plus_components": int(sum(d["gt_components_5pct"] >= 2 for d in dg)),
            "test_multi_centres_mean": float(np.mean([d["n_multi_centres"] for d in dg])),
            "per_image": {k[1]: W.diag[k] for k in test}}
    out = {"dataset": ds, "source": source, "n_train": len(train), "n_test": len(test), "rows": rows,
           "dt_centre_fallbacks": n_fallback, "diagnostics": diag, "fake_inputs": fakes, "args": vars(args),
           "seconds": time.time() - t_start}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / (f"{ds}_src-{args.source}_seed{args.seed}" + (f"_lim{args.limit}" if args.limit else "") + ".json")
    path.write_text(json.dumps(out, indent=1))
    print(f"\n{ds}: train {len(train)} / test {len(test)}, cut source = SAM {source}"
          + (f"  [{fakes} FAKE SAM inputs]" if fakes else "") + f"  -> {path}")
    print(f"{'row':<15s} {'centre':<6s} {'train Dice':>10s} {'test Dice':>9s} {'test IoU':>9s} {'s/img':>7s}  params")
    for r in names:
        x = rows[r]
        print(f"{r:<15s} {str(x['centre'] or '-'):<6s} {x['train_dice']:10.4f} {x['test_dice']:9.4f} "
              f"{x['test_iou']:9.4f} {x['seconds_per_image']:7.3f}  "
              + " ".join(f"{k}={v:.3g}" for k, v in x["params"].items()), flush=True)
    print(f"test images: box centre outside the GT {diag['test_box_centre_outside_gt']}, GT with >= 2 components "
          f"(>= 5% each) {diag['test_gt_2plus_components']}, mean SAM centres {diag['test_multi_centres_mean']:.2f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(polyp.DATASETS))
    ap.add_argument("--source", default="auto", choices=["auto", "multi", "single"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N train and N test ids")
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-refine", type=int, default=20)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 = in this process")
    args = ap.parse_args()
    for ds in args.datasets:
        run_dataset(ds, args)


if __name__ == "__main__":
    main()
