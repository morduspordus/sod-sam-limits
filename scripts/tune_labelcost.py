"""Tune the label-cost fusion (sodsam/labelcost.py: a, d, lam_L, h0, sigma, lam) by random search with
rounds of refinement around the best (as tune2.py), objective = mean per-image F_beta of the binary
half-resolution mask (beta^2 = 0.3, F = 0 for an empty GT), then the full metrics of the best parameters
(nearest-upsampled masks) on the tuning and held-out halves.

  python scripts/tune_labelcost.py --sod-model samba --datasets MSOD MSRA-B1K --workers 2
--max-images N keeps a random subset of N images per dataset (seed --seed). Images whose automatic masks
(sam_auto npz) are missing use the three prompted masks only; their number is reported.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, labelcost, metrics, pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tune2"


def binary_f(pred, gt):
    tp = np.logical_and(pred, gt).sum()
    if tp == 0:
        return 0.0
    p, r = tp / pred.sum(), tp / gt.sum()
    return 1.3 * p * r / (0.3 * p + r)


def _init(sod_model, keys, opts):
    global X, OPTS
    OPTS = opts
    X = {}
    scores = {}
    for ds, i in keys:
        if ds not in scores:
            scores[ds] = pipeline.load_scores(sod_model, ds, "mask_zoom")
        X[ds, i] = labelcost.load_prepared(sod_model, ds, i, scores[ds], max_labels=opts["max_labels"],
                                           min_inside=opts["min_inside"])


def _fmean(args):
    th, ids = args
    return float(np.mean([binary_f(labelcost.fit(X[i], th, OPTS["sweeps"]), X[i]["gt"]) for i in ids]))


def _full(args):
    th, ids = args
    per = []
    for i in ids:
        m = labelcost.upsample(labelcost.fit(X[i], th, OPTS["sweeps"]), X[i]["full_shape"])
        per.append(metrics.image_scores(m.astype(np.uint8) * 255, data.load_gt(i[0], i[1])))
    return metrics.aggregate(per)


def _missing(ids):
    return sum(X[i]["no_auto"] for i in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default="samba")
    ap.add_argument("--datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--tune-half", default="all", choices=["A", "B", "all"])
    ap.add_argument("--max-images", type=int, default=0, help="per dataset; 0 = all")
    ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-refine", type=int, default=30)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--max-labels", type=int, default=40)
    ap.add_argument("--min-inside", type=float, default=0.3)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    keys = []
    for ds in args.datasets:
        ids = data.list_ids(ds)
        if args.max_images and len(ids) > args.max_images:
            ids = sorted(rng.choice(ids, args.max_images, replace=False))
        keys += [(ds, i) for i in ids]
    tune_ids = {"A": keys[0::2], "B": keys[1::2], "all": keys}[args.tune_half]
    test_ids = {"A": keys[1::2], "B": keys[0::2], "all": keys}[args.tune_half]
    ranges = labelcost.LABELCOST_RANGES
    names = list(ranges)
    lo0 = np.array([ranges[k][0] for k in names], float)
    hi0 = np.array([ranges[k][1] for k in names], float)
    lo, hi = lo0.copy(), hi0.copy()
    opts = {"sweeps": args.sweeps, "max_labels": args.max_labels, "min_inside": args.min_inside}
    best_v, best_f, t0 = None, -1.0, time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.sod_model, keys, opts)) as pool:
        n_missing = pool.apply(_missing, (keys,))
        print(f"{len(keys)} images ({n_missing} without automatic masks), {args.workers} workers", flush=True)
        for rnd in range(args.rounds + 1):
            n = args.n_random if rnd == 0 else args.n_refine
            cand = lo + (hi - lo) * rng.random((n, len(names)))
            if best_v is not None:
                cand[0] = best_v
            fs = pool.imap(_fmean, [(dict(zip(names, v)), tune_ids) for v in cand], chunksize=1)
            for v, f in zip(cand, fs):
                if f > best_f:
                    best_v, best_f = v, f
            width = (hi - lo) * args.shrink
            lo, hi = np.maximum(best_v - width / 2, lo0), np.minimum(best_v + width / 2, hi0)
            print(f"round {rnd}: best F={best_f:.4f} {dict(zip(names, np.round(best_v, 2).tolist()))} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        th = dict(zip(names, map(float, best_v)))
        tune_m = pool.apply(_full, ((th, tune_ids),))
        test_m = pool.apply(_full, ((th, test_ids),))
        test_per_ds = {ds: pool.apply(_full, ((th, [k for k in test_ids if k[0] == ds]),)) for ds in args.datasets}
    name = f"{args.sod_model}_{'+'.join(args.datasets)}_labelcost_{args.tune_half}"
    if labelcost.AUTO_DIR != "sam_auto":
        name += "_" + labelcost.AUTO_DIR.removeprefix("sam_auto_")
    out = {"model": ["labelcost", "potts", "binary"], "prompt": "auto+mask_zoom", "params": th, "auto_dir": labelcost.AUTO_DIR,
           "tune": tune_m, "test": test_m, "test_per_dataset": test_per_ds, "args": vars(args),
           "n_images": len(keys), "n_missing_auto": n_missing, "seconds": time.time() - t0}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name:<45s} tune F={tune_m['Fmax']:.4f} | held-out F={test_m['Fmax']:.4f} "
          f"MAE={test_m['MAE']:.4f} E={test_m['E']:.4f} S={test_m['S']:.4f} | "
          + " ".join(f"{ds}:F={m['Fmax']:.4f}" for ds, m in test_per_ds.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
