"""Parameter search on the held-out tuning set (Movahedi & Elder, 300 images).

Random search over the 11 parameters (unary table, sigma, lambda, c_neg) followed by rounds
of finer random search in ranges shrunk around the current best; objective F^max of the
binary fused masks (= mean per-image F_beta, see pipeline.binary_f).

  python scripts/tune.py --sod-model selfreformer --prompt mask_soft --workers 8
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, pipeline
from sodsam.fusion import FusionParams, color_sqdiff, fuse

RESULTS = Path(__file__).resolve().parents[1] / "results"

# name -> (low, high) of the initial search range
RANGES = {
    "f00": (0, 1000), "f01": (0, 1000), "f02": (-300, 300), "f03": (-300, 100),
    "f10": (-50, 50), "f11": (-100, 50), "f12": (-500, 0), "f13": (-1000, 0),
    "sigma": (1, 50), "lam": (0, 50), "c_neg": (-50, 0),
}
KEYS = list(RANGES)


def to_params(v):
    d = dict(zip(KEYS, v))
    return FusionParams(unary=[[d["f00"], d["f01"], d["f02"], d["f03"]],
                               [d["f10"], d["f11"], d["f12"], d["f13"]]],
                        sigma=d["sigma"], lam=d["lam"], c_neg=d["c_neg"])


def _init(sod_model, ds, prompt, variant):
    global INPUTS, VARIANT
    VARIANT = variant
    INPUTS = [pipeline.load_inputs(sod_model, ds, prompt, i) for i in data.list_ids(ds)]
    for x in INPUTS:
        x["sqdiff"] = color_sqdiff(x["img"])


def _score(v):
    p = to_params(v)
    fs = [pipeline.binary_f(fuse(x["img"], x["sod"], x["sam"], p, VARIANT, sqdiff=x["sqdiff"]), x["gt"])
          for x in INPUTS]
    return float(np.mean([f for f in fs if f is not None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_soft")
    ap.add_argument("--dataset", default=data.TUNE_DATASET)
    ap.add_argument("--variant", default="full", choices=["full", "potts", "unary"])
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-refine", type=int, default=200)
    ap.add_argument("--shrink", type=float, default=0.3, help="refined range = shrink * previous width")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lo = np.array([RANGES[k][0] for k in KEYS], float)
    hi = np.array([RANGES[k][1] for k in KEYS], float)
    bounds_lo, bounds_hi = lo.copy(), hi.copy()
    if args.variant != "full":   # c_neg is unused
        lo[KEYS.index("c_neg")] = hi[KEYS.index("c_neg")] = 0.0

    RESULTS.mkdir(exist_ok=True)
    name = f"tune_{args.sod_model}_{args.prompt}_{args.dataset}_{args.variant}"
    log = open(RESULTS / f"{name}.jsonl", "w")
    best_v, best_f = None, -1.0
    t0 = time.time()
    with Pool(args.workers, initializer=_init,
              initargs=(args.sod_model, args.dataset, args.prompt, args.variant)) as pool:
        for rnd in range(args.rounds + 1):
            n = args.n_random if rnd == 0 else args.n_refine
            cand = lo + (hi - lo) * rng.random((n, len(KEYS)))
            if best_v is not None:
                cand[0] = best_v
            for v, f in zip(cand, pool.imap(_score, list(cand), chunksize=1)):
                log.write(json.dumps({"round": rnd, "F": f, **dict(zip(KEYS, map(float, v)))}) + "\n")
                if f > best_f:
                    best_v, best_f = v, f
                log.flush()
            print(f"round {rnd}: best F={best_f:.4f} {dict(zip(KEYS, np.round(best_v, 3)))} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            width = (hi - lo) * args.shrink
            lo = np.maximum(best_v - width / 2, bounds_lo)
            hi = np.minimum(best_v + width / 2, bounds_hi)
            if args.variant != "full":
                lo[KEYS.index("c_neg")] = hi[KEYS.index("c_neg")] = 0.0

    out = {"F": best_f, "params": to_params(best_v).to_dict(), "args": vars(args)}
    (RESULTS / f"params_{args.sod_model}_{args.prompt}_{args.variant}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
