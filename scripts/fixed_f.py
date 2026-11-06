"""F at thresholds that do not favour soft maps (review: F^max picks the best threshold per dataset, which only
a soft map can exploit; the graph cut is binary).

For the SOD map, the paper's graph cut (output/<sod>/<ds>/gc_zoom) and the paper's fusion head
(output/<sod>_hz/<ds>/sod), per dataset:
  Fmax    max over thresholds of the mean F curve (the paper's measure)
  F05     mean per-image F at the fixed threshold 128/255 of the min-max normalized map (pixel >= 128)
  Fadp    mean per-image F at the adaptive threshold min(2 * mean, 1) of the normalized map (PySODMetrics)
  Fmean   mean over the 256 thresholds of the mean F curve
and, for F05, paired bootstrap 95% intervals (2000 resamples of the images) of cut - SOD, head - SOD and
head - cut. A binary map scores the same at every threshold in (0, 255], so F05 and Fadp equal its F^max
up to the normalization.

  python scripts/fixed_f.py --sod-model samba --workers 8
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
MAPS = ("sod", "gc", "head")
I05 = 255 - 128          # f_curve index of the threshold 128


def _f_adaptive(pred, gt):
    x = metrics._minmax(pred)
    th = min(2 * x.mean(), 1.0)
    b = x >= th
    tp, n_pred, n_gt = np.count_nonzero(b & gt), np.count_nonzero(b), np.count_nonzero(gt)
    if tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_gt
    return 1.3 * p * r / (0.3 * p + r)


def _image(id_, sod_model, ds):
    gt = data.load_gt(ds, id_)
    paths = {"sod": data.run_dir(sod_model, ds, "sod"), "gc": data.run_dir(sod_model, ds, "gc_zoom"),
             "head": data.run_dir(f"{sod_model}_hz", ds, "sod")}
    out = {}
    for m, d in paths.items():
        p = data.read_png(d / f"{id_}.png")
        out[m] = (metrics.f_curve(p, gt), _f_adaptive(p, gt))
    return out


def _ci(a, b, rng, n=2000):
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(axis=1)
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            per = pool.map(partial(_image, sod_model=args.sod_model, ds=ds), data.list_ids(ds), chunksize=16)
            res = {}
            f05 = {}
            for m in MAPS:
                curves = np.stack([r[m][0] for r in per])
                mean_curve = curves.mean(axis=0)
                f05[m] = curves[:, I05]
                res[m] = {"Fmax": float(mean_curve.max()), "F05": float(f05[m].mean()),
                          "Fadp": float(np.mean([r[m][1] for r in per])), "Fmean": float(mean_curve.mean())}
            res["ci_F05"] = {"gc-sod": _ci(f05["gc"], f05["sod"], rng), "head-sod": _ci(f05["head"], f05["sod"], rng),
                             "head-gc": _ci(f05["head"], f05["gc"], rng)}
            res["n"] = len(per)
            out[ds] = res
            print(f"{args.sod_model} {ds:10s} " + " | ".join(
                f"{m}: max={res[m]['Fmax']:.3f} .5={res[m]['F05']:.3f} adp={res[m]['Fadp']:.3f} mean={res[m]['Fmean']:.3f}"
                for m in MAPS) + " | " + " ".join(f"{k}={v[0]:+.4f}[{v[1]:+.4f},{v[2]:+.4f}]"
                                                 for k, v in res["ci_F05"].items()), flush=True)
    (RESULTS / f"fixed_f_{args.sod_model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
