"""Paired bootstrap over images: 95% confidence intervals of metric differences between methods.

Per image we keep the F curve (256 thresholds) and the MAE; a resample of the images (with replacement,
the same resample for all methods) gives F^max from the mean F curve and the mean MAE, as in the
benchmark metrics. Reported per dataset and comparison: the observed difference and its 2.5/97.5
percentiles over --b resamples, and the share of resamples where the difference is > 0 (F) or < 0 (MAE).

  python scripts/bootstrap_ci.py --maps sod=samba:sod gc=samba:gc_zoom head=samba_hz:sod \
      --compare gc-sod head-sod head-gc --tag samba
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, metrics


def _image(id_, ds, srcs):
    g = data.load_gt(ds, id_)
    out = []
    for run, sub in srcs:
        p = data.read_png(data.run_dir(run, ds, sub) / f"{id_}.png")
        if p.shape != g.shape:
            p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
        out.append((metrics.f_curve(p, g), metrics.mae(p, g)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True, help="name=<run>:<subdir>")
    ap.add_argument("--compare", nargs="+", required=True, help="a-b pairs of names")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    names = [m.split("=", 1)[0] for m in args.maps]
    srcs = [tuple(m.split("=", 1)[1].split(":", 1)) for m in args.maps]
    rng = np.random.default_rng(0)
    res = {}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            recs = [r for r in pool.map(partial(_image, ds=ds, srcs=srcs), data.list_ids(ds), chunksize=32) if r]
            F = np.stack([[r[k][0] for r in recs] for k in range(len(names))])     # methods x images x 256
            M = np.array([[r[k][1] for r in recs] for k in range(len(names))])     # methods x images
            n = F.shape[1]
            idx = rng.integers(0, n, size=(args.b, n))
            fmax = np.stack([F[:, ix].mean(1).max(-1) for ix in idx])             # b x methods
            mae = np.stack([M[:, ix].mean(1) for ix in idx])
            f0, m0 = F.mean(1).max(-1), M.mean(1)
            for c in args.compare:
                a, b = (names.index(x) for x in c.split("-"))
                df, dm = fmax[:, a] - fmax[:, b], mae[:, a] - mae[:, b]
                res[f"{c}/{ds}"] = {
                    "F_diff": float(f0[a] - f0[b]), "F_ci": [float(np.percentile(df, 2.5)), float(np.percentile(df, 97.5))],
                    "F_p_better": float((df > 0).mean()),
                    "MAE_diff": float(m0[a] - m0[b]), "MAE_ci": [float(np.percentile(dm, 2.5)), float(np.percentile(dm, 97.5))],
                    "MAE_p_better": float((dm < 0).mean()), "n": n}
                r = res[f"{c}/{ds}"]
                print(f"{args.tag} {c:9s} {ds:10s} dF {r['F_diff']:+.4f} [{r['F_ci'][0]:+.4f}, {r['F_ci'][1]:+.4f}] "
                      f"P(>0)={r['F_p_better']:.3f} | dMAE {r['MAE_diff']:+.4f} [{r['MAE_ci'][0]:+.4f}, "
                      f"{r['MAE_ci'][1]:+.4f}]", flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
