"""Confidence bins of the SOD maps, for one tuned parameter set per bin (tune2.py --bin-file/--bin,
fuse_eval.py v2bins:...).

Per image c = fraction of the pixels of the max-normalized SOD map (sod / sod.max()) in [0.2, 0.8]: an
uncertain map has many. The tertile cut points are computed on the tuning images (MSOD + MSRA-B1K)
and applied to every dataset. Writes results/bins_<sod>.json =
{"cuts": [t1, t2], "bins": {ds: {id: 0 | 1 | 2}}} (bin 0: c < t1, 1: t1 <= c < t2, 2: c >= t2).

  python scripts/bins.py --sod-model selfreformer
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tqdm import tqdm

from sodsam import data

RESULTS = Path(__file__).resolve().parents[1] / "results"
LO, HI = 0.2, 0.8


def confidence(id_, sod_model, ds):
    s = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png").astype(np.float64)
    s = s / (s.max() + 1e-13)
    return float(np.logical_and(s >= LO, s <= HI).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default="selfreformer")
    ap.add_argument("--tune-datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids per dataset (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    args = ap.parse_args()

    c = {}
    with Pool(args.workers) as pool:
        for ds in args.tune_datasets + [d for d in args.datasets if d not in args.tune_datasets]:
            ids = data.list_ids(ds)[:args.limit]
            f = partial(confidence, sod_model=args.sod_model, ds=ds)
            c[ds] = dict(zip(ids, tqdm(pool.imap(f, ids, chunksize=16), total=len(ids), desc=ds)))
    tune_c = np.array([v for ds in args.tune_datasets for v in c[ds].values()])
    cuts = np.quantile(tune_c, [1 / 3, 2 / 3])
    bins = {ds: {i: int(np.searchsorted(cuts, v, side="right")) for i, v in d.items()} for ds, d in c.items()}
    for ds, d in bins.items():
        counts = np.bincount(list(d.values()), minlength=3)
        print(f"{ds:<10s} n={len(d):5d}  bins {counts[0]:5d} {counts[1]:5d} {counts[2]:5d}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"bins_{args.sod_model}{args.tag}.json"
    out.write_text(json.dumps({"cuts": cuts.tolist(), "bins": bins, "tune_datasets": args.tune_datasets,
                               "range": [LO, HI]}, indent=1))
    print(f"cuts {cuts[0]:.4f} {cuts[1]:.4f} -> {out}")


if __name__ == "__main__":
    main()
