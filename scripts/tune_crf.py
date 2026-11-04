"""Grid search of the dense CRF baseline (SOD + dense CRF) on the tuning sets.

Objective F^max of the CRF output map over all tuning images, as for our method. The paper's first version
tuned on MSOD for SelfReformer only (results/crf_params_selfreformer.json); --datasets defaults to the same
tuning sets as the graph cut (MSOD, MSRA-B1K, MSRA-B4K), per SOD model, and the result is
results/crf_params_<sod>_<datasets>.json.
  python scripts/tune_crf.py --sod-model samba --workers 16
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics
from sodsam.dense_crf import dense_crf

RESULTS = Path(__file__).resolve().parents[1] / "results"
GRID = {"sxy_b": [30, 50, 80], "srgb": [3, 5, 10, 13], "compat_b": [3, 5, 10], "compat_g": [3]}


def _init(sod_model, datasets, limit=None):
    global INPUTS
    INPUTS = [(data.load_image(ds, i), data.read_png(data.run_dir(sod_model, ds, "sod") / f"{i}.png"),
               data.load_gt(ds, i)) for ds in datasets for i in data.list_ids(ds)[:limit]]


def _score(kw):
    curves = [metrics.f_curve(dense_crf(img, sod, **kw), gt) for img, sod, gt in INPUTS if gt.any()]
    return float(np.mean(curves, axis=0).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.TUNE_DATASETS_ALL)
    ap.add_argument("--limit", type=int, default=None, help="images per dataset (smoke tests)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    grid = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(args.workers, initializer=_init, initargs=(args.sod_model, args.datasets, args.limit)) as pool:
        scores = pool.map(_score, grid, chunksize=1)
        base = pool.apply(_score, ({},))   # DSS defaults
    grid, scores = [{}] + grid, [base] + scores   # the defaults compete too
    for kw, f in sorted(zip(grid, scores), key=lambda t: -t[1])[:5]:
        print(f"F={f:.4f} {kw or 'DSS defaults'}")
    best = int(np.argmax(scores))
    out = {"F": scores[best], "F_default": base, "crf_params": grid[best],
           "grid": [{"F": f, **kw} for kw, f in zip(grid, scores)]}
    RESULTS.mkdir(exist_ok=True)
    name = f"crf_params_{args.sod_model}_{'+'.join(args.datasets)}" + (f"_lim{args.limit}" if args.limit else "")
    out["datasets"] = args.datasets
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"best F={scores[best]:.4f} {grid[best] or 'DSS defaults'}; DSS defaults F={base:.4f}")


if __name__ == "__main__":
    main()
