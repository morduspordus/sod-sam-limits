"""Score maps of a high-resolution benchmark against its FULL-resolution GT, so the numbers are
comparable to the published ones.

  python scripts/hr_eval.py --sod-model birefnet --dataset DIS-VD-1k --sub sod full_v2 ...
  python scripts/hr_eval.py --sod-model birefnet --dataset DIS-VD --sub sod      # the imported full-res maps
  python scripts/hr_eval.py --dir <any directory of <id>.png> --dataset HRSOD-TE

Maps are read from output/<sod_model>/<dataset>/<sub>/<id>.png (or --dir) and resized to the GT size of
hr_parent(dataset): bilinear for soft maps, nearest for binary ones (values within {0, 255}). Metrics:
sodsam.metrics (PySODMetrics conventions: F^max, MAE, E, S). Writes
results/eval_hr_<sod_model>_<dataset>_<sub>.json and prints metrics.format_row.
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

RESULTS = Path(__file__).resolve().parents[1] / "results"


def to_gt_size(pred, shape):
    """Upsample (or downsample) a uint8 map to the GT shape; nearest for binary maps."""
    if pred.shape == shape:
        return pred
    binary = np.isin(np.unique(pred), (0, 255)).all()
    interp = cv2.INTER_NEAREST if binary else (cv2.INTER_LINEAR if shape[0] * shape[1] >= pred.size
                                               else cv2.INTER_AREA)
    return cv2.resize(pred, (shape[1], shape[0]), interpolation=interp)


def _one(id_, directory, parent, suffix):
    p = Path(directory) / f"{id_}{suffix}"
    if not p.exists():
        return id_, None
    gt = data.load_gt(parent, id_)
    pred = to_gt_size(data.read_png(p), gt.shape)
    s = metrics.image_scores(pred, gt)
    return id_, {"f_curve": s["f_curve"], "mae": s["mae"], "s": s["s"], "e": s["e"]}


def evaluate(directory, dataset, suffix=".png", workers=1, limit=None):
    parent = data.hr_parent(dataset)
    ids = data.list_ids(dataset)[:limit]
    f = partial(_one, directory=directory, parent=parent, suffix=suffix)
    if workers > 1:
        with Pool(workers) as pool:
            res = list(pool.imap(f, ids, chunksize=4))
    else:
        res = [f(i) for i in ids]
    missing = [i for i, s in res if s is None]
    per = [s for _, s in res if s is not None]
    if not per:
        raise FileNotFoundError(f"no maps for {dataset} in {directory}")
    agg = metrics.aggregate(per)
    agg["missing"] = len(missing)
    agg["per_image"] = {i: {"mae": float(s["mae"]), "s": float(s["s"]), "e": float(s["e"]),
                            "fmax": float(s["f_curve"].max())} for i, s in res if s is not None}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default=None)
    ap.add_argument("--dataset", required=True, help="e.g. DIS-VD-1k (scored at DIS-VD resolution) or DIS-VD")
    ap.add_argument("--sub", nargs="+", default=["sod"], help="subdirs under output/<sod_model>/<dataset>/")
    ap.add_argument("--dir", default=None, help="score this directory of <id>.png instead")
    ap.add_argument("--suffix", default=".png", help="file suffix after the id, e.g. _1.png for SAM masks")
    ap.add_argument("--tag", default="", help="appended to the result file name")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not args.dir and not args.sod_model:
        ap.error("--sod-model or --dir is required")

    runs = ([(Path(args.dir), Path(args.dir).name)] if args.dir
            else [(data.run_dir(args.sod_model, args.dataset, s), s) for s in args.sub])
    RESULTS.mkdir(exist_ok=True)
    for directory, sub in runs:
        agg = evaluate(directory, args.dataset, args.suffix, args.workers, args.limit)
        model = args.sod_model or "dir"
        out = RESULTS / f"eval_hr_{model}_{args.dataset}_{sub}{args.tag}.json"
        agg.update({"sod_model": model, "dataset": args.dataset, "gt": data.hr_parent(args.dataset),
                    "sub": sub, "dir": str(directory)})
        out.write_text(json.dumps(agg, indent=1))
        print(metrics.format_row(f"{model}/{args.dataset}/{sub}", agg) +
              (f" missing={agg['missing']}" if agg["missing"] else ""), flush=True)


if __name__ == "__main__":
    main()
