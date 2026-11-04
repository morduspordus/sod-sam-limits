"""Apply a tuned label-cost fusion (tune_labelcost.py json) to the benchmarks: writes the full-resolution
masks to output/<sod>/<ds>/labelcost/<id>.png and results/eval_<sod>_labelcost.json (fuse_eval format).

  python scripts/eval_labelcost.py --sod-model samba --params results/tune2/samba_MSOD+MSRA-B1K_labelcost_all.json
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

from sodsam import data, labelcost, metrics, pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _init(scores):
    global SCORES
    SCORES = scores


def _image(id_, sod_model, ds, th, opts, tag):
    x = labelcost.load_prepared(sod_model, ds, id_, SCORES, max_labels=opts["max_labels"], min_inside=opts["min_inside"])
    m = labelcost.upsample(labelcost.fit(x, th, opts["sweeps"]), x["full_shape"]).astype(np.uint8) * 255
    data.write_png(data.run_dir(sod_model, ds, tag) / f"{id_}.png", m)
    gt = data.load_gt(ds, id_)
    if m.shape != gt.shape:
        import cv2
        m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return metrics.image_scores(m, gt), x["no_auto"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="first N images per dataset (smoke tests)")
    ap.add_argument("--tag", default="labelcost")
    args = ap.parse_args()
    res = json.loads(Path(args.params).read_text())
    th, a = res["params"], res["args"]
    opts = {"sweeps": a.get("sweeps", 2), "max_labels": a.get("max_labels", 40), "min_inside": a.get("min_inside", 0.3)}
    out = {"sod_model": args.sod_model, "prompt": "auto+mask_zoom", "params": th, "opts": opts,
           "auto_dir": labelcost.AUTO_DIR, "datasets": {}}
    for ds in args.datasets:
        ids = data.list_ids(ds)
        if args.limit:
            ids = ids[:args.limit]
        scores = pipeline.load_scores(args.sod_model, ds, "mask_zoom")
        with Pool(args.workers, initializer=_init, initargs=(scores,)) as pool:
            f = partial(_image, sod_model=args.sod_model, ds=ds, th=th, opts=opts, tag=args.tag)
            per = list(tqdm(pool.imap(f, ids, chunksize=4), total=len(ids), desc=ds))
        agg = metrics.aggregate([p for p, _ in per])
        agg["n_missing_auto"] = int(sum(m for _, m in per))
        out["datasets"][ds] = {"labelcost": agg}
        print(metrics.format_row(f"{ds} labelcost", agg) + f" missing auto {agg['n_missing_auto']}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"eval_{args.sod_model}_{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
