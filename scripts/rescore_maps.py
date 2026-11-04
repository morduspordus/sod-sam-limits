"""Re-score saved maps with the current metrics (sodsam/metrics.py).

Heads: the test maps output/<tag>/<ds>/sod of each head; writes results/heads/<tag>.json = the head's own json
(--head-root) with "test" replaced and the old values kept as "test_toolbox".
Other maps: --maps name=<run>:<subdir> writes results/eval_maps_<name>.json as {"datasets": {ds: {"map": m}}}.

  python scripts/rescore_maps.py --heads samba_hz selfreformer_hz --head-root import/hz/hz_results
  python scripts/rescore_maps.py --maps samba_samrefiner=samba:samrefiner
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from sodsam import data, metrics

ROOT = Path(__file__).resolve().parents[1]


def _one(id_, ds, run, sub):
    g = data.load_gt(ds, id_)
    p = data.read_png(data.run_dir(run, ds, sub) / f"{id_}.png")
    if p.shape != g.shape:
        p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return metrics.image_scores(p, g)


def score(pool, run, sub, datasets):
    out = {}
    for ds in datasets:
        out[ds] = metrics.aggregate(pool.map(partial(_one, ds=ds, run=run, sub=sub), data.list_ids(ds), chunksize=16))
        print(metrics.format_row(f"{run}:{sub} {ds}", out[ds]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="*", default=[])
    ap.add_argument("--head-root", nargs="*", default=[], help="directories with the heads' json files")
    ap.add_argument("--maps", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    out = ROOT / "results" / "heads"
    out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for tag in args.heads:
            src = next(Path(r) / f"{tag}.json" for r in args.head_root if (Path(r) / f"{tag}.json").exists())
            d = json.loads(src.read_text())
            d.setdefault("test_toolbox", d.get("test"))
            d["test"] = score(pool, tag, "sod", args.datasets)
            (out / f"{tag}.json").write_text(json.dumps(d, indent=1))
        for spec in args.maps:
            name, src = spec.split("=", 1)
            run, sub = src.split(":", 1)
            res = {"datasets": {ds: {"map": m} for ds, m in score(pool, run, sub, args.datasets).items()}}
            (ROOT / "results" / f"eval_maps_{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
