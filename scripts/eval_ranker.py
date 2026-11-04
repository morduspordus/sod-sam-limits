"""Evaluate the trained ranker (scripts/train_ranker.py) on the benchmarks.

Per dataset the chosen map of every image is written to <OUT_ROOT>/<sod_model>/<dataset>/ranker/<id>.png
(the soft SOD map when the ranker keeps it, else the chosen binary mask x 255) and the metrics
(F^max, MAE, E, S of metrics.image_scores + aggregate, as in fuse_eval.py) of the ranker, the SOD map, the
graph cut and the oracle (true best candidate) go to results/eval_<sod_model>_ranker<tag>.json, with the
pick rate per candidate and the fraction of images whose pick is the true best.

  python scripts/eval_ranker.py --sod-model samba --workers 8
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from sodsam import data, metrics, ranker
from train_ranker import ROOT, add_common_args, ensure_cache, get_device

GC = ranker.CANDIDATES.index("gc")


def _score(job, sod_model, cache_root, maps_root):
    """job = ((dataset, id), pick). Per-image metrics of the chosen, SOD, graph-cut and oracle maps;
    writes the chosen map."""
    (ds, id_), pick = job
    gt = data.load_gt(ds, id_)
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    c = ranker.load_cached(sod_model, ds, id_, cache_root)
    oracle = int(np.argmax(c["f"]))
    chosen = ranker.candidate_map(pick, sod, c["masks"])
    data.write_png(Path(maps_root or data.OUT_ROOT) / sod_model / ds / "ranker" / f"{id_}.png", chosen)
    out = {"ranker": metrics.image_scores(chosen, gt), "sod": metrics.image_scores(sod, gt),
           "gc": metrics.image_scores(ranker.candidate_map(GC, sod, c["masks"]), gt),
           "oracle": metrics.image_scores(ranker.candidate_map(oracle, sod, c["masks"]), gt)}
    return out, c["f"]


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--weights", default=None, help="model state dict (default weights/ranker_<sod_model><tag>.pt)")
    ap.add_argument("--train-json", default=None, help="training log with tau (default results/ranker_<...>_train.json)")
    ap.add_argument("--tau", type=float, default=None, help="override the abstention threshold")
    ap.add_argument("--maps-root", default=None, help="where <sod_model>/<dataset>/ranker/ maps go (default OUT_ROOT)")
    ap.add_argument("--untrained", action="store_true", help="debug: a fresh model instead of the weights")
    args = ap.parse_args()
    device = get_device(args)
    name = f"ranker_{args.sod_model}{args.tag}"
    out_root = Path(args.out_root)
    train_json = Path(args.train_json or out_root / "results" / f"{name}_train.json")
    tau = args.tau
    if tau is None:
        if args.untrained and not train_json.exists():
            tau = 0.0
        else:
            tau = float(json.loads(train_json.read_text())["tau"])
    net = ranker.Ranker(pretrained=args.untrained, backbone_weights=None).to(device)
    if not args.untrained:
        weights = Path(args.weights or out_root / "weights" / f"{name}.pt")
        net.load_state_dict(torch.load(weights, map_location=device))
    print(f"{name}: tau {tau:.3f}, device {device}", flush=True)

    results = {"sod_model": args.sod_model, "tau": tau, "candidates": ranker.CANDIDATES, "datasets": {}}
    for ds in args.datasets:
        ensure_cache(args, [ds])
        dset = ranker.RankerSet(args.sod_model, [ds], args.size, False, args.cache_root, args.limit)
        loader = DataLoader(dset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            pin_memory=device == "cuda")
        pred, f, curves = ranker.predict(net, loader, device)
        picks = ranker.decide(pred, tau)
        fn = partial(_score, sod_model=args.sod_model, cache_root=args.cache_root, maps_root=args.maps_root)
        jobs = [(item, int(p)) for item, p in zip(dset.items, picks)]
        if args.workers > 1:
            with Pool(args.workers) as pool:
                per = pool.map(fn, jobs, chunksize=8)
        else:
            per = [fn(j) for j in jobs]
        scores = [p[0] for p in per]
        res = {m: metrics.aggregate([s[m] for s in scores]) for m in ("ranker", "sod", "gc", "oracle")}
        best = f.max(1)
        res["pick_rate"] = {c: float(np.mean(picks == k)) for k, c in enumerate(ranker.CANDIDATES)}
        res["pick_is_best"] = float(np.mean(f[np.arange(len(f)), picks] >= best - 1e-6))
        res["mean_f"] = {"ranker": float(f[np.arange(len(f)), picks].mean()), "sod": float(f[:, 0].mean()),
                         "gc": float(f[:, GC].mean()), "oracle": float(best.mean())}
        res["top1"] = float(np.mean(pred.argmax(1) == f.argmax(1)))
        res["n"] = len(f)
        results["datasets"][ds] = res
        for m in ("ranker", "sod", "gc", "oracle"):
            print(metrics.format_row(f"{ds} {m}", res[m]), flush=True)
        print(f"{ds}: picks " + " ".join(f"{c}={r:.1%}" for c, r in res["pick_rate"].items())
              + f", pick is the true best on {res['pick_is_best']:.1%}, argmax top-1 {res['top1']:.1%}", flush=True)

    (out_root / "results").mkdir(parents=True, exist_ok=True)
    out = out_root / "results" / f"eval_{args.sod_model}_ranker{args.tag}.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
