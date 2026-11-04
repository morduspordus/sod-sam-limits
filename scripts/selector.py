"""Learned per-image selector: for each image, keep the SOD map or replace it by one of SAM's masks.

The oracle diagnostic (oracle.py) showed that for a strong SOD model the headroom from SAM sits in
5-11% of the images, where one SAM mask is clearly better than the SOD map. This learns to find them.

  features  per (image, SAM mask k): SAM score and rank, mask and SOD areas, agreement between the
            mask and the SOD map (IoU, SOD mass inside / outside the mask, mean SOD inside), SOD
            confidence, mask components, border contact, overlap with the other two masks.
  target    per-image F of mask k minus per-image F of the SOD map (F = max over thresholds of the
            image's F curve, as in oracle.py).
  model     gradient-boosted regression trees, trained on the tuning images (MSOD + MSRA-B1K +
            MSRA-B4K, 5300 images, none seen by the SOD models). An image takes mask k* = argmax of the
            predicted gain when that gain exceeds tau; tau is chosen to maximize the dataset F^max of
            5-fold out-of-fold predictions on the tuning images.
  output    metrics of the selected maps on the five benchmarks, next to the SOD map and the oracle.

  python scripts/selector.py --sod-model samba --workers 2
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

from sodsam import data, metrics, pipeline

OUT = Path(__file__).resolve().parents[1] / "results" / "selector"
TRAIN_DATASETS = ["MSOD", "MSRA-B1K", "MSRA-B4K"]


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def mask_features(m, s, sod_bin, others, score, rank, scores):
    h, w = m.shape
    area = m.mean()
    n_comp = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean()
    mass = s.sum() + 1e-9
    return [score, rank, max(scores) - score, area, _iou(m, sod_bin),
            s[m].sum() / mass, s[m].mean() if m.any() else 0.0, s[~m].mean() if (~m).any() else 0.0,
            area - sod_bin.mean(), n_comp, border, *[_iou(m, o) for o in others]]


def image_record(id_, sod_model, ds, full_scores):
    """Features (3 x F), per-image F of [SOD, mask 1..3], and, if full_scores, image_scores of all four."""
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sam_dir = data.run_dir(sod_model, ds, "sam_mask_soft")
    masks = [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
    scores = _SCORES[id_]
    s = sod.astype(np.float64) / (sod.max() + 1e-13)
    sod_bin = s >= 127 / 255
    conf = np.abs(s - 0.5).mean()
    unsure = ((s > 0.2) & (s < 0.8)).mean()
    border_sod = np.concatenate([sod_bin[0], sod_bin[-1], sod_bin[:, 0], sod_bin[:, -1]]).mean()
    order = np.argsort(-np.asarray(scores))
    rank = {int(k): r for r, k in enumerate(order)}
    common = [sod_bin.mean(), conf, unsure, border_sod, np.log(gt.size)]
    feats = [mask_features(masks[k], s, sod_bin, [masks[j] for j in range(3) if j != k],
                           scores[k], rank[k], scores) + common for k in range(3)]
    cands = [sod] + [m.astype(np.uint8) * 255 for m in masks]
    f = [float(metrics.f_curve(c, gt).max()) for c in cands]
    rec = {"id": id_, "X": np.asarray(feats, np.float64), "f": np.asarray(f)}
    if full_scores:
        rec["scores"] = [metrics.image_scores(c, gt) for c in cands]
    return rec


def _init(sod_model, ds):
    global _SCORES
    _SCORES = pipeline.load_scores(sod_model, ds, "mask_soft")


def records(sod_model, ds, full_scores, workers):
    with Pool(workers, initializer=_init, initargs=(sod_model, ds)) as pool:
        f = partial(image_record, sod_model=sod_model, ds=ds, full_scores=full_scores)
        return [r for r in pool.imap(f, data.list_ids(ds), chunksize=16) if r is not None]


def choose(pred, tau):
    """Index into [SOD, mask1, mask2, mask3] per image: argmax predicted gain if above tau, else SOD."""
    k = pred.argmax(1)
    return np.where(pred.max(1) > tau, k + 1, 0)


def mean_f(recs, pick):
    """Mean per-image F of the picked maps: the proxy used to choose tau (dataset F^max needs the
    mean F curve, which is computed only for the final evaluation)."""
    return float(np.mean([r["f"][p] for r, p in zip(recs, pick)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-datasets", nargs="+", default=TRAIN_DATASETS)
    ap.add_argument("--eval-datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    OUT.mkdir(parents=True, exist_ok=True)

    train = [r for ds in args.train_datasets for r in records(args.sod_model, ds, False, args.workers)]
    X = np.concatenate([r["X"] for r in train])                         # 3 rows per image
    y = np.concatenate([r["f"][1:] - r["f"][0] for r in train])
    groups = np.repeat(np.arange(len(train)), 3)
    print(f"{args.sod_model}: {len(train)} training images, {X.shape[1]} features; "
          f"SAM mask better by >.02 on {np.mean([(r['f'][1:] - r['f'][0]).max() > .02 for r in train]):.1%}",
          flush=True)

    def model():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=40, l2_regularization=1.0, random_state=0)

    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(len(train))):
        rows_tr, rows_te = np.isin(groups, tr), np.isin(groups, te)
        oof[rows_te] = model().fit(X[rows_tr], y[rows_tr]).predict(X[rows_te])
    oof = oof.reshape(-1, 3)
    taus = np.linspace(-0.02, 0.2, 45)
    gains = [mean_f(train, choose(oof, t)) for t in taus]
    tau = float(taus[int(np.argmax(gains))])
    base = mean_f(train, np.zeros(len(train), int))
    print(f"out-of-fold mean per-image F: SOD {base:.4f}, selector {max(gains):.4f} at tau={tau:.3f}, "
          f"oracle {np.mean([r['f'].max() for r in train]):.4f}", flush=True)
    final = model().fit(X, y)

    out = {"tau": tau, "train_images": len(train), "oof_mean_f": {"sod": base, "selector": max(gains)},
           "datasets": {}}
    for ds in args.eval_datasets:
        recs = records(args.sod_model, ds, True, args.workers)
        pred = final.predict(np.concatenate([r["X"] for r in recs])).reshape(-1, 3)
        pick = choose(pred, tau)
        oracle = np.array([int(np.argmax(r["f"])) for r in recs])
        res = {name: metrics.aggregate([r["scores"][p] for r, p in zip(recs, picks)])
               for name, picks in (("sod", np.zeros(len(recs), int)), ("selector", pick), ("oracle", oracle))}
        res["replaced"] = float((pick > 0).mean())
        res["replaced_correctly"] = float(np.mean([r["f"][p] >= r["f"][0] for r, p in zip(recs, pick) if p > 0]))\
            if (pick > 0).any() else None
        out["datasets"][ds] = res
        print(f"{ds:10s} " + " | ".join(f"{n}: F={res[n]['Fmax']:.3f} MAE={res[n]['MAE']:.3f} "
                                         f"E={res[n]['E']:.3f} S={res[n]['S']:.3f}" for n in ("sod", "selector", "oracle"))
              + f" | SAM mask used on {res['replaced']:.1%}"
              + (f", right on {res['replaced_correctly']:.0%}" if res["replaced_correctly"] is not None else ""),
              flush=True)
    (OUT / f"{args.sod_model}{args.tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
