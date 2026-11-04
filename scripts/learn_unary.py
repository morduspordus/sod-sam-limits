"""Fit the learned log-odds unary (sodsam/unary_learn.py: table | gbt) and write its per-image maps.

  python scripts/learn_unary.py --sod-model selfreformer --model table --fit-datasets MSRA-B4K \
      --write-datasets MSOD MSRA-B1K DUTS-TE ECSSD DUT-OMRON PASCAL-S HKU-IS --workers 16

The model is fitted on all pixels (table: np.bincount of the cell indices per image, summed) or on a
boundary-dense stratified pixel sample (gbt: --sample pixels in all, half within 16 px of a boundary,
--heldout fraction of the images held out for early stopping on the staged log loss, then refitted with
the best iteration count) of the fit datasets and saved to results/unary/<sod>_<model>.{npz,joblib};
an existing file is loaded instead (--refit forces a new fit). Maps go to output/<sod>/<ds>/lo_<model>/.
Diagnostic per written dataset: per-pixel accuracy and F/MAE/E/S of the thresholded log-odds lo > 0 next
to the binarized SOD map (s >= 0.5 after max normalization), i.e. whether the learned unary alone beats
the SOD map before any cut; saved to results/unary/<sod>_<model>_diag.json.
--limit N uses the first N ids of every dataset (smoke tests).
"""
import argparse
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics, pipeline, unary_learn as ul

OUT = Path(__file__).resolve().parents[1] / "results" / "unary"
WRITE_DATASETS = ["MSOD", "MSRA-B1K", "DUTS-TE", "ECSSD", "DUT-OMRON", "PASCAL-S", "HKU-IS"]


def _init(sod_model, datasets, prompt, model=None):
    """Per-worker: SAM scores of the datasets and the fitted model (table log-odds or classifier)."""
    global SCORES, MODEL
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)                      # the workers are the parallelism, not OpenMP
    SCORES = {ds: pipeline.load_scores(sod_model, ds, prompt) for ds in datasets}
    MODEL = model


def _features(key, sod_model, prompt):
    ds, id_ = key
    inp = pipeline.load_inputs(sod_model, ds, prompt, id_, SCORES[ds])
    return inp, ul.features(inp)


def _counts(key, sod_model, prompt):
    inp, f = _features(key, sod_model, prompt)
    return ul.table_counts(f, inp["gt"])


def _sample(key, sod_model, prompt, n, seed):
    """(X, y) of a stratified pixel sample of one image (seed: per-image, recorded in the model)."""
    inp, f = _features(key, sod_model, prompt)
    idx = ul.sample_pixels(f, np.random.default_rng(seed), n)
    return ul.feature_matrix(f, idx), inp["gt"].ravel()[idx]


def _write(key, sod_model, prompt, model_name):
    """Writes the log-odds map of one image; returns the diagnostic of lo > 0 and of the SOD mask."""
    ds, id_ = key
    inp, f = _features(key, sod_model, prompt)
    lo = ul.table_predict(MODEL, f) if model_name == "table" else ul.gbt_predict(MODEL, f)
    ul.write_lo(sod_model, ds, model_name, id_, lo)
    gt = inp["gt"]
    out = {}
    for name, pred in (("lo", lo > 0), ("sod", f["s"] >= 0.5)):
        out[name] = metrics.image_scores(pred.astype(np.uint8) * 255, gt)
        out[name]["acc"] = float((pred == gt).mean())
    return out


def keys_of(datasets, limit):
    return [(ds, i) for ds in datasets for i in data.list_ids(ds)[:limit]]


def fit_table(args, keys, pool):
    n1 = np.zeros(ul.N_CELLS, np.int64)
    n0 = np.zeros(ul.N_CELLS, np.int64)
    f = partial(_counts, sod_model=args.sod_model, prompt=args.prompt)
    for a, b in pool.imap_unordered(f, keys, chunksize=8):
        n1 += a
        n0 += b
    meta = {"sod_model": args.sod_model, "prompt": args.prompt, "fit_datasets": args.fit_datasets,
            "n_images": len(keys)}
    ul.save_table(OUT / f"{args.sod_model}_table.npz", n1, n0, meta)
    lo = ul.table_logodds(n1, n0)
    print(f"table: {len(keys)} images, {(n1 + n0).sum() / 1e6:.0f} M pixels, {((n1 + n0) > 0).sum()} of "
          f"{ul.N_CELLS} cells seen, |lo| clipped in {(np.abs(lo) >= ul.LO_CLIP).sum()} cells", flush=True)
    return lo


def fit_gbt(args, keys, pool):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(keys))
    n_held = max(1, int(round(args.heldout * len(keys))))
    held = [keys[i] for i in sorted(perm[:n_held])]
    train = [keys[i] for i in sorted(perm[n_held:])]
    per_image = int(np.ceil(args.sample / len(train)))
    sampling = {"sample": args.sample, "per_image": per_image, "near_frac": 0.5, "near_px": ul.NEAR_PX,
                "seed": args.seed, "n_train_images": len(train), "n_heldout_images": len(held),
                "heldout_ids": held}

    def collect(ks, offset):
        out = [pool.apply_async(_sample, (k, args.sod_model, args.prompt, per_image, args.seed + offset + j))
               for j, k in enumerate(ks)]
        Xy = [r.get() for r in out]
        return np.concatenate([x for x, _ in Xy]), np.concatenate([y for _, y in Xy])

    t0 = time.time()
    X, y = collect(train, 1)
    Xv, yv = collect(held, 1 + len(train))
    print(f"gbt: {len(X)} training pixels of {len(train)} images, {len(Xv)} held-out pixels of {len(held)} "
          f"images, {len(ul.FEATURES)} features, positives {y.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

    def model(n_iter):
        return HistGradientBoostingClassifier(max_iter=n_iter, learning_rate=0.1, max_leaf_nodes=31,
                                              early_stopping=False, random_state=args.seed)
    clf = model(args.max_iter).fit(X, y)
    losses = [log_loss(yv, p[:, 1], labels=[False, True]) for p in clf.staged_predict_proba(Xv)]
    best = int(np.argmin(losses)) + 1
    print(f"gbt: held-out log loss {losses[0]:.4f} (1 iter) -> {min(losses):.4f} (best {best} of {len(losses)}), "
          f"final {losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    if best < clf.n_iter_:
        clf = model(best).fit(X, y)
    acc = float((clf.predict(Xv) == yv).mean())
    print(f"gbt: held-out per-pixel accuracy {acc:.4f} on the stratified sample (boundary-dense), "
          f"{best} iterations", flush=True)
    importance = None
    if args.importance:
        from sklearn.inspection import permutation_importance
        sub = rng.choice(len(Xv), min(50000, len(Xv)), replace=False)
        imp = permutation_importance(clf, Xv[sub], yv[sub], n_repeats=3, random_state=args.seed,
                                     scoring="neg_log_loss")
        importance = dict(zip(ul.FEATURES, map(float, imp.importances_mean)))
        order = sorted(importance, key=importance.get, reverse=True)
        print("gbt: permutation importance (log loss): " + " ".join(f"{k}={importance[k]:.4f}" for k in order),
              flush=True)
    import joblib
    joblib.dump({"clf": clf, "features": ul.FEATURES, "sampling": sampling, "n_iter": best,
                 "heldout_losses": losses, "heldout_acc": acc, "importance": importance,
                 "sod_model": args.sod_model, "prompt": args.prompt, "fit_datasets": args.fit_datasets},
                OUT / f"{args.sod_model}_gbt.joblib")
    return clf


def load_model(args):
    if args.model == "table":
        return ul.load_table(OUT / f"{args.sod_model}_table.npz")
    import joblib
    return joblib.load(OUT / f"{args.sod_model}_gbt.joblib")["clf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--model", required=True, choices=["table", "gbt"])
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--fit-datasets", nargs="+", default=["MSRA-B4K"])
    ap.add_argument("--write-datasets", nargs="+", default=WRITE_DATASETS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids per dataset (smoke tests)")
    ap.add_argument("--refit", action="store_true", help="fit even when the model file exists")
    ap.add_argument("--sample", type=int, default=3_000_000, help="gbt: training pixels in all")
    ap.add_argument("--max-iter", type=int, default=400, help="gbt: boosting iterations before early stopping")
    ap.add_argument("--heldout", type=float, default=0.1, help="gbt: fraction of the fit images held out")
    ap.add_argument("--importance", action="store_true", help="gbt: permutation importance on 50k held-out pixels")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.sod_model}_{args.model}.{'npz' if args.model == 'table' else 'joblib'}"
    t0 = time.time()

    if path.exists() and not args.refit:
        model = load_model(args)
        print(f"loaded {path}", flush=True)
    else:
        keys = keys_of(args.fit_datasets, args.limit)
        with Pool(args.workers, initializer=_init, initargs=(args.sod_model, args.fit_datasets, args.prompt)) as pool:
            model = fit_table(args, keys, pool) if args.model == "table" else fit_gbt(args, keys, pool)
        print(f"saved {path} ({time.time() - t0:.0f}s)", flush=True)

    diag = {"sod_model": args.sod_model, "model": args.model, "prompt": args.prompt, "limit": args.limit,
            "datasets": {}}
    f = partial(_write, sod_model=args.sod_model, prompt=args.prompt, model_name=args.model)
    with Pool(args.workers, initializer=_init, initargs=(args.sod_model, args.write_datasets, args.prompt, model)) as pool:
        for ds in args.write_datasets:
            keys = keys_of([ds], args.limit)
            per = pool.map(f, keys, chunksize=4)
            res = {}
            for name in ("lo", "sod"):
                res[name] = metrics.aggregate([p[name] for p in per])
                res[name]["acc"] = float(np.mean([p[name]["acc"] for p in per]))
                print(metrics.format_row(f"{ds} {name}", res[name]) + f" acc={res[name]['acc']:.4f}", flush=True)
            diag["datasets"][ds] = res
    (OUT / f"{args.sod_model}_{args.model}_diag.json").write_text(json.dumps(diag, indent=1))
    print(f"wrote lo_{args.model} maps of {' '.join(args.write_datasets)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
