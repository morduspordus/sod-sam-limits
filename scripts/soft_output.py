"""Calibrated soft output of the graph cut: the cut is binary, so where the model is uncertain a soft value
lowers MAE and raises E without touching F^max (the cut stays a level set of the output).

  (A) band:   within r px of the cut's boundary (distance transforms of the mask and of its complement)
              out = alpha * gc + (1 - alpha) * s, the cut elsewhere; s = max-normalized SOD map in 0..255.
              r, alpha from a grid on the tuning sets, objective = mean of F^max, E, S and 1 - MAE.
              The cut stays a level set (threshold 128) only for alpha >= 1/2; below it the band values of
              the two sides overlap and F^max can move either way.
  (B) nested: 255 inside the inner answer, s between the inner and the outer answer (eval_ideas.py nested
              maps, <id>_in.png / _out.png), 0 outside.
The binary cut is <sod>/<ds>/gc_zoom/<id>.png, recomputed from --gc-params where it is not saved (the
tuning sets; pipeline.method_output reproduces gc_zoom exactly).
Maps: <sod>/<ds>/gc_soft/<id>.png (A) and gc_nestsoft/<id>.png (B); results/eval_<sod>_soft.json
({"sod_model", "params": {"r", "alpha"}, "grid", "datasets": {ds: {"gc", "soft_band", "soft_nested"}}}).

  python scripts/soft_output.py --sod-model samba --workers 14
"""
import argparse
import json
import sys
import time
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from sodsam import data, metrics, pipeline

ROOT = Path(__file__).resolve().parents[1]
GRID_R = (2, 4, 8, 16, 32)
GRID_ALPHA = (0.0, 0.25, 0.5, 0.75)
OBJECTIVE_KEYS = ("Fmax", "E", "S")   # plus 1 - MAE


def soft_band(gc, s, r, alpha):
    """(A): uint8 map; gc boolean cut, s uint8 (0..255)."""
    out = gc.astype(np.float64) * 255
    if gc.any() and not gc.all():
        dist = distance_transform_edt(gc) + distance_transform_edt(~gc)   # one term is 0 at every pixel
        band = dist <= r
        out[band] = alpha * out[band] + (1 - alpha) * s[band]
    return np.round(out).astype(np.uint8)


def soft_nested(inner, outer, s):
    """(B): uint8 map; inner and outer boolean answers, inner within outer."""
    return np.where(inner, 255, np.where(outer, s, 0)).astype(np.uint8)


def load_cut(sod, ds, id_, gc_params, prompt):
    """Boolean cut from gc_zoom, or recomputed from the tuned model json."""
    p = data.run_dir(sod, ds, "gc_zoom") / f"{id_}.png"
    if p.exists():
        return data.read_png(p) > 0
    inp = pipeline.load_inputs(sod, ds, prompt, id_)
    return pipeline.method_output(f"v2:{gc_params}", inp, None) > 0


def load_s(sod, ds, id_):
    s = data.read_png(data.run_dir(sod, ds, "sod") / f"{id_}.png").astype(np.float64)
    return np.round(255.0 * s / (s.max() + 1e-13)).astype(np.uint8)


def _tune_image(key, sod, gc_params, prompt):
    """gc scores and the scores of every grid point for one tuning image."""
    ds, id_ = key
    gc, s, gt = load_cut(sod, ds, id_, gc_params, prompt), load_s(sod, ds, id_), data.load_gt(ds, id_)
    out = {"gc": metrics.image_scores(gc.astype(np.uint8) * 255, gt)}
    for r, alpha in product(GRID_R, GRID_ALPHA):
        out[r, alpha] = metrics.image_scores(soft_band(gc, s, r, alpha), gt)
    return out


def _eval_image(key, sod, gc_params, prompt, r, alpha):
    """Write gc_soft (and gc_nestsoft when the nested maps exist) and score gc, (A) and (B)."""
    ds, id_ = key
    gc, s, gt = load_cut(sod, ds, id_, gc_params, prompt), load_s(sod, ds, id_), data.load_gt(ds, id_)
    band = soft_band(gc, s, r, alpha)
    data.write_png(data.run_dir(sod, ds, "gc_soft") / f"{id_}.png", band)
    out = {"gc": metrics.image_scores(gc.astype(np.uint8) * 255, gt),
           "soft_band": metrics.image_scores(band, gt), "soft_nested": None}
    nested = data.run_dir(sod, ds, "nested")
    if (nested / f"{id_}_in.png").exists():
        inner, outer = (data.read_png(nested / f"{id_}_{k}.png") > 0 for k in ("in", "out"))
        nest = soft_nested(inner, outer | inner, s)
        data.write_png(data.run_dir(sod, ds, "gc_nestsoft") / f"{id_}.png", nest)
        out["soft_nested"] = metrics.image_scores(nest, gt)
    return out


def objective(agg):
    return float(np.mean([agg[k] for k in OBJECTIVE_KEYS] + [1 - agg["MAE"]]))


def keys_of(datasets, limit):
    return [(ds, i) for ds in datasets for i in (data.list_ids(ds)[:limit] if limit else data.list_ids(ds))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--tune-datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--gc-params", default=None,
                    help="tune2 json of the cut (default: the tuned zoom model of --sod-model)")
    ap.add_argument("--r", type=float, default=None, help="with --alpha: skip the grid search")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="images per dataset (smoke tests)")
    args = ap.parse_args()
    sod = args.sod_model
    gc_params = args.gc_params or str(
        ROOT / "results" / "tune2" / f"{sod}_MSOD+MSRA-B1K+MSRA-B4K_mask_zoom_linear_paper_binary_all.json")
    if not Path(gc_params).exists():
        raise FileNotFoundError(gc_params)
    out_path = ROOT / "results" / f"eval_{sod}_soft.json"
    res = {"sod_model": sod, "gc_params": gc_params, "params": None, "grid": {}, "datasets": {}}

    t0 = time.time()
    with Pool(args.workers) as pool:
        if args.r is not None and args.alpha is not None:
            r, alpha = args.r, args.alpha
        else:
            keys = keys_of(args.tune_datasets, args.limit)
            f = partial(_tune_image, sod=sod, gc_params=gc_params, prompt=args.prompt)
            per = list(tqdm(pool.imap(f, keys, chunksize=8), total=len(keys), desc="tune " + "+".join(args.tune_datasets)))
            best = None
            for k in ["gc"] + list(product(GRID_R, GRID_ALPHA)):
                agg = metrics.aggregate([p[k] for p in per])
                name = "gc" if k == "gc" else f"r={k[0]:g},alpha={k[1]:g}"
                res["grid"][name] = {**agg, "objective": objective(agg)}
                print(metrics.format_row(f"tune {name}", agg) + f" obj={objective(agg):.4f}", flush=True)
                if k != "gc" and (best is None or objective(agg) > best[0]):
                    best = (objective(agg), k)
            r, alpha = best[1]
            print(f"best r={r:g} alpha={alpha:g} (objective {best[0]:.4f}, gc {res['grid']['gc']['objective']:.4f}, "
                  f"{time.time() - t0:.0f} s)", flush=True)
        res["params"] = {"r": r, "alpha": alpha}

        for ds in args.datasets:
            keys = keys_of([ds], args.limit)
            f = partial(_eval_image, sod=sod, gc_params=gc_params, prompt=args.prompt, r=r, alpha=alpha)
            per = list(tqdm(pool.imap(f, keys, chunksize=8), total=len(keys), desc=ds))
            res["datasets"][ds] = {m: metrics.aggregate([p[m] for p in per])
                                   for m in ("gc", "soft_band", "soft_nested") if all(p[m] is not None for p in per)}
            for m, agg in res["datasets"][ds].items():
                print(metrics.format_row(f"{ds} {m}", agg), flush=True)
            out_path.parent.mkdir(exist_ok=True)
            out_path.write_text(json.dumps(res, indent=1))
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
