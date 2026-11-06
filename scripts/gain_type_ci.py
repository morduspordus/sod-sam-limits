"""Bootstrap intervals for the boundary share of the fusion's gain (Table suppl_gain_type; second review).

review_tables.py's gain table gives, per error type of the SOD map, the share of the graph cut's (and the head's)
reduction of the summed loss sum (1 - F), F at each map's F^max threshold (review_eval.py --report). Where the
total reduction is small, these shares are noisy. Here, per image of each benchmark, the F curves of the SOD map,
the graph cut (output/<sod>/<ds>/gc_zoom) and the head (output/<sod>_hz/<ds>/sod) and the SOD map's error type
(error_types.classify on the map at half its maximum, as in review_eval.py) are recomputed; the per-type losses
are checked against import/review/report_<sod>.json; and the boundary share
  (loss_sod(boundary) - loss_m(boundary)) / (loss_sod(all) - loss_m(all)),  m = gc or head,
gets a 95% percentile interval from --boot resamples of the images (each resample re-estimates every map's
F^max threshold). Images with an empty ground truth are left out of the sums, as in the table.
Output: results/gain_type_ci.json (per SOD model and benchmark: point share, interval, the total reduction as a
fraction of the SOD map's loss, the fraction of resamples with a share below 50%, the per-type check against the
report, and per-image arrays: id, type, loss of each map at its F^max threshold).

  python scripts/gain_type_ci.py --workers 2
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from error_types import CATS, classify
from sodsam import data, metrics

ROOT = Path(__file__).resolve().parents[1]
MAPS = ("sod", "gc", "head")
SODS = ("selfreformer", "samba")


def binarize(m):
    """The map at half its maximum, as review_eval.binarize."""
    return m.astype(np.float64) / m.max() >= 127 / 255 if m.max() > 0 else np.zeros(m.shape, bool)


def _image(id_, sod_model, ds):
    gt = data.load_gt(ds, id_)
    maps = {"sod": data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png"),
            "gc": data.read_png(data.run_dir(sod_model, ds, "gc_zoom") / f"{id_}.png"),
            "head": data.read_png(data.OUT_ROOT / f"{sod_model}_hz" / ds / "sod" / f"{id_}.png")}
    curves = np.stack([metrics.f_curve(maps[k], gt) for k in MAPS]).astype(np.float32)
    cat = classify(binarize(maps["sod"]), gt)[0] if gt.any() else "empty"
    return id_, cat, curves


def _boot(curves, bnd, valid, boot, rng):
    """Boundary share of each map's loss reduction, for the full sample and for `boot` resamples.
    curves: maps x n x 256; bnd, valid: n booleans (boundary type; non-empty GT)."""
    n = curves.shape[1]
    w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(boot)]).astype(np.float64)
    w = np.vstack([np.ones(n), w])                                   # row 0: the sample itself
    sums = {}
    for j, k in enumerate(MAPS):
        c = curves[j].astype(np.float64)
        istar = np.argmax(w @ c, axis=1)                             # F^max threshold of every resample
        loss = 1.0 - c[:, istar]                                     # n x (boot + 1)
        sums[k] = {"all": np.einsum("bi,ib->b", w * valid, loss), "bnd": np.einsum("bi,ib->b", w * bnd, loss)}
    out = {}
    for k in ("gc", "head"):
        red = sums["sod"]["all"] - sums[k]["all"]
        share = (sums["sod"]["bnd"] - sums[k]["bnd"]) / red
        lo, hi = np.percentile(share[1:], [2.5, 97.5])
        out[k] = {"share": float(share[0]), "ci": [float(lo), float(hi)],
                  "rel_reduction": float(red[0] / sums["sod"]["all"][0]),
                  "p_below_half": float(np.mean(share[1:] < 0.5)),
                  "robust_below_half": bool(hi < 0.5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sods", nargs="+", default=list(SODS))
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = {"boot": args.boot, "seed": args.seed}
    with Pool(args.workers) as pool:
        for sod in args.sods:
            rep = json.loads((ROOT / "import" / "review" / f"report_{sod}.json").read_text())
            out[sod] = {}
            for ds in args.datasets:
                ids, cats, curves = zip(*pool.map(partial(_image, sod_model=sod, ds=ds), data.list_ids(ds),
                                                  chunksize=16))
                curves = np.stack(curves, axis=1)                    # maps x n x 256
                cats = np.array(cats)
                istar = np.argmax(curves.mean(axis=1), axis=1)
                loss = {k: 1.0 - curves[j][:, istar[j]].astype(np.float64) for j, k in enumerate(MAPS)}
                ref = rep[ds]["_loss_by_sod_type"]
                check = {c: {k: float(loss[k][cats == c].sum() - ref[c]["loss"][k]) for k in MAPS}
                         for c in CATS if c in ref}
                nmismatch = {c: int((cats == c).sum()) - ref[c]["n"] for c in CATS if c in ref}
                res = _boot(curves, cats == "boundary", cats != "empty", args.boot,
                            np.random.default_rng(args.seed))
                res["check_vs_report"] = {"loss_diff": check, "n_diff": nmismatch,
                                          "max_abs_loss_diff": max(abs(v) for d in check.values() for v in d.values())}
                res["images"] = {"id": list(ids), "type": cats.tolist(),
                                 **{f"loss_{k}": np.round(loss[k], 6).tolist() for k in MAPS}}
                out[sod][ds] = res
                print(f"{sod:12s} {ds:10s} check max |diff| {res['check_vs_report']['max_abs_loss_diff']:.2e} | "
                      + " | ".join(f"{k}: share {res[k]['share']:+.2f} [{res[k]['ci'][0]:+.2f}, {res[k]['ci'][1]:+.2f}]"
                                   f" red {res[k]['rel_reduction']:.1%}" for k in ("gc", "head")), flush=True)
    (ROOT / "results" / "gain_type_ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
