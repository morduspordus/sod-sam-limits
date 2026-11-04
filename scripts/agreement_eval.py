"""Predicting the per-pixel AGREEMENT map (fraction of people who include a pixel) with the graph cut's
offset sweep (sodsam/agreement.py), calibrated cross-wise: isotonic maps fitted on one multi-annotator dataset
(--fit) and applied to the other (--test), never fitted on the test set.

Rows (each calibrated by the same isotonic procedure, fitted on the --fit set, on its own input):
  cut sweep        flip offset d*(p) of the sweep of the paper's unary offset (calibrated)
  SOD (cal)        raw saliency value 0..255 (calibrated)
  head (cal)       fusion-head soft map output/<sod>_hz/<ds>/sod/<id>.png (calibrated; skipped unless present
                   for both datasets)
  graph cut        the paper's binary cut as 0/1 (uncalibrated)
  SAM mean (cal)   mean of SAM's three masks, 0, 1/3, 2/3, 1 (calibrated)
  SOD raw          the SOD map min-max normalized, uncalibrated, as the standard benchmark treats it
Metrics against the agreement map A per image, averaged over images: MAE; Pearson correlation (images where
either map is constant are skipped); level-set F: F_beta (beta^2 = 0.3) between {P >= q} and {A >= q} per
q in {0.25, 0.5, 0.75}, averaged over the images where {A >= q} is not empty (counts reported), and the mean of
the three level means; and the benchmark F^max against the standard binary GT (reference).

Flip maps are cached in output/<sod>/<ds>/agreement_flip/<id>.npz (with the grid, the non-nestedness counts and
the sweep time) and reused by the other direction; --recompute ignores the cache. --save writes the calibrated
cut maps to output/<sod>/<test>/agreement_cut/<id>.png (uint8).

  python scripts/agreement_eval.py --sod-model samba --fit MSOD --test PASCAL-S --limit 8
"""
import argparse
import json
import sys
import time
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np

from sodsam import agreement, data, metrics, pipeline
from sodsam.fusion2 import fuse2, prepare

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "agreement"
LEVELS = (0.25, 0.5, 0.75)
CAL = {"cut": "cut sweep", "sod": "SOD (cal)", "head": "head (cal)", "sam": "SAM mean (cal)"}
ROWS = ["cut sweep", "SOD (cal)", "head (cal)", "graph cut", "SAM mean (cal)", "SOD raw"]


def tune_json(sod):
    return ROOT / "results" / "tune2" / f"{sod}_MSOD+MSRA-B1K+MSRA-B4K_mask_zoom_linear_paper_binary_all.json"


def head_dir(sod, ds):
    return data.OUT_ROOT / f"{sod}_hz" / ds / "sod"


def _fit_size(m, shape, interp):
    if m.shape != shape:
        m = cv2.resize(m.astype(np.float32), (shape[1], shape[0]), interpolation=interp).astype(np.float64)
    return m


def cache_key(th, prompt):
    """Identifies what a flip map depends on besides the image: every parameter and the SAM prompt."""
    return f"{prompt}|" + "|".join(f"{k}={th[k]!r}" for k in sorted(th))


def get_flip(sod, ds, id_, x, th, grid, recompute, prompt):
    """Flip map of the sweep, from the cache when its grid and its key (parameters, prompt) match."""
    p = data.run_dir(sod, ds, "agreement_flip") / f"{id_}.npz"
    key = cache_key(th, prompt)
    if p.exists() and not recompute:
        z = np.load(p)
        if ("key" in z and str(z["key"]) == key and z["ds"].shape == grid.shape and np.allclose(z["ds"], grid)):
            return z["flip"], {"nonnested_pixels": int(z["nonnested_pixels"]), "nonnested_pairs": int(z["nonnested_pairs"]),
                               "time": float(z["time"]), "cached": True}
    flip, info = agreement.sweep_offsets(x, th, grid)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, flip=flip, ds=grid, key=np.array(key), nonnested_pixels=info["nonnested_pixels"],
                        nonnested_pairs=info["nonnested_pairs"], time=info["time"])
    return flip, {**{k: info[k] for k in ("nonnested_pixels", "nonnested_pairs", "time")}, "cached": False}


def _image(id_, ds, sod, th, prompt, grid, recompute, use_head):
    """Inputs of one image: values of the calibrated rows, the agreement map, the GT and the sweep info."""
    if not agreement.has_answers(ds, id_):
        return None
    inp = pipeline.load_inputs(sod, ds, prompt, id_)
    gt = inp["gt"]
    A = agreement.agreement_maps(ds, id_, gt.shape)
    if A is None:
        return None
    x = prepare(inp)
    flip, info = get_flip(sod, ds, id_, x, th, grid, recompute, prompt)
    vals = {"cut": _fit_size(agreement.flip_offset(flip, grid), gt.shape, cv2.INTER_NEAREST),
            "sod": _fit_size(inp["sod"].astype(np.float64), gt.shape, cv2.INTER_LINEAR),
            "sam": _fit_size(x["n"] / x["k"], gt.shape, cv2.INTER_NEAREST)}
    if use_head and (head_dir(sod, ds) / f"{id_}.png").exists():
        vals["head"] = _fit_size(data.read_png(head_dir(sod, ds) / f"{id_}.png").astype(np.float64), gt.shape,
                                 cv2.INTER_LINEAR)
    gc = fuse2(x, "linear", "paper", "binary", th) > 0
    j0 = int(np.argmin(np.abs(grid - th["d"])))
    info["gc_vs_sweep_pixels"] = int((gc != (flip >= j0)).sum())       # the cut at d0 recovered by the sweep
    return {"x": x, "inp": inp, "vals": vals, "A": A, "gt": gt, "gc": _fit_size(gc.astype(np.float64), gt.shape,
                                                                               cv2.INTER_NEAREST), "info": info}


def fit_worker(id_, n_sub, seed, **kw):
    r = _image(id_, **kw)
    if r is None:
        return None
    A = r["A"].ravel()
    rng = np.random.default_rng([seed, zlib.crc32(id_.encode())])
    idx = rng.choice(A.size, min(n_sub, A.size), replace=False)
    return {"id": id_, "A": A[idx], "vals": {k: v.ravel()[idx] for k, v in r["vals"].items()}, "info": r["info"]}


def level_f(P, A, q):
    g = A >= q - 1e-9
    if not g.any():
        return None
    return agreement_binary_f(P >= q - 1e-9, g)


def agreement_binary_f(m, g):
    tp = np.logical_and(m, g).sum()
    if tp == 0:
        return 0.0
    p, r = tp / m.sum(), tp / g.sum()
    return float(1.3 * p * r / (0.3 * p + r))


def scores(P, A, gt, u8=None, P_l1=None):
    """Per-image metrics of a map P in [0, 1]; u8 = the uint8 map for the benchmark F curve (default round(255 P));
    P_l1 = the same input under the least-absolute-deviations calibration, used for MAE (a least-squares
    calibration is not the MAE minimizer, which would bias MAE against the calibrated rows); MSE uses P."""
    pear = None
    if P.std() > 0 and A.std() > 0:
        pear = float(np.corrcoef(P.ravel(), A.ravel())[0, 1])
    if u8 is None:
        u8 = np.round(np.clip(P, 0, 1) * 255).astype(np.uint8)
    return {"mae": float(np.abs((P if P_l1 is None else P_l1) - A).mean()), "mse": float(((P - A) ** 2).mean()),
            "pearson": pear, "levels": {q: level_f(P, A, q) for q in LEVELS}, "f_curve": metrics.f_curve(u8, gt)}


def test_worker(id_, cals, save, sod, **kw):
    r = _image(id_, sod=sod, **kw)
    if r is None:
        return None
    A, gt = r["A"], r["gt"]
    res = {}
    for key, row in CAL.items():
        if key in r["vals"] and key in cals:
            P = agreement.apply_isotonic(cals[key]["l2"], r["vals"][key])
            res[row] = scores(P, A, gt, P_l1=agreement.apply_isotonic(cals[key]["l1"], r["vals"][key]))
            if key == "cut" and save:
                data.write_png(data.run_dir(sod, kw["ds"], "agreement_cut") / f"{id_}.png",
                               np.round(np.clip(P, 0, 1) * 255).astype(np.uint8))
    res["graph cut"] = scores(r["gc"], A, gt)
    raw = r["inp"]["sod"]
    raw = raw if raw.shape == gt.shape else cv2.resize(raw, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    res["SOD raw"] = scores(metrics._minmax(raw.astype(np.float64)), A, gt, u8=raw)
    return {"id": id_, "rows": res, "info": r["info"]}


def _map(fn, ids, workers):
    if workers <= 1:
        return [r for r in map(fn, ids) if r]
    with Pool(workers) as pool:
        return [r for r in pool.imap(fn, ids, chunksize=2) if r]


def aggregate(recs, row):
    rs = [r["rows"][row] for r in recs if row in r["rows"]]
    if not rs:
        return None
    pear = [s["pearson"] for s in rs if s["pearson"] is not None]
    lv = {}
    for q in LEVELS:
        v = [s["levels"][q] for s in rs if s["levels"][q] is not None]
        lv[str(q)] = {"F": float(np.mean(v)) if v else None, "n": len(v)}
    good = [lv[str(q)]["F"] for q in LEVELS if lv[str(q)]["F"] is not None]
    return {"n": len(rs), "MAE": float(np.mean([s["mae"] for s in rs])), "MSE": float(np.mean([s["mse"] for s in rs])),
            "pearson": float(np.mean(pear)) if pear else None, "pearson_n": len(pear),
            "levelF": float(np.mean(good)) if good else None, "levels": lv,
            "Fmax": float(np.mean([s["f_curve"] for s in rs], axis=0).max())}


def _ids(ds, limit):
    ids = data.list_ids(ds)
    return ids[:limit] if limit else ids


def _sweep_stats(recs):
    infos = [r["info"] for r in recs]
    fresh = [i["time"] for i in infos if not i["cached"]]
    return {"images": len(infos), "nonnested_pixels": int(sum(i["nonnested_pixels"] for i in infos)),
            "nonnested_pairs": int(sum(i["nonnested_pairs"] for i in infos)),
            "images_with_nonnested": int(sum(i["nonnested_pixels"] > 0 for i in infos)),
            "gc_vs_sweep_pixels": int(sum(i["gc_vs_sweep_pixels"] for i in infos)),
            "sweep_time_mean": float(np.mean([i["time"] for i in infos])) if infos else None,
            "sweeps_computed": len(fresh), "sweep_time_mean_computed": float(np.mean(fresh)) if fresh else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--fit", required=True, choices=["MSOD", "PASCAL-S"])
    ap.add_argument("--test", required=True, choices=["MSOD", "PASCAL-S"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N images of both the fitting and the test set")
    ap.add_argument("--n-sub", type=int, default=20000, help="random pixels per image for the isotonic fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--recompute", action="store_true", help="ignore the cached flip maps")
    args = ap.parse_args()
    if args.fit == args.test:
        raise SystemExit("--fit and --test must differ (never fit the calibration on the test set)")
    res = json.loads(tune_json(args.sod_model).read_text())
    assert tuple(res["model"]) == ("linear", "paper", "binary"), res["model"]
    th, prompt = res["params"], res["prompt"]
    grid = agreement.default_offsets(th)
    use_head = all(head_dir(args.sod_model, d).is_dir() and any(head_dir(args.sod_model, d).glob("*.png"))
                   for d in (args.fit, args.test))
    common = dict(sod=args.sod_model, th=th, prompt=prompt, grid=grid, recompute=args.recompute, use_head=use_head)
    t0 = time.time()

    fit_recs = _map(partial(fit_worker, n_sub=args.n_sub, seed=args.seed, ds=args.fit, **common),
                    _ids(args.fit, args.limit), args.workers)
    cals = {}
    for key in CAL:
        rs = [r for r in fit_recs if key in r["vals"]]
        if rs:
            v, a = np.concatenate([r["vals"][key] for r in rs]), np.concatenate([r["A"] for r in rs])
            cals[key] = {loss: agreement.fit_isotonic(v, a, loss=loss) for loss in ("l2", "l1")}
    print(f"fit on {args.fit}: {len(fit_recs)} images, {time.time() - t0:.0f} s; calibrated rows: "
          + ", ".join(CAL[k] for k in cals), flush=True)
    kw = dict(common)
    sod = kw.pop("sod")
    test_recs = _map(partial(test_worker, cals=cals, save=args.save, sod=sod, ds=args.test, **kw),
                     _ids(args.test, args.limit), args.workers)
    agg = {row: aggregate(test_recs, row) for row in ROWS}
    agg = {k: v for k, v in agg.items() if v is not None}

    sweep = {args.fit: _sweep_stats(fit_recs), args.test: _sweep_stats(test_recs)}
    print(f"\n{args.sod_model}: calibration fitted on {args.fit} ({len(fit_recs)} images), tested on {args.test} "
          f"({len(test_recs)} images); grid {len(grid)} offsets {grid[0]:.1f} .. {grid[-1]:.1f} (step "
          f"{grid[1] - grid[0]:.1f}, paper d {th['d']:.1f})")
    hdr = (f"{'':16s}{'MAE':>7s}{'MSE':>8s}{'Pearson':>9s}{'(n)':>6s}{'levelF':>8s}"
           + "".join(f"{'F@' + str(q):>8s}" for q in LEVELS) + f"{'Fmax':>7s}{'n':>5s}")
    print(hdr)
    fmt = lambda v: "   -  " if v is None else f"{v:.3f}"
    for row, a in agg.items():
        print(f"{row:16s}{a['MAE']:7.3f}{a['MSE']:8.4f}{fmt(a['pearson']):>9s}{a['pearson_n']:6d}{fmt(a['levelF']):>8s}"
              + "".join(f"{fmt(a['levels'][str(q)]['F']):>8s}" for q in LEVELS) + f"{a['Fmax']:7.3f}{a['n']:5d}")
    first = next(iter(agg.values()))
    print("images with a non-empty level set {A >= q}: " + ", ".join(f"q={q}: {first['levels'][str(q)]['n']}" for q in LEVELS)
          + f" of {first['n']}. MAE of the calibrated rows uses the least-absolute-deviations calibration, the other"
          " metrics the least-squares one.")
    if args.test == "MSOD":
        print("CAVEAT: MSOD's binary GT was seen by the cut parameters (tuned on MSOD + MSRA-B1K + MSRA-B4K) and by "
              "the head's epoch selection; the SOD and SAM rows are clean. PASCAL-S is the clean test direction.")
    for d, s in sweep.items():
        print(f"sweep {d}: {s['images']} images, non-nested pixels {s['nonnested_pixels']} (pairs {s['nonnested_pairs']}, "
              f"{s['images_with_nonnested']} images), paper cut vs sweep at d0: {s['gc_vs_sweep_pixels']} pixels, "
              f"mean sweep time {fmt(s['sweep_time_mean'])} s ({s['sweeps_computed']} computed now)")
    out = {"sod_model": args.sod_model, "fit": args.fit, "test": args.test, "args": vars(args), "tune_json": str(tune_json(args.sod_model)),
           "params": th, "grid": grid.tolist(), "rows": agg, "sweep": sweep,
           "calibration": {CAL[k]: {loss: {"x": m["x"].tolist(), "y": m["y"].tolist()} for loss, m in v.items()}
                           for k, v in cals.items()},
           "caveat": ("MSOD GT seen by the cut parameters and the head's epoch selection; PASCAL-S is the clean test "
                      "direction" if args.test == "MSOD" else None),
           "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_limit{args.limit}" if args.limit else ""
    p = OUT / f"{args.sod_model}_fit-{args.fit}_test-{args.test}{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"written {p}")


if __name__ == "__main__":
    main()
