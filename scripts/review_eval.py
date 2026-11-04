"""Experiments requested by the reviews, on the cached SOD / SAM / fusion outputs (no GPU).

Per image of each test set, for these maps:
  sod          the SOD map
  gc           the paper's graph cut (output/<sod>/<ds>/gc_zoom)
  head         the paper's fusion head (output/<sod>_hz/<ds>/sod)
  sam_conf     SAM's most confident mask
  rule_major   untuned rule "SOD mask or at least two of the three SAM masks" (SOD mask: s >= 1/2)
  rule_samiou  untuned rule: the SAM mask with the highest IoU with the SOD mask
  rule_avg     untuned rule: mean of the SOD map and the mean of SAM's three probability maps (the
               masks where the probabilities are missing)
  transfer     the graph cut with the parameters tuned for the other SOD model (--transfer json)
it stores the standard scores (metrics.image_scores) and, for the map binarized at each threshold t of
THRESHOLDS (on the min-max normalized 0..255 scale of metrics.f_curve, pixel >= t):
  boundary     boundary F-measure (BF, tolerance 3 px) and trimap IoU in bands of 5 and 20 px around the
               GT contour
  errors       wrong pixels within 5 px of the GT contour and elsewhere
and the error type of scripts/error_types.py (classify) of the map at half its maximum, as in the paper.
The report (--report) uses, for each map and dataset, its F^max threshold (the argmax of the mean F curve;
the nearest of THRESHOLDS for the boundary measures), so that every map is binarized as F^max scores it:
the score tables, the boundary measures, the share of wrong pixels near the contour, and the loss
1 - F at that threshold summed per error type of the SOD map (images grouped by the SOD map's type, as in
the paper's Table 14) for each map; these sums add up to n (1 - F^max).

  python scripts/review_eval.py --sod-model samba --transfer results/tune2/selfreformer_..._all.json
  python scripts/review_eval.py --sod-model samba --report
"""
import argparse
import json
import pickle
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tqdm import tqdm

from error_types import CATS, classify
from sodsam import data, metrics, pipeline
from sodsam.fusion2 import fuse2, prepare

OUT = Path(__file__).resolve().parents[1] / "results" / "review"
MAPS = ("sod", "gc", "head", "sam_conf", "rule_major", "rule_samiou", "rule_avg", "transfer")
BF_TOL = 3
BANDS = (5, 20)
NEAR = 5
THRESHOLDS = (64, 128, 192, 216, 232, 243, 250)


def binarize(m):
    """The map at half its maximum (the SOD mask of the paper; 0/255 maps are unchanged)."""
    return m.astype(np.float64) / m.max() >= 127 / 255 if m.max() > 0 else np.zeros(m.shape, bool)


def contour(mask):
    """Pixels of a boolean mask with a 4-neighbour of the other label (both sides of the boundary)."""
    b = np.zeros(mask.shape, bool)
    jr, jd = mask[:, 1:] != mask[:, :-1], mask[1:] != mask[:-1]
    b[:, 1:] |= jr
    b[:, :-1] |= jr
    b[1:] |= jd
    b[:-1] |= jd
    return b


def dist_to(b):
    """Euclidean distance of every pixel to the True pixels of b (inf if there are none)."""
    if not b.any():
        return np.full(b.shape, np.inf)
    return cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def boundary_scores(P, G, dG):
    """BF (tolerance BF_TOL), trimap IoU per band, wrong pixels near / far from the GT contour."""
    bP = contour(P)
    bG = dG == 0
    if bP.any() and bG.any():
        prec = float((dG[bP] <= BF_TOL).mean())
        rec = float((dist_to(bP)[bG] <= BF_TOL).mean())
        bf = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    else:
        bf = 1.0 if bP.any() == bG.any() else 0.0
    out = {"bf": bf}
    for w in BANDS:
        band = dG <= w
        u = ((P | G) & band).sum()
        out[f"tiou{w}"] = float(((P & G) & band).sum() / u) if u else 1.0
    wrong = P != G
    out["wrong_near"] = int((wrong & (dG <= NEAR)).sum())
    out["wrong_far"] = int((wrong & (dG > NEAR)).sum())
    return out


def levels(m):
    """The map on the 0..255 scale that metrics.f_curve thresholds (min-max normalized, truncated)."""
    return (metrics._minmax(m) * 255).astype(np.uint8)


def _scores(m, gt, dG):
    sc = metrics.image_scores(m, gt)
    sc["f_curve"] = sc["f_curve"].astype(np.float32)
    x = levels(m)
    sc["bnd"] = {}
    for t in THRESHOLDS:
        P = x >= t
        key = P.tobytes()
        prev = next((v for v in sc["bnd"].values() if v[0] == key), None)   # binary maps: one computation
        sc["bnd"][t] = (key, prev[1] if prev else boundary_scores(P, gt, dG))
    sc["bnd"] = {t: v[1] for t, v in sc["bnd"].items()}
    sc["cat"], _ = classify(binarize(m), gt)
    return sc


def _image(id_, sod_model, ds, prompt, transfer):
    inp = pipeline.load_inputs(sod_model, ds, prompt, id_, _image.scores)
    gt = inp["gt"]
    dG = dist_to(contour(gt))
    sod = inp["sod"]
    S = binarize(sod)
    sam = [m.astype(bool) for m in inp["sam"]]
    n = np.sum(sam, axis=0)
    ious = [np.logical_and(m, S).sum() / max(np.logical_or(m, S).sum(), 1) for m in sam]
    probs = inp["probs"] if inp["probs"] is not None else [m.astype(np.float64) * 255 for m in sam]
    sam_mean = np.mean([p.astype(np.float64) for p in probs], axis=0)
    s01 = sod.astype(np.float64) / (sod.max() + 1e-13)
    maps = {
        "sod": sod,
        "gc": data.read_png(data.run_dir(sod_model, ds, "gc_zoom") / f"{id_}.png"),
        "head": data.read_png(data.OUT_ROOT / f"{sod_model}_hz" / ds / "sod" / f"{id_}.png"),
        "sam_conf": sam[int(np.argmax(inp["scores"]))].astype(np.uint8) * 255,
        "rule_major": (S | (n >= 2)).astype(np.uint8) * 255,
        "rule_samiou": sam[int(np.argmax(ious))].astype(np.uint8) * 255,
        "rule_avg": np.round((s01 * 255 + sam_mean) / 2).astype(np.uint8),
    }
    if transfer:
        model = pipeline._v2_model(transfer)
        maps["transfer"] = fuse2(prepare(inp), *model["model"][:2], "binary", model["params"])
    for k, m in maps.items():
        if m.shape != gt.shape:
            raise ValueError(f"{ds}/{id_} {k}: {m.shape} vs GT {gt.shape}")
    return {"id": id_, "gt_empty": not gt.any(), "probs": inp["probs"] is not None,
            "maps": {k: _scores(m, gt, dG) for k, m in maps.items()}}


def _init(scores):
    _image.scores = scores


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        ids = data.list_ids(ds)[:args.limit]
        scores = pipeline.load_scores(args.sod_model, ds, args.prompt)
        f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt, transfer=args.transfer)
        with Pool(args.workers, initializer=_init, initargs=(scores,)) as pool:
            recs = list(tqdm(pool.imap(f, ids, chunksize=4), total=len(ids), desc=ds))
        name = f"{args.sod_model}_{ds}" + (f"_lim{args.limit}" if args.limit else "")
        with open(OUT / f"{name}.pkl", "wb") as fh:
            pickle.dump({"sod_model": args.sod_model, "ds": ds, "transfer": args.transfer, "records": recs}, fh)
        print(f"{ds}: {len(recs)} images, SAM probabilities for {sum(r['probs'] for r in recs)}", flush=True)


def report(args):
    suffix = f"_lim{args.limit}" if args.limit else ""
    out = {}
    for ds in args.datasets:
        recs = pickle.load(open(OUT / f"{args.sod_model}_{ds}{suffix}.pkl", "rb"))["records"]
        keys = [k for k in MAPS if k in recs[0]["maps"]]
        res = {}
        print(f"== {args.sod_model} {ds} (n={len(recs)})")
        print(f"  {'map':12s} {'F':>6s} {'MAE':>6s} {'E':>6s} {'S':>6s} | thr {'BF3':>6s} "
              + " ".join(f"tIoU{w:<3d}" for w in BANDS) + f" | wrong<= {NEAR}px")
        istar = {}
        for k in keys:
            sc = [r["maps"][k] for r in recs]
            m = metrics.aggregate(sc)
            istar[k] = int(np.argmax(np.mean([s["f_curve"] for s in sc], axis=0)))
            t = min(THRESHOLDS, key=lambda t: abs(t - (255 - istar[k])))
            m["thr_fmax"], m["thr_used"] = 255 - istar[k], t
            nonempty = [s["bnd"][t] for s, r in zip(sc, recs) if not r["gt_empty"]]
            for b in ["bf"] + [f"tiou{w}" for w in BANDS]:
                m[b] = float(np.mean([s[b] for s in nonempty]))
            near = sum(s["bnd"][t]["wrong_near"] for s in sc)
            far = sum(s["bnd"][t]["wrong_far"] for s in sc)
            m["wrong_near_share"] = near / max(near + far, 1)
            m["wrong_px"] = near + far
            res[k] = m
            print(f"  {k:12s} {m['Fmax']:.4f} {m['MAE']:.4f} {m['E']:.4f} {m['S']:.4f} | {t:3d} {m['bf']:.4f} "
                  + " ".join(f"{m[f'tiou{w}']:.4f}  " for w in BANDS)
                  + f" | {m['wrong_near_share']:.1%} of {m['wrong_px']} px")
        # loss 1 - F at each map's F^max threshold, images grouped by the SOD map's type (sums: n (1 - F^max))
        def loss(r, k):
            return 1.0 - float(r["maps"][k]["f_curve"][istar[k]])
        rs = recs
        tot = {k: sum(loss(r, k) for r in rs) for k in keys}
        by = {}
        print(f"  loss at the F^max threshold by the SOD map's error type (share of the SOD map's total loss; "
              f"change for each map)")
        for c in CATS:
            g = [r for r in rs if not r["gt_empty"] and r["maps"]["sod"]["cat"] == c]
            if not g:
                continue
            ls = {k: sum(loss(r, k) for r in g) for k in keys}
            by[c] = {"n": len(g), "loss": ls}
            print(f"   {c:13s} n={len(g):5d} share {ls['sod'] / tot['sod']:5.1%} | "
                  + " ".join(f"{k} {(ls[k] - ls['sod']) / max(ls['sod'], 1e-9):+.1%}" for k in keys if k != "sod"))
        res["_loss_by_sod_type"] = by
        out[ds] = res
    (OUT / f"report_{args.sod_model}{suffix}.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--transfer", default=None, help="tune2 result json of the other SOD model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else run(args)


if __name__ == "__main__":
    main()
