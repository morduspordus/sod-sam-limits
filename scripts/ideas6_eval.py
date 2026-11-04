"""Per-image records for the analyses of scripts/ideas6_report.py, one pass over the images.

For every image of every dataset, with the paper model of --base (a tune2.py result, any fusion2 model):
  maps     image_scores (f_curve as float32, mae, s, e) of
             sod       the SOD map
             cut       the binary cut of --base
             soft      the fusion2 "soft" output of the same cut
             blend<a>  (1 - a/100) * SOD + (a/100) * cut, a in BLENDS
             cv        with --cv: the cut with the parameters tuned on the other half of the image's own
                       dataset (tune2.py --datasets <ds> --tune-half A | B: an image of half A uses the
                       result tuned on B and vice versa; both result files must exist)
             oracle    the per-image best (max of the F curve) of sod, cut and the three SAM masks
  feats    per-image statistics for the gate (FEATURES)
  thin     thin-structure diagnostic per radius r in THIN_R: thin fraction of the SOD mask (|mask minus
           its opening| / |mask|) and, inside the region T (fusion2.thin_mask) and outside it, the pixels
           the cut fixes (SOD mask wrong, cut right) and breaks (SOD mask right, cut wrong)

  python scripts/ideas6_eval.py --sod-model samba --datasets DUTS-TE --base results/tune2/<paper>.json --cv
Writes results/ideas6/per_image_<sod>_<ds>[_cv].pkl (a list of dicts; "_cv" with --cv, so a run with and
one without never write the same file); --limit N for smoke tests (file name carries "_lim<N>").
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

from sodsam import data, metrics, pipeline
from sodsam.fusion2 import fuse2, prepare, thin_mask

OUT = Path(__file__).resolve().parents[1] / "results" / "ideas6"
TUNE2 = Path(__file__).resolve().parents[1] / "results" / "tune2"
BLENDS = (25, 50, 75)
THIN_R = (1, 2, 3, 4)
FEATURES = ("iou_maj", "iou_cut", "changed", "sod_unc", "sod_mid", "score_mean", "score_min", "sam_agree",
            "area", "comp_sod", "comp_cut", "thin2")


def fold_paths(sod_model, ds, base):
    """Result files of tune2.py tuned on half A and on half B of ds, same model and prompt as base."""
    res = json.loads(Path(base).read_text())
    unary, pairwise, output = res["model"]
    stem = f"{sod_model}_{ds}_{res['prompt']}_{unary}_{pairwise}_{output}"
    return {h: TUNE2 / f"{stem}_{h}.json" for h in "AB"}


def _scores(pred, gt):
    sc = metrics.image_scores(pred, gt)
    sc["f_curve"] = sc["f_curve"].astype(np.float32)
    return sc


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def _components(mask):
    return int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)


def features(x, lab, scores):
    s, sm, m = x["s"], x["sod_mask"], x["sel"][3]
    k = len(m)
    agree = [_iou(m[i], m[j]) for i in range(k) for j in range(i + 1, k)]
    thin = sm & ~cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    return {"iou_maj": float(x["iou"]), "iou_cut": _iou(sm, lab),
            "changed": float(np.logical_xor(sm, lab).sum() / max(sm.sum(), 1)),
            "sod_unc": float((4 * s * (1 - s)).mean()),
            "sod_mid": float(((s > 0.1) & (s < 0.9)).sum() / max(sm.sum(), 1)),
            "score_mean": float(np.mean(scores)), "score_min": float(np.min(scores)),
            "sam_agree": float(np.mean(agree)), "area": float(sm.mean()),
            "comp_sod": _components(sm), "comp_cut": _components(lab),
            "thin2": float(thin.sum() / max(sm.sum(), 1))}


def thin_diag(x, lab, gt):
    sm = x["sod_mask"]
    fix, brk = (lab == gt) & (sm != gt), (lab != gt) & (sm == gt)
    out = {}
    for r in THIN_R:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        thin = sm & ~cv2.morphologyEx(sm.astype(np.uint8), cv2.MORPH_OPEN, disk).astype(bool)
        T = thin_mask(sm, r)
        out[r] = {"frac": float(thin.sum() / max(sm.sum(), 1)), "T_px": int(T.sum()),
                  "fix_in": int((fix & T).sum()), "break_in": int((brk & T).sum()),
                  "fix_out": int((fix & ~T).sum()), "break_out": int((brk & ~T).sum())}
    return out


def _image(id_, sod_model, ds, prompt, base, folds):
    inp = pipeline.load_inputs(sod_model, ds, prompt, id_, _image.scores)
    x = prepare(inp)
    gt = inp["gt"]
    model = pipeline._v2_model(base)
    lab = fuse2(x, *model["model"][:2], "binary", model["params"]) > 0
    cut = lab.astype(np.uint8) * 255
    v = np.round(x["s"] * 255)
    soft = np.where(lab, np.maximum(v, 128), np.minimum(v, 127)).astype(np.uint8)
    sod = inp["sod"]
    maps = {"sod": _scores(sod, gt), "cut": _scores(cut, gt), "soft": _scores(soft, gt)}
    for a in BLENDS:
        bl = np.round((1 - a / 100) * sod.astype(np.float64) + (a / 100) * cut).astype(np.uint8)
        maps[f"blend{a}"] = _scores(bl, gt)
    if folds:
        # an image of half A is scored with the result tuned on half B and vice versa
        half = "A" if id_ in _image.half_a else "B"
        fm = pipeline._v2_model(str(folds["B" if half == "A" else "A"]))
        maps["cv"] = _scores(fuse2(x, *fm["model"][:2], "binary", fm["params"]), gt)
    cands = [maps["sod"], maps["cut"]] + [_scores(m.astype(np.uint8) * 255, gt) for m in x["sel"][3]]
    maps["oracle"] = max(cands, key=lambda c: float(c["f_curve"].max()))
    return {"id": id_, "ds": ds, "maps": maps, "feats": features(x, lab, inp["scores"]),
            "thin": thin_diag(x, lab, gt), "gt_empty": not gt.any()}


def _init(scores, half_a):
    _image.scores = scores
    _image.half_a = half_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="tune2.py result json of the paper model")
    ap.add_argument("--cv", action="store_true", help="also the cut tuned on the other half of the dataset")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        folds = None
        if args.cv:
            folds = fold_paths(args.sod_model, ds, args.base)
            missing = [str(p) for p in folds.values() if not p.exists()]
            if missing:
                sys.exit(f"--cv: missing {missing}")
            folds = {h: str(p) for h, p in folds.items()}
        ids = data.list_ids(ds)[:args.limit]
        # the halves of tune2.py (even / odd positions of all ids, before any --limit)
        half_a = set(data.list_ids(ds)[0::2])
        scores = pipeline.load_scores(args.sod_model, ds, args.prompt)
        f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt, base=args.base, folds=folds)
        with Pool(args.workers, initializer=_init, initargs=(scores, half_a)) as pool:
            recs = list(tqdm(pool.imap(f, ids, chunksize=4), total=len(ids), desc=ds))
        name = f"per_image_{args.sod_model}_{ds}" + ("_cv" if args.cv else "") + (f"_lim{args.limit}" if args.limit else "")
        with open(OUT / f"{name}.pkl", "wb") as fh:
            pickle.dump({"sod_model": args.sod_model, "ds": ds, "base": args.base, "folds": folds,
                         "records": recs}, fh)
        for key in recs[0]["maps"]:
            print(metrics.format_row(f"{ds} {key}", metrics.aggregate([r["maps"][key] for r in recs])), flush=True)


if __name__ == "__main__":
    main()
