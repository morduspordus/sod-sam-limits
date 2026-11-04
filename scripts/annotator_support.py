"""Annotation-free evidence on whether SOD "errors" are answers that other people also give, from
existing multi-annotator data (no new annotations):

  SOD (Movahedi & Elder 2010; "MSOD" here): 300 BSDS images, 7 subjects, several sessions each
      (one per BSDS segmentation); per session the subject picks salient objects as unions of
      segments and ranks them (IMP = 1 most salient). Gives (a) how often people themselves split on
      part vs whole, and (b) per model error, the share of sessions whose answer is closer to the
      model's mask than to the benchmark GT.
  DUT-OMRON: rectangles from 5 participants and eye fixations of 5 viewers. Per model error, the
      participants whose rectangle is closer to the model mask's box than to the GT's box, and how
      densely the model's extra region (S \\ G) was fixated compared with the background.

Error categories (boundary is the control) come from error_types.classify on the thresholded map.

  python scripts/annotator_support.py --models samba selfreformer birefnet mdsam vscode vstpp
"""
import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_fill_holes

from error_types import CATS, classify
from sodsam import data, metrics

DL = data.DATA_ROOT / "downloads"
SOD_RAW = DL / "SOD_raw" / "x"
BOX = DL / "DUT-OMRON_extra" / "box" / "DUT-OMRON-bounding-box" / "mat"
FIX = DL / "DUT-OMRON_extra" / "fix" / "DUT-OMRON-eye-fixations" / "mat"
MAPS = {"birefnet": "maps_BiRefNet", "mdsam": "maps_MDSAM", "vscode": "maps_VSCode", "vstpp": "maps_VSTpp"}
DS_ALIAS = {"MSOD": "SOD"}            # dataset folder name in the downloaded map sets
OUT = Path(__file__).resolve().parents[1] / "results" / "error_types"


def map_path(model, ds, id_):
    if model in MAPS:
        return DL / MAPS[model] / DS_ALIAS.get(ds, ds) / f"{id_}.png"
    return data.run_dir(model, ds, "sod") / f"{id_}.png"


def model_mask(model, ds, id_, g):
    """Soft map resized to the GT, its per-image loss, and the thresholded mask (as error_types)."""
    p = map_path(model, ds, id_)
    if not p.exists():
        return None
    m = data.read_png(p)
    if m.shape != g.shape:
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    loss = 1.0 - float(metrics.f_curve(m, g).max())
    S = m.astype(np.float64) / (m.max() + 1e-13) >= 127 / 255 if m.max() > 0 else np.zeros_like(g)
    return loss, S


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def nested_part_whole(a, b):
    """a and b nested (smaller >= 90% inside larger) with the larger >= 2x the smaller."""
    sa, sb = a.sum(), b.sum()
    if not sa or not sb:
        return False
    small, big = (a, b) if sa <= sb else (b, a)
    return np.logical_and(small, big).sum() >= 0.9 * small.sum() and big.sum() >= 2 * small.sum()


# ---------------------------------------------------------------- SOD (per-subject sessions)

def _objects(o):
    return list(o) if isinstance(o, np.ndarray) else [o]


def sod_sessions(id_, shape):
    """Per session: subject, top-object mask (IMP 1) and union of all chosen objects."""
    s = sio.loadmat(SOD_RAW / f"SO{id_}.mat", squeeze_me=True, struct_as_record=False)["SES"]
    out = []
    for e in _objects(s):
        top, union = np.zeros(shape, bool), np.zeros(shape, bool)
        for o in _objects(e.obj):
            bnds = o.BND if isinstance(o.BND, np.ndarray) and o.BND.dtype == object else [o.BND]
            m = np.zeros(shape, bool)
            for b in bnds:
                b = np.atleast_2d(np.asarray(b, int)) - 1
                ok = (b[:, 0] >= 0) & (b[:, 0] < shape[0]) & (b[:, 1] >= 0) & (b[:, 1] < shape[1])
                edge = np.zeros(shape, bool)
                edge[b[ok, 0], b[ok, 1]] = True
                m |= binary_fill_holes(edge)
            union |= m
            if int(o.IMP) == 1:
                top |= m
        out.append({"sj": int(e.sj), "top": top, "union": union})
    return out


def _sod_image(id_, models):
    g = data.load_gt("MSOD", id_)
    ses = sod_sessions(id_, g.shape)
    rec = {"id": id_, "gt_iou_top": [float(iou(x["top"], g)) for x in ses],
           "gt_iou_union": [float(iou(x["union"], g)) for x in ses]}
    for key in ("top", "union"):
        pairs = [(a, b) for a, b in combinations(ses, 2) if a["sj"] != b["sj"]]
        rec[f"pw_pairs_{key}"] = sum(nested_part_whole(a[key], b[key]) for a, b in pairs)
        rec[f"n_pairs"] = len(pairs)
    rec["models"] = {}
    for model in models:
        r = model_mask(model, "MSOD", id_, g)
        if r is None:
            continue
        loss, S = r
        cat, _ = classify(S, g)
        rec["models"][model] = {"loss": loss, "cat": cat, **{
            f"closer_{key}": float(np.mean([iou(x[key], S) > iou(x[key], g) for x in ses])) for key in ("top", "union")},
            **{f"best_{key}": float(max(iou(x[key], S) for x in ses)) for key in ("top", "union")}}
    return rec


# ---------------------------------------------------------------- DUT-OMRON (boxes, fixations)

def box_mask(r, shape):
    """Participant rectangle [row0, row1, col0, col1] (1-based, verified against the GT boxes)."""
    m = np.zeros(shape, bool)
    m[max(r[0] - 1, 0):r[1], max(r[2] - 1, 0):r[3]] = True
    return m


def bbox_mask(a):
    m = np.zeros_like(a)
    if a.any():
        rows, cols = np.where(a)
        m[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    return m


def _omron_image(id_, models):
    g = data.load_gt("DUT-OMRON", id_)
    if not g.any():
        return None
    recs = sio.loadmat(BOX / f"{id_}.mat")["rec"].astype(int)          # stored as uint16
    boxes = [box_mask(r[:4], g.shape) for r in recs]
    fx = sio.loadmat(FIX / f"{id_}.mat")["s"].astype(int)
    fx = fx[fx[:, 2] > 0]
    xs, ys = np.clip(fx[:, 0] - 1, 0, g.shape[1] - 1), np.clip(fx[:, 1] - 1, 0, g.shape[0] - 1)
    hit = np.zeros(g.shape, np.int32)
    np.add.at(hit, (ys, xs), 1)
    bG = bbox_mask(g)
    rec = {"id": id_, "gt_box_iou": [float(iou(b, bG)) for b in boxes], "models": {}}
    for model in models:
        r = model_mask(model, "DUT-OMRON", id_, g)
        if r is None:
            continue
        loss, S = r
        cat, _ = classify(S, g)
        bS = bbox_mask(S)
        extra, bg = S & ~g, ~(S | g)
        dens = lambda m: hit[m].sum() / max(m.sum(), 1)            # fixations per pixel
        rec["models"][model] = {
            "loss": loss, "cat": cat, "box_iou_sg": float(iou(bS, bG)),
            "closer": int(sum(iou(b, bS) > iou(b, bG) for b in boxes)),
            "extra_frac": float(extra.mean()),
            "fix_extra": float(dens(extra)), "fix_bg": float(dens(bg)), "fix_gt": float(dens(g)),
            "n_fix": int(hit.sum())}
    return rec


# ---------------------------------------------------------------- summaries

def summarize_sod(recs, models):
    out = {"n": len(recs),
           "gt_iou_top_mean": float(np.mean([np.mean(r["gt_iou_top"]) for r in recs])),
           "gt_iou_union_mean": float(np.mean([np.mean(r["gt_iou_union"]) for r in recs]))}
    for key in ("top", "union"):
        out[f"human_part_whole_{key}"] = {
            "images_with_any_pair": float(np.mean([r[f"pw_pairs_{key}"] > 0 for r in recs])),
            "share_of_cross_subject_pairs": sum(r[f"pw_pairs_{key}"] for r in recs) / sum(r["n_pairs"] for r in recs)}
    print(f"SOD: {len(recs)} images; mean IoU(session answer, GT): top object {out['gt_iou_top_mean']:.2f}, "
          f"all chosen objects {out['gt_iou_union_mean']:.2f}")
    for key in ("top", "union"):
        h = out[f"human_part_whole_{key}"]
        print(f"   people split on part vs whole ({key}): {h['images_with_any_pair']:.0%} of images have a pair of "
              f"subjects with nested part/whole answers; {h['share_of_cross_subject_pairs']:.1%} of all pairs")
    for model in models:
        rs = [r["models"][model] for r in recs if model in r["models"]]
        if not rs:
            continue
        tot = sum(x["loss"] for x in rs)
        res = {"n": len(rs), "by_category": {}}
        for key in ("top", "union"):
            res[f"loss_share_majority_closer_{key}"] = sum(x["loss"] for x in rs if x[f"closer_{key}"] > 0.5) / tot
            res[f"loss_share_some_closer_{key}"] = sum(x["loss"] for x in rs if x[f"closer_{key}"] >= 0.25) / tot
        for c in CATS:
            cs = [x for x in rs if x["cat"] == c]
            if cs:
                res["by_category"][c] = {"n": len(cs), "loss_share": sum(x["loss"] for x in cs) / tot, **{
                    f"sessions_closer_{key}": float(np.mean([x[f"closer_{key}"] for x in cs])) for key in ("top", "union")},
                    **{f"some_session_iou80_{key}": float(np.mean([x[f"best_{key}"] >= 0.8 for x in cs])) for key in ("top", "union")}}
        out[model] = res
        print(f" {model}: loss on images where >=25% of sessions are closer to the model than to the GT: "
              f"{res['loss_share_some_closer_union']:.0%} (union) / {res['loss_share_some_closer_top']:.0%} (top); "
              f"majority closer: {res['loss_share_majority_closer_union']:.0%} / {res['loss_share_majority_closer_top']:.0%}")
        for c, v in res["by_category"].items():
            print(f"   {c:13s} n={v['n']:3d} loss {v['loss_share']:4.0%} | sessions closer to model: union "
                  f"{v['sessions_closer_union']:.0%} top {v['sessions_closer_top']:.0%} | some session matches model "
                  f"(IoU>=.8): union {v['some_session_iou80_union']:.0%} top {v['some_session_iou80_top']:.0%}")
    return out


def summarize_omron(recs, models):
    out = {"n": len(recs), "gt_box_iou_mean": float(np.mean([np.mean(r["gt_box_iou"]) for r in recs]))}
    print(f"DUT-OMRON: {len(recs)} images; mean IoU(participant box, GT box) {out['gt_box_iou_mean']:.2f}")
    for model in models:
        rs = [r["models"][model] for r in recs if model in r["models"]]
        if not rs:
            continue
        tot = sum(x["loss"] for x in rs)
        res = {"n": len(rs), "by_category": {},
               "loss_share_box_differs_and_2plus_closer":
                   sum(x["loss"] for x in rs if x["box_iou_sg"] < 0.7 and x["closer"] >= 2) / tot}
        for c in CATS:
            cs = [x for x in rs if x["cat"] == c]
            if not cs:
                continue
            ex = [x for x in cs if x["extra_frac"] >= 0.005]
            fixed = [x for x in ex if x["fix_bg"] > 0]
            res["by_category"][c] = {
                "n": len(cs), "loss_share": sum(x["loss"] for x in cs) / tot,
                "box_differs": len([x for x in cs if x["box_iou_sg"] < 0.7]),
                ">=1_closer": float(np.mean([x["closer"] >= 1 for x in cs if x["box_iou_sg"] < 0.7] or [np.nan])),
                ">=2_closer": float(np.mean([x["closer"] >= 2 for x in cs if x["box_iou_sg"] < 0.7] or [np.nan])),
                # fixations per pixel, pooled over images, extra region vs background vs GT
                "fix_density_extra": float(np.sum([x["fix_extra"] * x["extra_frac"] for x in ex]) /
                                           max(np.sum([x["extra_frac"] for x in ex]), 1e-9)) if ex else None,
                "fix_extra_over_bg_median": float(np.median([x["fix_extra"] / x["fix_bg"] for x in fixed])) if fixed else None,
                "extra_fixated_share": float(np.mean([x["fix_extra"] > 0 for x in ex])) if ex else None}
        out[model] = res
        print(f" {model}: loss on images whose box differs from the GT box (IoU<.7) and >=2 of 5 participants' "
              f"boxes are closer to the model: {res['loss_share_box_differs_and_2plus_closer']:.0%}")
        for c, v in res["by_category"].items():
            print(f"   {c:13s} n={v['n']:4d} loss {v['loss_share']:4.0%} | box differs on {v['box_differs']:4d}: "
                  f">=1 closer {v['>=1_closer']:.0%}, >=2 closer {v['>=2_closer']:.0%} | extra region fixated on "
                  f"{v['extra_fixated_share'] if v['extra_fixated_share'] is None else format(v['extra_fixated_share'], '.0%')}"
                  f", density extra/bg median {v['fix_extra_over_bg_median']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["samba", "selfreformer", "birefnet", "mdsam", "vscode", "vstpp"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", choices=["sod", "omron"])
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        if args.only != "omron":
            ids = [i for i in data.list_ids("MSOD") if (SOD_RAW / f"SO{i}.mat").exists()]
            recs = list(pool.imap(partial(_sod_image, models=args.models), ids, chunksize=4))
            out["SOD"] = summarize_sod(recs, args.models)
        if args.only != "sod":
            ids = [i for i in data.list_ids("DUT-OMRON") if (BOX / f"{i}.mat").exists()]
            recs = [r for r in pool.imap(partial(_omron_image, models=args.models), ids, chunksize=16) if r]
            out["DUT-OMRON"] = summarize_omron(recs, args.models)
    (OUT / f"annotator_support{'_' + args.only if args.only else ''}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
