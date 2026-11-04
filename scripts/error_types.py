"""Where does an SOD model's remaining error come from? A decomposition that compares the SOD mask
with the ground truth directly (no SAM or other proposals, unlike headroom.py).

Per image: loss = 1 - per-image F (max over thresholds of the F curve of the soft map), and the
thresholded SOD mask S (max-normalized map >= 127/255) against the GT mask G:
  boundary        IoU(S, G) >= 0.85: right object, imperfect outline
  part_of_gt      S mostly inside G (|S & G| / |S| >= 0.9) and |G| >= 2 |S|: SOD chose a part, GT the whole
  whole_of_gt     G mostly inside S (|S & G| / |G| >= 0.9) and |S| >= 2 |G|: SOD chose the whole, GT a part
  under_extent    S mostly inside G, 1.15 |S| < |G| < 2 |S|: SOD's object is too small (missing pieces)
  over_extent     G mostly inside S, 1.15 |G| < |S| < 2 |G|: SOD's object is too large (extra pieces)
  other_object    IoU < 0.5 and neither mostly inside the other (wrong or missing object; includes S empty)
  mixed           everything else
Reports per dataset the share of the total loss in each category, and contact sheets of the largest-
loss examples (image | GT | SOD) per category.

  python scripts/error_types.py --sod-model samba
  python scripts/error_types.py --maps-root <dir with <dataset>/<id>.png> --name birefnet
"""
import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, metrics

OUT = Path(__file__).resolve().parents[1] / "results" / "error_types"
CATS = ("boundary", "part_of_gt", "whole_of_gt", "under_extent", "over_extent", "other_object", "mixed")


def classify(S, G):
    inter, s_area, g_area = np.logical_and(S, G).sum(), S.sum(), G.sum()
    union = np.logical_or(S, G).sum()
    iou = inter / union if union else 1.0
    s_in_g = inter / s_area if s_area else 0.0
    g_in_s = inter / g_area if g_area else 0.0
    if iou >= 0.85:
        return "boundary", iou
    if s_area and s_in_g >= 0.9 and g_area >= 2 * s_area:
        return "part_of_gt", iou
    if g_in_s >= 0.9 and s_area >= 2 * g_area:
        return "whole_of_gt", iou
    if s_area and s_in_g >= 0.9 and g_area > 1.15 * s_area:
        return "under_extent", iou
    if g_in_s >= 0.9 and s_area > 1.15 * g_area:
        return "over_extent", iou
    if iou < 0.5 and s_in_g < 0.9 and g_in_s < 0.9:
        return "other_object", iou
    return "mixed", iou


def map_path(args, ds, id_):
    if args.maps_root:
        return Path(args.maps_root) / ds / f"{id_}.png"
    return data.run_dir(args.sod_model, ds, "sod") / f"{id_}.png"


def _image(id_, ds, args):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    p = map_path(args, ds, id_)
    if not p.exists():
        return {"id": id_, "ds": ds, "missing": True}
    sod = data.read_png(p)
    if sod.shape != gt.shape:
        sod = cv2.resize(sod, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    loss = 1.0 - float(metrics.f_curve(sod, gt).max())
    S = sod.astype(np.float64) / (sod.max() + 1e-13) >= 127 / 255 if sod.max() > 0 else np.zeros_like(gt)
    cat, iou = classify(S, gt)
    return {"id": id_, "ds": ds, "loss": loss, "cat": cat, "iou": float(iou), "missing": False}


def _tile(args, ds, id_, size=160):
    tiles = []
    for a in (data.load_image(ds, id_), data.load_gt(ds, id_).astype(np.uint8) * 255,
              data.read_png(map_path(args, ds, id_))):
        a = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
        h, w = a.shape[:2]
        s = size / max(h, w)
        a = cv2.resize(a, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        pad = np.full((size, size, 3), 255, np.uint8)
        pad[:a.shape[0], :a.shape[1]] = a
        tiles.append(pad)
    return np.concatenate(tiles, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", help="maps under output/<sod_model>/<dataset>/sod")
    ap.add_argument("--maps-root", help="or: maps under <maps-root>/<dataset>/<id>.png")
    ap.add_argument("--name", help="name for outputs (default: sod-model)")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-category", type=int, default=8)
    args = ap.parse_args()
    name = args.name or args.sod_model
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            recs += [r for r in pool.imap(partial(_image, ds=ds, args=args), data.list_ids(ds), chunksize=16) if r]
    missing = [r for r in recs if r["missing"]]
    recs = [r for r in recs if not r["missing"]]
    summary = {}
    for ds in args.datasets + ["all"]:
        rs = [r for r in recs if ds == "all" or r["ds"] == ds]
        if not rs:
            continue
        tot = sum(r["loss"] for r in rs)
        summary[ds] = {"n": len(rs), "mean_loss": tot / len(rs),
                       "share": {c: sum(r["loss"] for r in rs if r["cat"] == c) / max(tot, 1e-9) for c in CATS},
                       "count": {c: sum(r["cat"] == c for r in rs) for c in CATS}}
        s = summary[ds]
        print(f"{name} {ds:10s} n={s['n']} mean loss {s['mean_loss']:.3f} | "
              + " ".join(f"{c} {s['share'][c]:.0%} ({s['count'][c]})" for c in CATS), flush=True)
    if missing:
        print(f"{name}: {len(missing)} maps missing, e.g. {missing[0]['ds']}/{missing[0]['id']}", flush=True)
    (OUT / f"{name}.json").write_text(json.dumps({"summary": summary, "images": recs, "missing": len(missing)},
                                                 indent=1))
    by_cat = defaultdict(list)
    for r in sorted(recs, key=lambda r: -r["loss"]):
        if len(by_cat[r["cat"]]) < args.per_category:
            by_cat[r["cat"]].append(r)
    for c in CATS:
        if by_cat[c]:
            sheet = np.concatenate([_tile(args, r["ds"], r["id"]) for r in by_cat[c]], 0)
            cv2.imwrite(str(OUT / f"{name}_{c}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
