"""What do the images have in common where a SAM mask beats the SOD map (the oracle's headroom)?

For each image: per-image F (max over thresholds) of the SOD map and of each of SAM's three masks;
"headroom" images are those where the best SAM mask beats the SOD map by more than 0.02. Each
headroom image is put in one category by comparing the thresholded SOD mask S with the best SAM mask M:
  completion   S mostly inside M (|S & M| / |S| >= 0.9) and M larger (|M| > 1.15 |S|): SAM completes
               an object SOD found only in part
  trimming     M mostly inside S and smaller: SAM removes a false-positive part of SOD's mask
  boundary     IoU(S, M) >= 0.85: same object, SAM's boundary is better
  other_object IoU(S, M) < 0.5 and neither inside the other: SAM picked a different object
  mixed        everything else
Also counts the images where SOD fails (F < 0.7) and no SAM mask does better: errors that SAM,
prompted from the SOD map, cannot fix. Writes a contact sheet of examples per category.

  python scripts/headroom.py --sod-model samba --datasets DUTS-TE DUT-OMRON --workers 2
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, metrics

OUT = Path(__file__).resolve().parents[1] / "results" / "headroom"
CATEGORIES = ("completion", "trimming", "boundary", "other_object", "mixed")


def _image(id_, sod_model, ds):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sam_dir = data.run_dir(sod_model, ds, "sam_mask_soft")
    masks = [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
    f = [float(metrics.f_curve(c, gt).max()) for c in [sod] + [m.astype(np.uint8) * 255 for m in masks]]
    k = int(np.argmax(f[1:]))
    S = sod.astype(np.float64) / (sod.max() + 1e-13) >= 127 / 255
    M = masks[k]
    inter, s_area, m_area = np.logical_and(S, M).sum(), S.sum(), M.sum()
    union = np.logical_or(S, M).sum()
    iou = inter / union if union else 1.0
    s_in_m = inter / s_area if s_area else 1.0
    m_in_s = inter / m_area if m_area else 1.0
    gain = f[k + 1] - f[0]
    if gain <= 0.02:
        cat = None
    elif s_in_m >= 0.9 and m_area > 1.15 * s_area:
        cat = "completion"
    elif m_in_s >= 0.9 and m_area < s_area / 1.15:
        cat = "trimming"
    elif iou >= 0.85:
        cat = "boundary"
    elif iou < 0.5 and s_in_m < 0.9 and m_in_s < 0.9:
        cat = "other_object"
    else:
        cat = "mixed"
    return {"id": id_, "ds": ds, "f_sod": f[0], "f_sam": f[k + 1], "k": k + 1, "gain": gain, "cat": cat,
            "iou": float(iou), "s_in_m": float(s_in_m), "m_in_s": float(m_in_s),
            "sod_area": float(S.mean()), "sam_area": float(M.mean()), "gt_area": float(gt.mean()),
            "sod_fails": f[0] < 0.7, "sam_fixes": max(f[1:]) >= 0.7}


def _tile(ds, id_, sod_model, k, size=160):
    img = data.load_image(ds, id_)
    gt = data.load_gt(ds, id_).astype(np.uint8) * 255
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sam = data.read_png(data.run_dir(sod_model, ds, "sam_mask_soft") / f"{id_}_{k}.png")
    row = []
    for a in (img, gt, sod, sam):
        a = a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
        h, w = a.shape[:2]
        s = size / max(h, w)
        a = cv2.resize(a, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        pad = np.full((size, size, 3), 255, np.uint8)
        pad[:a.shape[0], :a.shape[1]] = a
        row.append(pad)
    return np.concatenate(row, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--per-category", type=int, default=6, help="contact-sheet rows per category")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            f = partial(_image, sod_model=args.sod_model, ds=ds)
            recs += [r for r in pool.imap(f, data.list_ids(ds), chunksize=16) if r is not None]
    summary = {}
    for ds in args.datasets + ["all"]:
        rs = [r for r in recs if ds == "all" or r["ds"] == ds]
        hr = [r for r in rs if r["cat"]]
        cats = Counter(r["cat"] for r in hr)
        fails = [r for r in rs if r["sod_fails"]]
        summary[ds] = {
            "n": len(rs), "headroom": len(hr),
            # share of the total oracle gain (sum of per-image F gains) carried by each category
            "categories": {c: {"n": cats[c], "share_of_gain": float(sum(r["gain"] for r in hr if r["cat"] == c)
                                                                 / max(sum(r["gain"] for r in hr), 1e-9)),
                               "mean_gain": float(np.mean([r["gain"] for r in hr if r["cat"] == c]))
                               if cats[c] else None} for c in CATEGORIES},
            "sod_fails": len(fails), "sod_fails_sam_fixes": sum(r["sam_fixes"] for r in fails),
        }
        s = summary[ds]
        print(f"{args.sod_model} {ds:10s} n={s['n']} headroom={s['headroom']} ({s['headroom'] / s['n']:.1%}) | "
              + " ".join(f"{c}={s['categories'][c]['n']} ({s['categories'][c]['share_of_gain']:.0%} of gain)"
                         for c in CATEGORIES)
              + f" | SOD fails on {s['sod_fails']}, a SAM mask fixes {s['sod_fails_sam_fixes']}", flush=True)
    (OUT / f"{args.sod_model}.json").write_text(json.dumps({"summary": summary, "images": recs}, indent=1))
    # contact sheet: per category the largest-gain examples; columns image | GT | SOD | best SAM mask
    by_cat = defaultdict(list)
    for r in sorted((r for r in recs if r["cat"]), key=lambda r: -r["gain"]):
        if len(by_cat[r["cat"]]) < args.per_category:
            by_cat[r["cat"]].append(r)
    for c in CATEGORIES:
        if by_cat[c]:
            sheet = np.concatenate([_tile(r["ds"], r["id"], args.sod_model, r["k"]) for r in by_cat[c]], 0)
            cv2.imwrite(str(OUT / f"{args.sod_model}_{c}.jpg"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{c}: " + ", ".join(f"{r['ds']}/{r['id']} +{r['gain']:.2f}" for r in by_cat[c]), flush=True)


if __name__ == "__main__":
    main()
