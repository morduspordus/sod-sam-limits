"""Is the room of the pixel oracle a choice of granularity? (second review; extends pixel_oracle.py)

pixel_oracle.py's "rule per image" oracle picks, per image and with the ground truth, the best of the 2^16
maps from the 4-bit signature (binarized SOD map, SAM masks 1-3) to a label. Here, on one dataset:
  - choose5: a restricted per-image oracle that picks, with the ground truth, the best (per-image F) of five
    interpretable candidates: the binarized SOD map, SAM mask 1, 2, 3, and the dataset rule of pixel_oracle.py
    ("SOD or >= 2 SAM masks" on DUTS-TE); compared with the rule-per-image oracle and the pixel oracle
  - for every image whose rule-per-image map differs from the binarized SOD map: which of the five candidates
    is closest to it (IoU), and the SOD error type (error_types.py thresholds) of the image, with the share of
    the rule-per-image oracle's gain in summed per-image F per type
  - nesting of SAM's masks: the share of images whose three masks, sorted by area, are nested (each at least
    90% inside the next larger one), and the median area ratios
Output: results/granularity_<sod>_<prompt>_<dataset>.json

  python -I scripts/granularity.py --sod-model samba --workers 8
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
CANDS = ("SOD", "SAM 1", "SAM 2", "SAM 3", "rule")


def _signature(sod, sams):
    sig = sod.astype(np.uint8)
    for k, m in enumerate(sams):
        sig |= m.astype(np.uint8) << (k + 1)
    return sig


def _inputs(id_, sod_model, ds, prompt, thr):
    gt = data.load_gt(ds, id_)
    soft = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sod = (metrics._minmax(soft) * 255).astype(np.uint8) >= thr
    sam_dir = data.run_dir(sod_model, ds, f"sam_{prompt}")
    sams = [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
    return gt, soft, sod, sams


def _curve(id_, sod_model, ds):
    return metrics.f_curve(data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png"), data.load_gt(ds, id_))


def _counts(id_, sod_model, ds, prompt, thr):
    gt, _, sod, sams = _inputs(id_, sod_model, ds, prompt, thr)
    sig = _signature(sod, sams)
    return np.stack([np.bincount(sig[~gt], minlength=16), np.bincount(sig[gt], minlength=16)], 1)


def _type(S, G):
    """error_types.classify with the paper's thresholds, grouped as in the paper's table."""
    inter, s, g = np.logical_and(S, G).sum(), S.sum(), G.sum()
    union = np.logical_or(S, G).sum()
    iou = inter / union if union else 1.0
    s_in_g = inter / s if s else 0.0
    g_in_s = inter / g if g else 0.0
    if iou >= 0.85:
        return "boundary"
    if (s and s_in_g >= 0.9 and g >= 2 * s) or (g_in_s >= 0.9 and s >= 2 * g):
        return "part/whole"
    if (s and s_in_g >= 0.9 and g > 1.15 * s) or (g_in_s >= 0.9 and s > 1.15 * g):
        return "extent"
    if iou < 0.5 and s_in_g < 0.9 and g_in_s < 0.9:
        return "other object"
    return "mixed"


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def _f(m, gt):
    return float(metrics.f_curve(m.astype(np.uint8) * 255, gt).max())


def _image(id_, sod_model, ds, prompt, thr, rule):
    gt, soft, sod, sams = _inputs(id_, sod_model, ds, prompt, thr)
    sig = _signature(sod, sams)
    fg = np.bincount(sig.ravel(), weights=gt.ravel(), minlength=16)
    tot = np.bincount(sig.ravel(), minlength=16)
    per_image = (2 * fg > tot)[sig]                          # rule per image (pixel_oracle.py)
    cands = [sod] + sams + [rule[sig]]
    fs = [_f(c, gt) for c in cands]
    best = int(np.argmax(fs))
    out = {"choose5": cands[best], "rule_image": per_image, "sod": sod}
    rec = {k: {"scores": metrics.image_scores(m.astype(np.uint8) * 255, gt), "f": _f(m, gt)} for k, m in out.items()}
    S = soft.astype(np.float64) / (soft.max() + 1e-13) >= 127 / 255 if soft.max() > 0 else np.zeros_like(gt)
    areas = sorted([m.sum() for m in sams])
    by_area = sorted(sams, key=lambda m: m.sum())
    nested = all(by_area[i].sum() == 0 or np.logical_and(by_area[i], by_area[i + 1]).sum() >= 0.9 * by_area[i].sum()
                 for i in range(2))
    return {"rec": rec, "best5": CANDS[best], "differs": bool((per_image != sod).any()),
            "closest": CANDS[int(np.argmax([_iou(per_image, c) for c in cands]))],
            "type": _type(S, gt) if gt.any() else "empty", "nested": bool(nested),
            "ratios": [float(areas[1] / max(areas[0], 1)), float(areas[2] / max(areas[1], 1))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--dataset", default="DUTS-TE")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ds = args.dataset
    ids = data.list_ids(ds)
    with Pool(args.workers) as pool:
        mean_curve = np.mean(pool.map(partial(_curve, sod_model=args.sod_model, ds=ds), ids, chunksize=16), axis=0)
        thr = 255 - int(np.argmax(mean_curve))
        kw = dict(sod_model=args.sod_model, ds=ds, prompt=args.prompt, thr=thr)
        counts = np.sum(pool.map(partial(_counts, **kw), ids, chunksize=16), axis=0)
        rule = counts[:, 1] > counts[:, 0]
        per = pool.map(partial(_image, rule=rule, **kw), ids, chunksize=16)
    res = {"threshold": thr, "n": len(per), "rule_dataset": rule.astype(int).tolist()}
    for k in ("sod", "choose5", "rule_image"):
        res[k] = metrics.aggregate([r["rec"][k]["scores"] for r in per])["Fmax"]
    gain_ri = res["rule_image"] - res["sod"]
    res["choose5_share_of_rule_image_gain"] = (res["choose5"] - res["sod"]) / gain_ri if gain_ri else None
    res["best5"] = dict(Counter(r["best5"] for r in per))
    diff = [r for r in per if r["differs"]]
    res["closest_to_rule_image"] = dict(Counter(r["closest"] for r in diff))
    gain_by_type = defaultdict(float)
    for r in per:
        gain_by_type[r["type"]] += r["rec"]["rule_image"]["f"] - r["rec"]["sod"]["f"]
    tot = sum(gain_by_type.values())
    res["rule_image_gain_share_by_sod_type"] = {k: v / tot for k, v in gain_by_type.items()}
    res["nested_share"] = float(np.mean([r["nested"] for r in per]))
    res["median_area_ratio"] = [float(np.median([r["ratios"][i] for r in per])) for i in range(2)]
    print(json.dumps(res, indent=1), flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"granularity_{args.sod_model}_{args.prompt}_{ds}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
