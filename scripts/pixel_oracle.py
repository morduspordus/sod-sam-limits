"""Pixel-level oracle: per pixel, take the label of whichever candidate is right (SOD map binarized, SAM's
three masks, optionally the graph cut). The output equals the ground truth except where every candidate
is wrong, so it bounds every per-pixel combination of these sources (the graph cut among them).

The SOD map is binarized at the dataset's best threshold (argmax of its mean F curve, first pass).
Rows per dataset: sod (binary), oracle_sam (SAM only), oracle (sod + SAM), oracle_gc (+ gc_zoom when
saved), each with F (binary, so F = F^max), MAE and the fraction of pixels where all candidates are wrong.
Region-level rows (the four binary candidates give each pixel a 4-bit signature): oracle_rule_dataset (one
signature -> label table, the majority GT label over the dataset), oracle_rule_image (the same, per image),
oracle_region_all (majority GT label of each 4-connected cell of constant signature), and oracle_region_0.1 /
oracle_region_1 (the same for cells of at least 0.1% / 1% of the image, the dataset rule elsewhere).
Output: results/pixel_oracle_<sod>_<prompt>.json

  python scripts/pixel_oracle.py --sod-model samba --datasets DUTS-TE --workers 2
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
REGION_MIN = {"oracle_region_0.1": 0.001, "oracle_region_1": 0.01}     # cell area, fraction of the image


def _curve(id_, sod_model, ds):
    gt = data.load_gt(ds, id_)
    return metrics.f_curve(data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png"), gt)


def _oracle(cands, gt):
    """Label of a correct candidate where one exists, else the (wrong) common label."""
    right = np.zeros(gt.shape, bool)
    for c in cands:
        right |= c == gt
    return np.where(right, gt, ~gt)


def _candidates(id_, sod_model, ds, prompt, thr):
    sod = metrics._minmax(data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png"))
    sod = (sod * 255).astype(np.uint8) >= thr
    sam_dir = data.run_dir(sod_model, ds, f"sam_{prompt}")
    return sod, [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]


def _signature(sod, sams):
    """Per-pixel code 0..15 of the four binary candidates (bit 0: SOD, bits 1-3: SAM masks 1-3)."""
    sig = sod.astype(np.uint8)
    for k, m in enumerate(sams):
        sig |= m.astype(np.uint8) << (k + 1)
    return sig


def _sig_counts(id_, sod_model, ds, prompt, thr):
    """16 x 2 pixel counts of (signature, GT label), for the dataset-level rule."""
    gt = data.load_gt(ds, id_)
    sig = _signature(*_candidates(id_, sod_model, ds, prompt, thr))
    return np.stack([np.bincount(sig[~gt], minlength=16), np.bincount(sig[gt], minlength=16)], 1)


def _majority(labels, gt, n):
    """Each label 0..n-1 gets the GT label of the majority of its pixels (ties: background)."""
    fg = np.bincount(labels.ravel(), weights=gt.ravel(), minlength=n)
    tot = np.bincount(labels.ravel(), minlength=n)
    return (2 * fg > tot)[labels]


def _cells(sig):
    """Connected cells (4-connectivity) of constant signature, numbered 0..n-1."""
    cells = np.zeros(sig.shape, np.int32)
    n = 0
    for s in np.unique(sig):
        k, lab = cv2.connectedComponents((sig == s).astype(np.uint8), connectivity=4)
        cells[lab > 0] = lab[lab > 0] - 1 + n
        n += k - 1
    return cells, n


def _image(id_, sod_model, ds, prompt, thr, rule):
    gt = data.load_gt(ds, id_)
    sod, sams = _candidates(id_, sod_model, ds, prompt, thr)
    sets = {"sod": [sod], "oracle_sam": sams, "oracle": [sod] + sams}
    p = data.run_dir(sod_model, ds, "gc_zoom") / f"{id_}.png"
    if p.exists():
        sets["oracle_gc"] = [sod] + sams + [data.read_png(p) > 0]
    masks = {name: _oracle(cands, gt) for name, cands in sets.items()}
    # region-level oracles, between choosing one candidate per image and the per-pixel oracle: a fixed
    # rule on the four binary labels (fit on the whole dataset), the same rule fit per image, and the
    # majority label of each connected cell of constant signature. Most cells are slivers along boundaries on
    # which the masks do not quite agree (98% of them under 50 pixels), and labeling those is the pixel oracle
    # again; so only cells of at least REGION_MIN of the image get the oracle label, the rest the dataset rule.
    sig = _signature(sod, sams)
    cells, n = _cells(sig)
    masks["oracle_rule_dataset"] = rule[sig]
    masks["oracle_rule_image"] = _majority(sig, gt, 16)
    masks["oracle_region_all"] = _majority(cells, gt, n)
    size = np.bincount(cells.ravel(), minlength=n)
    for name, frac in REGION_MIN.items():
        big = (size >= frac * gt.size)[cells]
        masks[name] = np.where(big, masks["oracle_region_all"], rule[sig])
    out = {}
    for name, m in masks.items():
        out[name] = {"scores": metrics.image_scores(m.astype(np.uint8) * 255, gt), "wrong": float((m != gt).mean())}
    out["oracle_region_all"]["cells"] = n
    for name, frac in REGION_MIN.items():
        out[name]["cells"] = int((size >= frac * gt.size).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--datasets", nargs="+", default=["DUTS-TE"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            ids = data.list_ids(ds)
            curves = pool.map(partial(_curve, sod_model=args.sod_model, ds=ds), ids, chunksize=16)
            mean_curve = np.mean(curves, axis=0)
            thr = 255 - int(np.argmax(mean_curve))        # f_curve index i is the threshold ">= 255 - i"
            kw = dict(sod_model=args.sod_model, ds=ds, prompt=args.prompt, thr=thr)
            counts = np.sum(pool.map(partial(_sig_counts, **kw), ids, chunksize=16), axis=0)
            rule = counts[:, 1] > counts[:, 0]               # per signature: majority GT label over the dataset
            per = pool.map(partial(_image, rule=rule, **kw), ids, chunksize=16)
            res = {"sod_soft_Fmax": float(mean_curve.max()), "sod_threshold": thr, "n": len(per),
                   "rule_dataset": rule.astype(int).tolist(),
                   "cells_mean": {k: float(np.mean([r[k]["cells"] for r in per]))
                                  for k in ["oracle_region_all", *REGION_MIN]}}
            for name in per[0]:
                if all(name in r for r in per):
                    agg = metrics.aggregate([r[name]["scores"] for r in per])
                    agg["wrong_pixels"] = float(np.mean([r[name]["wrong"] for r in per]))
                    res[name] = agg
                    print(metrics.format_row(f"{args.sod_model} {ds} {name}", agg)
                          + f" wrong pixels {agg['wrong_pixels']:.4f}", flush=True)
            print(f"  (soft SOD F^max {res['sod_soft_Fmax']:.3f}, binarized at >= {thr})", flush=True)
            out[ds] = res
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"pixel_oracle_{args.sod_model}_{args.prompt}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
