"""Robustness of the error decomposition of error_types.py (second review).

Per image (ground truth G non-empty), with S the thresholded SOD mask (max-normalized map >= 127/255, as in
error_types.py) and loss = 1 - per-image F of the soft map, it records |S & G|, |S|, |G|, |S | G| and the
wrong pixels of S by their distance to the ground-truth contour. From these:
  1. pixel-level decomposition: the share of all wrong pixels (S xor G) within k = 1, 3, 5, 10, 20 px of the
     contour of G, pooled over the dataset and as the mean of the per-image shares;
  2. image-level shares of error_types.py under three weightings: loss (1 - F, the paper), unweighted
     (image counts), and wrong pixels as a fraction of the image;
  3. a sweep of the four thresholds of the taxonomy (boundary IoU, containment, part/whole ratio, lower
     extent ratio), with the loss weighting: the range of each share over the grid.
The precedence of the types is that of error_types.classify: boundary, part/whole, extent, other, mixed.

  python -I scripts/error_robust.py --sod-model samba
  python -I scripts/error_robust.py --maps-root <dir with <dataset>/<id>.png> --name vscode
"""
import argparse
import itertools
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from sodsam import data, metrics

OUT = Path(__file__).resolve().parents[1] / "results" / "error_robust"
KS = (1, 3, 5, 10, 20)
GROUPS = {"boundary": ("boundary",), "part/whole": ("part_of_gt", "whole_of_gt"),
          "extent": ("under_extent", "over_extent"), "other object": ("other_object",), "mixed": ("mixed",)}
PAPER = {"iou": 0.85, "cont": 0.9, "pw": 2.0, "ext": 1.15}
GRID = {"iou": (0.75, 0.80, 0.85, 0.90), "cont": (0.8, 0.9, 0.95), "pw": (1.5, 2.0, 3.0), "ext": (1.05, 1.15, 1.3)}


def classify(r, iou_t, cont_t, pw_t, ext_t):
    """error_types.classify with its thresholds as parameters (the paper: 0.85, 0.9, 2, 1.15)."""
    inter, s, g, union = r["inter"], r["s"], r["g"], r["union"]
    iou = inter / union if union else 1.0
    s_in_g = inter / s if s else 0.0
    g_in_s = inter / g if g else 0.0
    if iou >= iou_t:
        return "boundary"
    if s and s_in_g >= cont_t and g >= pw_t * s:
        return "part_of_gt"
    if g_in_s >= cont_t and s >= pw_t * g:
        return "whole_of_gt"
    if s and s_in_g >= cont_t and g > ext_t * s:
        return "under_extent"
    if g_in_s >= cont_t and s > ext_t * g:
        return "over_extent"
    if iou < 0.5 and s_in_g < cont_t and g_in_s < cont_t:
        return "other_object"
    return "mixed"


def group_of(cat):
    return next(k for k, v in GROUPS.items() if cat in v)


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
        return {"id": id_, "missing": True}
    sod = data.read_png(p)
    if sod.shape != gt.shape:
        sod = cv2.resize(sod, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    loss = 1.0 - float(metrics.f_curve(sod, gt).max())
    S = sod.astype(np.float64) / (sod.max() + 1e-13) >= 127 / 255 if sod.max() > 0 else np.zeros_like(gt)
    G = gt.astype(bool)
    # distance of every pixel to the contour of G: to the nearest pixel of the other side
    d_in = cv2.distanceTransform(G.astype(np.uint8), cv2.DIST_L2, 5)        # inside G: to the background
    d_out = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 5)    # outside G: to G
    dist = np.where(G, d_in, d_out)
    wrong = S ^ G
    dw = dist[wrong]
    return {"id": id_, "missing": False, "loss": loss, "inter": int((S & G).sum()), "s": int(S.sum()),
            "g": int(G.sum()), "union": int((S | G).sum()), "npix": int(G.size), "wrong": int(wrong.sum()),
            "near": [int((dw <= k).sum()) for k in KS]}


def shares(recs, weight, th):
    tot = sum(weight(r) for r in recs)
    out = {k: 0.0 for k in GROUPS}
    for r in recs:
        out[group_of(classify(r, *th))] += weight(r)
    return {k: v / max(tot, 1e-12) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model")
    ap.add_argument("--maps-root")
    ap.add_argument("--name")
    ap.add_argument("--datasets", nargs="+", default=["DUTS-TE", "DUT-OMRON"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    name = args.name or args.sod_model
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    paper = tuple(PAPER[k] for k in ("iou", "cont", "pw", "ext"))
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            recs = [r for r in pool.imap(partial(_image, ds=ds, args=args), data.list_ids(ds), chunksize=16) if r]
            missing = sum(r["missing"] for r in recs)
            recs = [r for r in recs if not r["missing"]]
            wrong_tot = sum(r["wrong"] for r in recs)
            pix = {f"<={k}px": {"pooled": sum(r["near"][i] for r in recs) / max(wrong_tot, 1),
                                "per_image": float(np.mean([r["near"][i] / r["wrong"] for r in recs if r["wrong"]]))}
                   for i, k in enumerate(KS)}
            weights = {"loss": lambda r: r["loss"], "count": lambda r: 1.0,
                       "wrong_frac": lambda r: r["wrong"] / r["npix"]}
            img = {w: shares(recs, f, paper) for w, f in weights.items()}
            grid = []
            for combo in itertools.product(*(GRID[k] for k in ("iou", "cont", "pw", "ext"))):
                grid.append({"th": combo, "share": shares(recs, weights["loss"], combo)})
            rng = {g: [min(x["share"][g] for x in grid), max(x["share"][g] for x in grid)] for g in GROUPS}
            one = {}                     # one threshold at a time, the others at the paper's values
            for i, k in enumerate(("iou", "cont", "pw", "ext")):
                for v in GRID[k]:
                    th = list(paper)
                    th[i] = v
                    one[f"{k}={v}"] = shares(recs, weights["loss"], tuple(th))
            res[ds] = {"n": len(recs), "missing": missing, "pixel": pix, "image": img, "grid_range": rng,
                       "one_at_a_time": one}
            print(f"{name} {ds} n={len(recs)} missing={missing}", flush=True)
            print("  wrong pixels within k px of the GT contour (pooled / per-image mean): "
                  + "  ".join(f"{k} {v['pooled']:.0%}/{v['per_image']:.0%}" for k, v in pix.items()), flush=True)
            for w, s in img.items():
                print(f"  {w:10s} " + "  ".join(f"{g} {v:.0%}" for g, v in s.items()), flush=True)
            print("  grid range " + "  ".join(f"{g} {a:.0%}-{b:.0%}" for g, (a, b) in rng.items()), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
