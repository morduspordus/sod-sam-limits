"""Pixel-level decomposition of the fusion's gain (third review): how much of the F gain of the graph cut (or
the head) over the SOD map comes from the pixels near the ground-truth contour, and how much from those far
from it.

Per image (ground truth G non-empty): S the SOD map and R the fused map, each binarized at its dataset's best
threshold (argmax of its mean F curve, as in pixel_oracle.py), so that the mean per-image F of S and R are
their F^max, and the mean of F(R) - F(S) over all images (empty ground truths count as 0 for both) is the gain
in F^max. Pixels are binned by their distance d to the contour of G (as in error_robust.py), in pixels (<= 3,
(3, 5], (5, 10], > 10 px) and relative to the object's equivalent radius r = sqrt(|G| / pi) (d / r <= 0.05,
(0.05, 0.1], (0.1, 0.2], > 0.2). The gain of a bin b is exact, not linearized: the mean of
  F(S with the labels of R in b) - F(S)   and   F(R) - F(R with the labels of S in b),
averaged over the images like F^max. The bins' gains add up to the total only approximately (they interact
through the F denominator); both are reported. For reference, the share of the loss 1 - F of S in each bin
(false positives weighted 1 and false negatives 0.3, as they enter 1 - F_beta with beta^2 = 0.3).

  python -I scripts/gain_pixel.py --sod-model samba --fused gc --workers 8
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

OUT = Path(__file__).resolve().parents[1] / "results" / "gain_pixel"
DATASETS = ["DUTS-TE", "ECSSD", "DUT-OMRON", "PASCAL-S", "HKU-IS"]
ABS = (3, 5, 10)                 # px; the last bin is > 10 px
REL = (0.05, 0.1, 0.2)           # fraction of the equivalent radius; the last bin is > 0.2


def map_path(sod, ds, which, id_):
    if which == "sod":
        return data.run_dir(sod, ds, "sod") / f"{id_}.png"
    if which == "gc":
        return data.run_dir(sod, ds, "gc_zoom") / f"{id_}.png"
    return data.run_dir(f"{sod}_hz", ds, "sod") / f"{id_}.png"


def load(sod, ds, which, id_, shape):
    m = data.read_png(map_path(sod, ds, which, id_))
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return m


def _curve(id_, ds, sod, which):
    gt = data.load_gt(ds, id_)
    return metrics.f_curve(load(sod, ds, which, id_, gt.shape), gt)


def binarize(m, thr):
    """As metrics.f_curve: the min-max normalized map, quantized to 0..255, >= thr."""
    return (metrics._minmax(m) * 255).astype(np.uint8) >= thr


def f_beta(M, G):
    tp = np.count_nonzero(M & G)
    den = 1.3 * tp + 0.3 * np.count_nonzero(~M & G) + np.count_nonzero(M & ~G)
    return 1.3 * tp / den if den else 0.0


def bins(d, edges):
    """Boolean masks of the bins (<= e1, (e1, e2], ..., > e_last) of the values d."""
    out, lo = [], -1.0
    for e in edges:
        out.append((d > lo) & (d <= e))
        lo = e
    out.append(d > lo)
    return out


def _image(id_, ds, sod, fused, thr):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    G = gt.astype(bool)
    S = binarize(load(sod, ds, "sod", id_, G.shape), thr["sod"])
    R = binarize(load(sod, ds, fused, id_, G.shape), thr[fused])
    d_in = cv2.distanceTransform(G.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, 5)
    dist = np.where(G, d_in, d_out)
    f_s, f_r = f_beta(S, G), f_beta(R, G)
    den_s = 1.3 * np.count_nonzero(S & G) + 0.3 * np.count_nonzero(~S & G) + np.count_nonzero(S & ~G)
    w_loss = np.where(S, 1.0, 0.3) * (S ^ G) / den_s           # sums to 1 - F of S
    out = {"f_sod": f_s, "f_fused": f_r}
    for scale, d, edges in (("abs", dist, ABS), ("rel", dist / np.sqrt(G.sum() / np.pi), REL)):
        gain, loss = [], []
        for b in bins(d, edges):
            gain.append(0.5 * ((f_beta(np.where(b, R, S), G) - f_s) + (f_r - f_beta(np.where(b, S, R), G))))
            loss.append(float(w_loss[b].sum()))
        out[scale] = {"gain": gain, "loss": loss}
    return out


def summarize(recs, n_all, scale):
    gain = np.sum([r[scale]["gain"] for r in recs], 0) / n_all
    loss = np.sum([r[scale]["loss"] for r in recs], 0) / n_all
    total = sum(r["f_fused"] - r["f_sod"] for r in recs) / n_all
    return {"gain": gain.tolist(), "gain_share": (gain / gain.sum()).tolist() if gain.sum() else None,
            "gain_total": float(total), "gain_sum_of_bins": float(gain.sum()),
            "loss": loss.tolist(), "loss_share": (loss / loss.sum()).tolist(),
            "reduction": (gain / np.maximum(loss, 1e-12)).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--fused", choices=["gc", "head"], default="gc")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"bins_abs": [f"<={e}px" for e in ABS] + [f">{ABS[-1]}px"],
           "bins_rel": [f"<={e}r" for e in REL] + [f">{REL[-1]}r"]}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            ids = data.list_ids(ds)
            thr = {}
            for which in ("sod", args.fused):            # f_curve index i is the threshold ">= 255 - i"
                curves = pool.map(partial(_curve, ds=ds, sod=args.sod_model, which=which), ids, chunksize=16)
                thr[which] = 255 - int(np.argmax(np.mean(curves, axis=0)))
            recs = [r for r in pool.imap(partial(_image, ds=ds, sod=args.sod_model, fused=args.fused, thr=thr),
                                         ids, chunksize=16) if r]
            res[ds] = {"n": len(ids), "n_nonempty": len(recs), "thresholds": thr,
                       "abs": summarize(recs, len(ids), "abs"), "rel": summarize(recs, len(ids), "rel")}
            a = res[ds]["abs"]
            print(f"{args.sod_model} {args.fused} {ds}: gain {a['gain_total']:.4f} (bins {a['gain_sum_of_bins']:.4f}) "
                  f"gain share {np.round(a['gain_share'], 2)} loss share {np.round(a['loss_share'], 2)} "
                  f"rel gain share {np.round(res[ds]['rel']['gain_share'], 2)}", flush=True)
    (OUT / f"{args.sod_model}_{args.fused}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
