"""Sizes of the cells of constant signature behind the pixel oracle's region rows (Sect. 5.1, "A pixel-level oracle").

pixel_oracle.py gives each pixel a 4-bit signature (SOD map binarized at its best dataset threshold, SAM's three
masks of the zoom-in prompt) and splits the image into 4-connected cells of constant signature; it saves only the
mean number of cells per image. Here, with its functions (imported), per image the number of cells and their
sizes: the mean number of cells per image, the share of cells under 50 pixels (pooled over the dataset and the
mean of the per-image shares), and the share of the image area in such cells. The SOD threshold is read from
results/granularity_<sod>_<prompt>_<dataset>.json (the same argmax of the mean F curve as pixel_oracle.py).
Output: results/sliver_stats.json

  python scripts/sliver_stats.py --workers 8
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from pixel_oracle import _candidates, _cells, _signature
from sodsam import data

RESULTS = Path(__file__).resolve().parents[1] / "results"
SMALL = 50


def _image(id_, sod_model, ds, prompt, thr):
    cells, n = _cells(_signature(*_candidates(id_, sod_model, ds, prompt, thr)))
    size = np.bincount(cells.ravel(), minlength=n)
    return n, int((size < SMALL).sum()), float(size[size < SMALL].sum() / cells.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sods", nargs="+", default=["selfreformer", "samba"])
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--dataset", default="DUTS-TE")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        for sod in args.sods:
            thr = json.loads((RESULTS / f"granularity_{sod}_{args.prompt}_{args.dataset}.json").read_text())["threshold"]
            f = partial(_image, sod_model=sod, ds=args.dataset, prompt=args.prompt, thr=thr)
            per = np.array(pool.map(f, data.list_ids(args.dataset), chunksize=16), dtype=np.float64)
            res = {"dataset": args.dataset, "threshold": thr, "n": len(per),
                   "cells_mean": float(per[:, 0].mean()), "cells_median": float(np.median(per[:, 0])),
                   f"small_share_pooled": float(per[:, 1].sum() / per[:, 0].sum()),
                   f"small_share_image_mean": float(np.mean(per[:, 1] / np.maximum(per[:, 0], 1))),
                   f"small_area_share": float(per[:, 2].mean()), "small_px": SMALL}
            out[sod] = res
            print(sod, json.dumps(res), flush=True)
    (RESULTS / "sliver_stats.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
