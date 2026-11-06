"""Overlap between the fusion head's training images (MSRA-B4K, or --train DUTS-TR for the ablation head) and its
validation images (MSOD, MSRA-B1K), with the duplicate test of duplicates.py (8x8 thumbnail neighbours, 32x32
correlation >= 0.97, aspect within 3%). Writes the duplicate ids to results/val_overlap.json
(val_overlap_<train>.json for another training set).

  python scripts/val_overlap.py --workers 4
  python scripts/val_overlap.py --train DUTS-TR --workers 8
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from duplicates import _thumb
from sodsam import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train", default="MSRA-B4K")
    args = ap.parse_args()
    from sklearn.neighbors import NearestNeighbors
    sets = ["MSOD", "MSRA-B1K", args.train]
    items = [(ds, i, data.image_path(ds, i)) for ds in sets for i in data.list_ids(ds)]
    for ds in sets:
        print(ds, sum(1 for x in items if x[0] == ds), "images", flush=True)
    with Pool(args.workers) as pool:
        th = pool.map(_thumb, [p for _, _, p in items], chunksize=64)
    train = [k for k, x in enumerate(items) if x[0] == args.train]
    X8 = np.stack([th[k][0] for k in train])
    nn = NearestNeighbors(n_neighbors=4).fit(X8)
    res = {}
    for ds in ("MSOD", "MSRA-B1K"):
        val = [k for k, x in enumerate(items) if x[0] == ds]
        _, idx = nn.kneighbors(np.stack([th[k][0] for k in val]))
        dup = []
        for k, row in zip(val, idx):
            for j in row:
                t = train[int(j)]
                if abs(th[k][2] / th[t][2] - 1) < 0.03 and float(th[k][1] @ th[t][1]) >= 0.97:
                    dup.append([items[k][1], items[t][1]])
                    break
        res[ds] = {"n": len(val), "dup": dup}
        print(f"{ds}: {len(dup)} of {len(val)} validation images are also in {args.train}", flush=True)
    out = Path(__file__).resolve().parents[1] / "results" / ("val_overlap.json" if args.train == "MSRA-B4K"
                                                           else f"val_overlap_{args.train}.json")
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
