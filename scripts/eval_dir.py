"""Evaluate a directory of saliency maps <id><suffix> against a dataset's GT.

  python scripts/eval_dir.py --dataset DUTS-TE --dir <maps> [--suffix _sal.png]
Used to check the metrics against the MATLAB toolbox on the old outputs.
"""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from sodsam import data, metrics


def _one(id_, ds, directory, suffix):
    p = Path(directory) / (id_ + suffix)
    if not p.exists():
        return None
    pred = data.read_png(p)
    gt = data.load_gt(ds, id_)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
    return metrics.image_scores(pred, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dir", nargs="+", required=True)
    ap.add_argument("--suffix", default="_sal.png")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ids = data.list_ids(args.dataset)
    with Pool(args.workers) as pool:
        for d in args.dir:
            per = [s for s in pool.imap(partial(_one, ds=args.dataset, directory=d, suffix=args.suffix),
                                        ids, chunksize=16) if s is not None]
            print(metrics.format_row(Path(d).name[-40:], metrics.aggregate(per)), flush=True)


if __name__ == "__main__":
    main()
