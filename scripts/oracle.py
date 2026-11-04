"""Oracle upper bound: how much could SAM add to an SOD model if we always knew which map to trust?

Per image the candidates are the SOD map and the three SAM masks of a prompt. Using the ground truth,
the oracle picks the candidate with the highest per-image F (max over thresholds of its F curve);
dataset metrics are then computed on the picked maps as usual (F^max from the mean F curve, MAE).
Reported per dataset:
  sod           the SOD map alone
  oracle_sam    best of the three SAM masks (SAM alone with a perfect mask chooser)
  oracle_all    best of SOD map and SAM masks (fusion that only ever picks one source per image)
  sam_wins      fraction of images where the best SAM mask beats the SOD map by more than 0.02 F

  python scripts/oracle.py --sod-model samba --workers 2
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _image(id_, sod_model, ds, prompt):
    gt = data.load_gt(ds, id_)        # an empty GT counts (F = 0 for every candidate), as in sodsam.metrics
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sam_dir = data.run_dir(sod_model, ds, f"sam_{prompt}")
    cands = [sod] + [(data.read_png(sam_dir / f"{id_}_{k}.png") > 0).astype(np.uint8) * 255 for k in (1, 2, 3)]
    curves = [metrics.f_curve(c, gt) for c in cands]
    maes = [metrics.mae(c, gt) for c in cands]
    best = [float(c.max()) for c in curves]
    i_sam = 1 + int(np.argmax(best[1:]))
    i_all = int(np.argmax(best))
    return {"curves": {"sod": curves[0], "oracle_sam": curves[i_sam], "oracle_all": curves[i_all]},
            "mae": {"sod": maes[0], "oracle_sam": maes[i_sam], "oracle_all": maes[i_all]},
            "sam_wins": best[i_sam] > best[0] + 0.02, "picked_sam": i_all > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_soft")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt)
            per = [r for r in pool.imap(f, data.list_ids(ds), chunksize=16) if r is not None]
            res = {m: {"Fmax": float(np.mean([r["curves"][m] for r in per], axis=0).max()),
                       "MAE": float(np.mean([r["mae"][m] for r in per]))}
                   for m in ("sod", "oracle_sam", "oracle_all")}
            res["sam_wins"] = float(np.mean([r["sam_wins"] for r in per]))
            res["picked_sam"] = float(np.mean([r["picked_sam"] for r in per]))
            res["n"] = len(per)
            out[ds] = res
            print(f"{args.sod_model} {ds:10s} " + " ".join(f"{m}: F={res[m]['Fmax']:.3f} MAE={res[m]['MAE']:.3f} |"
                                                           for m in ("sod", "oracle_sam", "oracle_all"))
                  + f" SAM wins by >.02 on {res['sam_wins']:.1%}, oracle picks SAM on {res['picked_sam']:.1%}"
                  + f" (n={res['n']})", flush=True)
    (RESULTS / f"oracle_{args.sod_model}_{args.prompt}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
