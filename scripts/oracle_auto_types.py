"""Where is the extra room of SAM's automatic masks? oracle_auto.py per image, split by the SOD map's error type.

For each image, with oracle_auto.py's candidates and functions (imported, not copied), the per-image F (max over
thresholds of the image's F curve) of
  sod             the SOD map
  oracle_prompt   best of the SOD map and the three prompted masks
  all_single      best of these and the single automatic masks
  all_union       the same plus the greedy union of automatic masks
and the error type of the SOD map (error_types.classify on the map at half its maximum, the taxonomy of Sect. 5.4
of the paper; images with an empty ground truth get "empty" and have F = 0 for every candidate). The added
oracle gain of the automatic masks, sum over images of F(all_single) - F(oracle_prompt) (a reduction of the summed
loss 1 - F), is split by the type of the image; the same for all_union and, for comparison, for the prompted
oracle's own gain F(oracle_prompt) - F(sod) and the SOD map's loss 1 - F(sod).
Per-image arrays and the per-type sums: results/oracle_auto_types_<sod>.json. The dataset F^max of each
candidate set (from the mean F curve) is recomputed as a check against results/oracle_<auto>_<sod>_<prompt>.json.

  python scripts/oracle_auto_types.py --sod-model samba --workers 8
"""
import argparse
import json
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

import oracle_auto
from error_types import CATS, classify
from review_tables import TYPES
from sodsam import data

RESULTS = Path(__file__).resolve().parents[1] / "results"
KEYS = ("sod", "oracle_prompt", "all_single", "all_union")
COARSE = {c: name for name, cats in TYPES.items() for c in cats}


def _type(id_, sod_model, ds):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return "empty"
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    S = sod.astype(np.float64) / (sod.max() + 1e-13) >= 127 / 255 if sod.max() > 0 else np.zeros_like(gt)
    return classify(S, gt)[0]


def _image(id_, sod_model, ds, prompt, auto):
    r = oracle_auto._image(id_, sod_model, ds, prompt, auto)
    return {"id": id_, "type": _type(id_, sod_model, ds), "n_auto": r["n_auto"],
            "f": {k: float(r["curves"][k].max()) for k in KEYS},
            "curves": {k: r["curves"][k] for k in KEYS}}


def _split(per, a, b):
    """Sum over images of F(b) - F(a) per fine and coarse SOD error type (a = None: the loss 1 - F(b))."""
    fine = defaultdict(float)
    for r in per:
        fine[r["type"]] += (1.0 - r["f"][b]) if a is None else r["f"][b] - r["f"][a]
    coarse = defaultdict(float)
    for c, v in fine.items():
        coarse[COARSE.get(c, c)] += v
    return {"total": sum(fine.values()), "fine": dict(fine), "coarse": dict(coarse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--auto", default="auto_loose", choices=["auto", "auto_loose"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    ref_path = RESULTS / f"oracle_{args.auto}_{args.sod_model}_{args.prompt}.json"
    ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    out = {"auto": args.auto, "prompt": args.prompt, "datasets": {}}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt, auto=args.auto)
            per = list(pool.imap(f, data.list_ids(ds), chunksize=8))
            fmax = {k: float(np.mean([r["curves"][k] for r in per], axis=0).max()) for k in KEYS}
            check = {k: fmax[k] - ref[ds][k]["Fmax"] for k in KEYS if ds in ref}
            res = {"n": len(per), "Fmax": fmax, "Fmax_minus_oracle_auto_json": check,
                   "type_count": {c: sum(r["type"] == c for r in per) for c in CATS + ("empty",)},
                   "sod_loss": _split(per, None, "sod"),
                   "prompt_gain": _split(per, "sod", "oracle_prompt"),
                   "auto_gain_single": _split(per, "oracle_prompt", "all_single"),
                   "auto_gain_union": _split(per, "oracle_prompt", "all_union"),
                   "images": {"id": [r["id"] for r in per], "type": [r["type"] for r in per],
                              "n_auto": [r["n_auto"] for r in per],
                              **{f"f_{k}": [round(r["f"][k], 6) for r in per] for k in KEYS}}}
            out["datasets"][ds] = res
            g = res["auto_gain_single"]
            print(f"{args.sod_model} {ds:10s} n={len(per)} " + " ".join(f"{k}={fmax[k]:.4f}" for k in KEYS)
                  + f" | check vs oracle_auto json: " + " ".join(f"{v:+.1e}" for v in check.values())
                  + f" | auto gain (single) {g['total'] / len(per):.4f}/image: "
                  + " ".join(f"{c} {v / g['total']:.0%}" for c, v in g["coarse"].items() if g["total"]), flush=True)
    (RESULTS / f"oracle_auto_types_{args.sod_model}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
