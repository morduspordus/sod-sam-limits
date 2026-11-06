"""Oracle over SAM's automatic ("everything") masks: is the room for SAM larger than its masks for our prompt?

Per image, with the ground truth, the oracle picks the best candidate by per-image F (max over thresholds of
its F curve, as in oracle.py). Candidates are the SOD map, the three masks of the zoom-in prompt, and SAM 2.1's
automatic masks (run_sam_extra.py --prompt auto: 32 x 32 point grid, 0.2% .. 95% of the image, stored at half
resolution; --auto auto_loose: the masks with SAM's quality thresholds lowered to pred_iou 0.6, stability 0.85,
about three times as many). Reported per dataset (F^max from the mean F curve, MAE):
  sod             the SOD map alone
  oracle_prompt   best of the SOD map and the prompted masks (Table "oracle" of the paper)
  auto_single     best single automatic mask
  auto_union      greedy union of automatic masks: start empty, add the mask that raises F most, stop when
                  none does (selected at half resolution, evaluated at full resolution)
  all_single      best of SOD map, prompted masks and single automatic masks
  all_union       the same plus the greedy union
Images without automatic masks get an empty mask for the automatic candidates.

  python scripts/oracle_auto.py --sod-model samba --workers 8
  python scripts/oracle_auto.py --sod-model samba --auto auto_loose
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
B2 = 0.3
KEYS = ("sod", "oracle_prompt", "auto_single", "auto_union", "all_single", "all_union")


def _auto_masks(ds, id_, auto="auto"):
    z = np.load(data.OUT_ROOT / f"sam_{auto}" / ds / f"{id_}.npz")
    h2, w2 = z["shape"]
    H, W = z["full_shape"]
    half = np.unpackbits(z["masks"], axis=-1)[..., :w2].astype(bool)
    return half, (int(H), int(W))


def _full(m, H, W):
    return np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)[:H, :W]


def _f(tp, n_pred, n_gt):
    if tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_gt
    return (1 + B2) * p * r / (B2 * p + r)


def _greedy_union(half, gt2):
    """Greedy union of half-resolution masks maximizing binary F against the half-resolution GT."""
    n_gt = int(gt2.sum())
    if n_gt == 0 or len(half) == 0:
        return np.zeros(gt2.shape, bool)
    flat = half.reshape(len(half), -1)
    g = gt2.reshape(-1)
    cur = np.zeros_like(g)
    best_f, left = 0.0, list(range(len(half)))
    while left:
        cand = [(_f(int((cur | flat[k])[g].sum()), int((cur | flat[k]).sum()), n_gt), k) for k in left]
        f, k = max(cand)
        if f <= best_f + 1e-9:
            break
        best_f, cur = f, cur | flat[k]
        left.remove(k)
    return cur.reshape(gt2.shape)


def _image(id_, sod_model, ds, prompt, auto="auto"):
    gt = data.load_gt(ds, id_)
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    sam_dir = data.run_dir(sod_model, ds, f"sam_{prompt}")
    prompted = [(data.read_png(sam_dir / f"{id_}_{k}.png") > 0).astype(np.uint8) * 255 for k in (1, 2, 3)]
    half, (H, W) = _auto_masks(ds, id_, auto)
    assert (H, W) == gt.shape, (ds, id_, (H, W), gt.shape)
    auto = [_full(m, H, W).astype(np.uint8) * 255 for m in half]
    union = _full(_greedy_union(half, gt[::2, ::2]), H, W).astype(np.uint8) * 255

    def ev(c):
        cur = metrics.f_curve(c, gt)
        return cur, float(cur.max()), metrics.mae(c, gt)

    e_sod = ev(sod)
    e_prompt = [ev(c) for c in prompted]
    e_auto = [ev(c) for c in auto] or [ev(np.zeros_like(sod))]
    e_union = ev(union)
    best = lambda es: max(es, key=lambda e: e[1])
    pick = {"sod": e_sod, "oracle_prompt": best([e_sod] + e_prompt), "auto_single": best(e_auto),
            "auto_union": e_union}
    pick["all_single"] = best([pick["oracle_prompt"], pick["auto_single"]])
    pick["all_union"] = best([pick["all_single"], e_union])
    return {"curves": {k: pick[k][0] for k in KEYS}, "mae": {k: pick[k][2] for k in KEYS},
            "n_auto": len(auto), "auto_beats_prompt": pick["auto_single"][1] > pick["oracle_prompt"][1] + 0.02}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--auto", default="auto", choices=["auto", "auto_loose"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt, auto=args.auto)
            per = list(pool.imap(f, data.list_ids(ds), chunksize=8))
            res = {k: {"Fmax": float(np.mean([r["curves"][k] for r in per], axis=0).max()),
                       "MAE": float(np.mean([r["mae"][k] for r in per]))} for k in KEYS}
            res["n_auto_mean"] = float(np.mean([r["n_auto"] for r in per]))
            res["no_auto"] = float(np.mean([r["n_auto"] == 0 for r in per]))
            res["auto_beats_prompt"] = float(np.mean([r["auto_beats_prompt"] for r in per]))
            res["n"] = len(per)
            out[ds] = res
            print(f"{args.sod_model} {ds:10s} " + " ".join(f"{k}={res[k]['Fmax']:.3f}" for k in KEYS)
                  + f" | {res['n_auto_mean']:.1f} auto masks, none on {res['no_auto']:.1%},"
                  f" auto beats prompt by >.02 on {res['auto_beats_prompt']:.1%}", flush=True)
    (RESULTS / f"oracle_{args.auto}_{args.sod_model}_{args.prompt}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
