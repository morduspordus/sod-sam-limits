"""SAMRefiner baseline (Lin et al., ICLR 2025): refine the thresholded SOD mask with SAM ViT-H, as published.

The coarse mask is the SOD map max-normalized and thresholded at 127/255 (as in error_types.py); SAMRefiner's
default settings (box + point + Gaussian mask prompts, negative points, 5 iterations). An empty coarse mask
keeps the SOD map (and gives three empty prompt masks). Writes, per image:
  <OUT_ROOT>/<sod>/<ds>/samrefiner/<id>.png       the refined mask (the baseline)
  <OUT_ROOT>/<sod>/<ds>/sam_mask_srf/<id>_k.png   the three masks of SAMRefiner's last iteration, k = 1..3 in
                    SAM's output order, with <id>_k_p.png (sigmoid probabilities) and scores.json (predicted IoU),
                    in the format of run_sam.py, so that the fusion methods can use them as --prompt mask_srf
and results/eval_<sod>_samrefiner.json (metrics of the refined mask on the datasets that have GT).

  python scripts/run_samrefiner.py --repo <SAMRefiner checkout> --ckpt sam_vit_h_4b8939.pth --sod-model samba
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from sodsam import data, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.repo) / "segment-anything"))
    sys.path.insert(0, args.repo)
    from segment_anything import sam_model_registry
    from sam_refiner import sam_refiner

    sam = sam_model_registry["vit_h"](checkpoint=args.ckpt).to("cuda").eval()
    p_res = Path(__file__).resolve().parents[1] / "results" / f"eval_{args.sod_model}_samrefiner.json"
    res = json.loads(p_res.read_text()) if p_res.exists() else {}
    for ds in args.datasets:
        out = data.run_dir(args.sod_model, ds, "samrefiner")
        out_p = data.run_dir(args.sod_model, ds, "sam_mask_srf")
        out.mkdir(parents=True, exist_ok=True)
        out_p.mkdir(parents=True, exist_ok=True)
        ids = data.list_ids(ds)
        scores, ious, t0, n_empty, n_resized = [], {}, time.time(), 0, 0
        for i in ids:
            img_path = data.image_path(ds, i)
            s = data.read_png(data.run_dir(args.sod_model, ds, "sod") / f"{i}.png")
            h, w = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).shape[:2]
            coarse = (s.astype(np.float64) / (s.max() + 1e-13) >= 127 / 255) if s.max() > 0 else np.zeros(s.shape, bool)
            if s.shape != (h, w):                   # SAMRefiner takes the original size from the mask
                coarse = cv2.resize(coarse.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                n_resized += 1
            if coarse.any():
                ref, iou, logits = sam_refiner(str(img_path), [coarse.astype(np.uint8)], sam)
                pred = ref[0].astype(np.uint8) * 255
                logits = logits[0].float().cpu().numpy()                  # 3 x h x w, last iteration
                iou = [float(v) for v in iou[0].cpu()]
            else:
                pred, n_empty = s, n_empty + 1
                logits, iou = np.full((3, h, w), -20.0, np.float32), [0.0, 0.0, 0.0]
            if pred.shape != s.shape:               # back to the size of the SOD map, as the other outputs
                pred = cv2.resize(pred, (s.shape[1], s.shape[0]), interpolation=cv2.INTER_NEAREST)
                logits = np.stack([cv2.resize(l, (s.shape[1], s.shape[0]), interpolation=cv2.INTER_LINEAR)
                                   for l in logits])
            data.write_png(out / f"{i}.png", pred)
            for k in range(3):
                data.write_png(out_p / f"{i}_{k + 1}.png", (logits[k] > 0).astype(np.uint8) * 255)
                data.write_png(out_p / f"{i}_{k + 1}_p.png", np.round(255 / (1 + np.exp(-np.clip(logits[k], -50, 50)))))
            ious[i] = iou
            g = data.load_gt(ds, i)
            if pred.shape != g.shape:
                pred = cv2.resize(pred, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
            scores.append(metrics.image_scores(pred, g))
        (out_p / "scores.json").write_text(json.dumps(ious))
        r = metrics.aggregate(scores)
        res[ds] = {"samrefiner": r, "n_empty": n_empty, "n_resized": n_resized,
                   "sec_per_image": (time.time() - t0) / len(ids)}
        print(f"{args.sod_model} {ds:10s} samrefiner F={r['Fmax']:.3f} MAE={r['MAE']:.3f} E={r['E']:.3f} "
              f"S={r['S']:.3f} (n={len(scores)}, empty {n_empty}, resized {n_resized}, "
              f"{res[ds]['sec_per_image']:.2f} s/img)", flush=True)
        p_res.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
