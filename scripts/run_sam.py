"""Run SAM2 with a given prompt and cache its masks.

Prompted by an SOD model (three masks each):
  python scripts/run_sam.py --sod-model selfreformer --prompt mask_soft --datasets ...
  python scripts/run_sam.py --sod-model selfreformer --prompt mask_binary --datasets ...
Complement prompt (the inverted map, so SAM segments the background; used as a veto):
  python scripts/run_sam.py --sod-model selfreformer --prompt mask_inv --save-probs --datasets ...
Zero-shot (Table 1; single mask, saved as <id>_1.png):
  python scripts/run_sam.py --sod-model none --prompt box_margin --margins 0 20 40 60 80 100 --datasets DUTS-TE
  python scripts/run_sam.py --sod-model none --prompt center --datasets DUTS-TE
  python scripts/run_sam.py --sod-model selfreformer --prompt box_sod --datasets DUTS-TE
Stronger naive baselines (single mask, saved as <id>_1.png; review of the paper):
  mask_zoom1  the zoom-in mask prompt with SAM's single-mask output
  box_thr     the box of the thresholded SOD mask (max-normalized map >= 127), single mask
  box_mask1   that box together with the soft mask prompt (on the whole image), single mask
  python scripts/run_sam.py --sod-model samba --prompt box_thr box_mask1 mask_zoom1 --datasets DUTS-TE
Mask encodings of the zoom-in prompt (second review): mask_zoom_binary (+-10), mask_zoom_logit (2 logit s),
mask_zoom_signed (10 (2s - 1)), mask_zoom_pts (the paper's soft encoding plus a positive and a negative point)
  python scripts/run_sam.py --sod-model samba --prompt mask_zoom_binary mask_zoom_logit mask_zoom_signed \
      mask_zoom_pts --save-probs --datasets MSOD MSRA-B1K DUTS-TE
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tqdm import tqdm

from sodsam import data
from sodsam.sam_prompting import SamRunner, margin_box, sod_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", required=True, nargs="+",
                    choices=["mask_soft", "mask_binary", "mask_logit", "box_sod", "box_margin", "center",
                             "mask_comp", "mask_erode", "mask_dilate", "box_mask", "mask_flip", "mask_inv",
                             "mask_zoom", "mask_zoom1", "box_thr", "box_mask1",
                             "mask_zoom_binary", "mask_zoom_logit", "mask_zoom_signed", "mask_zoom_pts"],
                    help="several prompts share one image encoding per image")
    ap.add_argument("--suffix", default="", help="appended to the output subdir, e.g. _hq for another SAM")
    ap.add_argument("--save-probs", action="store_true",
                    help="also write the mask probabilities as <id>_<k>_p.png (0..255)")
    ap.add_argument("--datasets", nargs="+", default=[data.TUNE_DATASET] + data.EVAL_DATASETS)
    ap.add_argument("--margins", nargs="+", type=int, default=[0, 20, 40, 60, 80, 100])
    ap.add_argument("--legacy-box", action="store_true",
                    help="reproduce the swapped x/y of apply_sam_zero_shot_with_box.py")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    sam = SamRunner()
    for ds in args.datasets:
        # one (prompt, margin, subdir) per output directory
        runs = []
        for p in args.prompt:
            if p == "box_margin":
                runs += [(p, m, f"sam_box_margin{m}{'_legacy' if args.legacy_box else ''}") for m in args.margins]
            else:
                runs.append((p, None, f"sam_{p}{args.suffix}"))
        dirs = [data.run_dir(args.sod_model, ds, s) for _, _, s in runs]
        all_scores = []
        for d in dirs:
            f = d / "scores.json"
            all_scores.append(json.loads(f.read_text()) if f.exists() and not args.overwrite else {})
        def save_scores():
            for d, scores in zip(dirs, all_scores):
                d.mkdir(parents=True, exist_ok=True)
                (d / "scores.json").write_text(json.dumps(scores))

        for n, id_ in enumerate(tqdm(data.list_ids(ds), desc=f"{'+'.join(args.prompt)}/{ds}")):
            if all(id_ in s for s in all_scores):
                continue
            if n % 500 == 499:
                save_scores()
            img = data.load_image(ds, id_)
            h, w = img.shape[:2]
            sam.set_image(img)
            sod = None
            if any(p.startswith("mask_") or p in ("box_sod", "box_mask", "box_thr", "box_mask1") for p in args.prompt):
                sod = data.read_png(data.run_dir(args.sod_model, ds, "sod") / f"{id_}.png")
            # the flipped-image and zoom prompts need their own image encoding: do them last
            order = sorted(range(len(runs)), key=lambda r: runs[r][0] == "mask_flip" or runs[r][0].startswith("mask_zoom"))
            flipped = False
            for r in order:
                (p, m, _), d, scores = runs[r], dirs[r], all_scores[r]
                if id_ in scores:
                    continue
                if p == "mask_flip":
                    if not flipped:
                        sam.set_image(np.ascontiguousarray(img[:, ::-1]))
                        flipped = True
                    masks, sc, probs = sam.from_mask(np.ascontiguousarray(sod[:, ::-1]), "soft")
                    masks, probs = masks[:, :, ::-1], probs[:, :, ::-1]
                elif p == "mask_comp":
                    masks, sc, probs = sam.from_components(sod)
                elif p in ("mask_erode", "mask_dilate"):
                    masks, sc, probs = sam.from_morph(sod, p[len("mask_"):])
                elif p == "box_mask":
                    masks, sc, probs = sam.from_mask_and_box(sod)
                elif p == "mask_zoom":
                    masks, sc, probs = sam.from_zoom(img, sod)
                elif p == "mask_zoom1":
                    masks, sc, probs = sam.from_zoom(img, sod, multimask=False)
                elif p == "mask_zoom_pts":       # the paper's soft zoom prompt plus a positive and a negative point
                    masks, sc, probs = sam.from_zoom(img, sod, mode="soft", pts=True)
                elif p.startswith("mask_zoom_"):  # the zoom-in prompt with another mask encoding
                    masks, sc, probs = sam.from_zoom(img, sod, mode=p[len("mask_zoom_"):])
                elif p == "box_mask1":
                    masks, sc, probs = sam.from_mask_and_box(sod, multimask=False)
                elif p == "box_thr":
                    masks, sc, probs = sam.from_box(sod_box(sod, 127.0))
                elif p == "mask_inv":
                    # complement prompt: SAM segments the background object(s); 255 - map, so the
                    # prompt is the exact mirror of mask_soft (no max-normalization, as there)
                    masks, sc, probs = sam.from_mask(255 - sod, "soft")
                elif p.startswith("mask_"):
                    masks, sc, probs = sam.from_mask(sod, p[len("mask_"):])
                elif p == "box_sod":
                    masks, sc, probs = sam.from_box(sod_box(sod))
                elif p == "box_margin":
                    box = [m, m, h - m, w - m] if args.legacy_box else margin_box(h, w, m)
                    masks, sc, probs = sam.from_box(box)
                else:
                    masks, sc, probs = sam.from_point((w / 2, h / 2))
                for j, mk in enumerate(masks, 1):
                    data.write_png(d / f"{id_}_{j}.png", mk.astype(np.uint8) * 255)
                    if args.save_probs:
                        data.write_png(d / f"{id_}_{j}_p.png", np.round(probs[j - 1] * 255))
                scores[id_] = [float(s) for s in sc]
        save_scores()


if __name__ == "__main__":
    main()
