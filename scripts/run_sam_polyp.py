"""SAM 2.1 with the ground-truth box prompt on the polyp datasets (sodsam.polyp): the standard
"SAM with box prompt" medical benchmark protocol (the box of the GT foreground is the prompt).

  python scripts/run_sam_polyp.py --datasets Kvasir-SEG CVC-ClinicDB
  python scripts/run_sam_polyp.py --fake --datasets Kvasir-SEG --limit 4     # CPU smoke test

Per image: set_image, then two predictions with the same GT box (xyxy, inclusive pixel coordinates of the
GT foreground; one box around all of it, also for images with several polyps):
  multi   multimask_output=True, the mask with the highest predicted IoU score;
  single  multimask_output=False.
Output under data.OUT_ROOT/polyp/<dataset>/sam_box/ (SODSAM_OUT on the cluster):
    <id>_multi.npy, <id>_single.npy   float16 logits cropped to `crop` (the GT box dilated by 10% of its
                                       width / height per side, padded by 1 pixel, clipped to the image)
    <id>_multi.png, <id>_single.png   full-image SAM masks (logit > 0), 0/255 (the baseline rows)
    <id>.json                         box, dbox (dilated box), crop (all xyxy inclusive), shape, scores
--limit N: only the first N train and the first N test ids. Existing ids are skipped unless --overwrite.
--fake replaces SAM by FakeRunner (logits of ellipses inscribed in the box), so the script runs without a GPU or torch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tqdm import tqdm

from sodsam import data, polyp

SUB = "sam_box"


def out_dir(ds):
    return data.OUT_ROOT / "polyp" / ds / SUB


def limited_ids(ds, limit):
    return polyp.list_ids(ds, "train")[:limit] + polyp.list_ids(ds, "test")[:limit]


class FakeRunner:
    """Stand-in for sam_prompting.SamRunner.box_logits: logit 6 * (1 - r^2 / s^2) of the ellipse inscribed
    in the box, r the normalized elliptic radius, s = 0.8, 1.0, 1.15 for the three masks (scores 0.7, 0.9,
    0.8) and s = 1.0 for the single mask; plus 2 * (red - mean red) / std of the image, so the colour matters."""

    def set_image(self, img):
        r = img[..., 0].astype(np.float64)
        self.col = 2 * (r - r.mean()) / (r.std() + 1e-9)

    def box_logits(self, box, multimask=False):
        h, w = self.col.shape
        x0, y0, x1, y1 = box
        yy, xx = np.mgrid[:h, :w]
        rr = (((xx - (x0 + x1) / 2) / max((x1 - x0) / 2, 1)) ** 2
              + ((yy - (y0 + y1) / 2) / max((y1 - y0) / 2, 1)) ** 2)
        scales = (0.8, 1.0, 1.15) if multimask else (1.0,)
        logits = np.stack([6 * (1 - rr / s ** 2) + self.col for s in scales]).astype(np.float32)
        return logits, np.array([0.7, 0.9, 0.8] if multimask else [0.9])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(polyp.DATASETS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N train and N test ids")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fake", action="store_true", help="FakeRunner instead of SAM (CPU, no torch)")
    args = ap.parse_args()

    if args.fake:
        sam = FakeRunner()
    else:
        from sodsam.sam_prompting import SamRunner
        sam = SamRunner()
    for ds in args.datasets:
        out = out_dir(ds)
        out.mkdir(parents=True, exist_ok=True)
        n_new, seconds, n_multi_comp = 0, 0.0, 0
        for id_ in tqdm(limited_ids(ds, args.limit), desc=f"{SUB}/{ds}"):
            meta = out / f"{id_}.json"
            if meta.exists() and not args.overwrite:
                continue
            t0 = time.time()
            img = polyp.load_image(ds, id_)
            gt = polyp.load_mask(ds, id_)
            if gt.shape != img.shape[:2]:
                raise ValueError(f"{ds}/{id_}: mask {gt.shape} vs image {img.shape[:2]}")
            box = polyp.gt_box(gt)
            if box is None:
                raise ValueError(f"{ds}/{id_}: empty ground truth")
            dbox = polyp.dilate_box(box, gt.shape)
            crop = polyp.pad_box(dbox, gt.shape)
            sl = (slice(crop[1], crop[3] + 1), slice(crop[0], crop[2] + 1))
            sam.set_image(img)
            lm, sm = sam.box_logits(box, multimask=True)
            ls, ss = sam.box_logits(box, multimask=False)
            best = int(np.argmax(sm))
            res = {"box": box, "dbox": dbox, "crop": crop, "shape": list(gt.shape),
                   "multi_scores": [float(s) for s in sm], "best": best, "single_score": float(ss[0]),
                   "fake": args.fake}
            for name, lg in (("multi", lm[best]), ("single", ls[0])):
                np.save(out / f"{id_}_{name}.npy", lg[sl].astype(np.float16))
                m = lg > 0
                data.write_png(out / f"{id_}_{name}.png", m.astype(np.uint8) * 255)
                res[f"{name}_outside_crop"] = int(m.sum() - m[sl].sum())
            meta.write_text(json.dumps(res))
            n_multi_comp += int(_n_components(gt) > 1)
            n_new += 1
            seconds += time.time() - t0
        print(f"{ds}: {n_new} new ids ({n_multi_comp} with several GT components), "
              f"{seconds / max(n_new, 1):.2f} s per image" + (" [FAKE]" if args.fake else ""), flush=True)


def _n_components(mask):
    import cv2
    return cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1


if __name__ == "__main__":
    main()
