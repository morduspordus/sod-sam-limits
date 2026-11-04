"""SAM2 on native-resolution crops along the boundary band of the SOD map (sodsam.crops).

  python scripts/run_sam_crops.py --sod-model isnet --datasets DIS-TE1 DIS-TE2 DIS-TE3 DIS-TE4
  python scripts/run_sam_crops.py --fake --sod-model isnet --datasets DIS-TE1 --limit 2   # CPU smoke test

Full-resolution dataset names. Per image: the band of radius crops.band_radius around the boundary of
the thresholded SOD map (output/<model>/<ds>/sod/<id>.png, image size), square crops of --size on a grid
of --stride over the band, SAM on each crop (set_image on the crop, the soft SOD map of the crop as the
mask prompt: three masks), and the stitched count (crops.stitch_counts). Output under
output/<model>/<ds>/sam_crops/:
    <id>_n.png     round(64 * count), uint8 in 0..192 (count = mean over the covering crops of the number
                   of masks containing the pixel)
    <id>_cov.png   0/255, the pixels covered by some crop
    <id>.json      the crop boxes (y0, x0, y1, x1), SAM's three scores per crop, the band radius
Existing ids are skipped unless --overwrite. --fake replaces SAM by FakeRunner (the SOD crop thresholded
at three levels), so the script runs without a GPU or torch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tqdm import tqdm

from sodsam import crops, data
from sodsam.crops import COUNT_SCALE, SUB


class FakeRunner:
    """Stand-in for sam_prompting.SamRunner: the three masks are the max-normalized SOD crop thresholded
    at 192/255, 127/255 and 64/255 (nested, like SAM's three), the scores their fraction of the crop."""

    def set_image(self, img):
        self.shape = img.shape[:2]

    def from_mask(self, sod_map, mode="soft", threshold=127.0):
        assert sod_map.shape == self.shape
        s = 255.0 * sod_map.astype(np.float64) / (sod_map.max() + 1e-13)
        masks = np.stack([s >= t for t in (192.0, 127.0, 64.0)])
        return masks, masks.reshape(3, -1).mean(1), masks.astype(np.float32)


def crop_image(sam, img, sod, boxes):
    """SAM's three masks and scores of every crop box."""
    masks, scores = [], []
    for y0, x0, y1, x1 in boxes:
        sam.set_image(np.ascontiguousarray(img[y0:y1, x0:x1]))
        mk, sc, _ = sam.from_mask(np.ascontiguousarray(sod[y0:y1, x0:x1]), "soft")
        masks.append(mk)
        scores.append([float(s) for s in sc])
    return masks, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True, choices=["birefnet", "isnet"])
    ap.add_argument("--datasets", nargs="+", default=["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"],
                    help="full-resolution names")
    ap.add_argument("--size", type=int, default=crops.CROP_SIZE)
    ap.add_argument("--stride", type=int, default=crops.CROP_STRIDE)
    ap.add_argument("--min-frac", type=float, default=crops.MIN_FRAC)
    ap.add_argument("--max-crops", type=int, default=crops.MAX_CROPS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--fake", action="store_true", help="FakeRunner instead of SAM (CPU, no torch)")
    args = ap.parse_args()

    if args.fake:
        sam = FakeRunner()
    else:
        from sodsam.sam_prompting import SamRunner
        sam = SamRunner()
    n_crops, n_images, seconds = [], 0, 0.0
    for ds in args.datasets:
        if ds.endswith(data.HR_SUFFIX):
            raise ValueError(f"{ds}: give the full-resolution name")
        out = data.run_dir(args.sod_model, ds, SUB)
        sod_dir = data.run_dir(args.sod_model, ds, "sod")
        for id_ in tqdm(data.list_ids(ds)[:args.limit], desc=f"{SUB}/{ds}"):
            meta = out / f"{id_}.json"
            if meta.exists() and not args.overwrite:
                n_crops.append(len(json.loads(meta.read_text())["crops"]))
                continue
            t0 = time.time()
            img = data.load_image(ds, id_)
            sod = data.read_png(sod_dir / f"{id_}.png")
            if sod.shape != img.shape[:2]:
                raise ValueError(f"{ds}/{id_}: SOD map {sod.shape} is not the image size {img.shape[:2]}")
            r = crops.band_radius(sod.shape)
            boxes = crops.place_crops(crops.boundary_band(sod, r), args.size, args.stride, args.min_frac,
                                      args.max_crops)
            masks, scores = crop_image(sam, img, sod, boxes)
            count, covered = crops.stitch_counts(sod.shape, boxes, masks)
            data.write_png(out / f"{id_}_n.png", np.round(COUNT_SCALE * count))
            data.write_png(out / f"{id_}_cov.png", covered.astype(np.uint8) * 255)
            meta.write_text(json.dumps({"crops": [[int(v) for v in b] for b in boxes], "scores": scores,
                                        "band_r": r, "shape": list(sod.shape), "size": args.size,
                                        "stride": args.stride, "fake": args.fake}))
            n_crops.append(len(boxes))
            n_images += 1
            seconds += time.time() - t0
    print(f"{args.sod_model} {SUB}: {len(n_crops)} ids, {np.mean(n_crops) if n_crops else 0:.2f} crops per image"
          f" (max {max(n_crops) if n_crops else 0}); {n_images} new, "
          f"{seconds / max(n_images, 1):.1f} s per new image" + (" [FAKE]" if args.fake else ""), flush=True)


if __name__ == "__main__":
    main()
