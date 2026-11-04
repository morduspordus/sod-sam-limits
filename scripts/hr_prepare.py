"""Make the "<name>-1k" copies of the high-resolution benchmarks that the pipeline runs on.

  python scripts/hr_prepare.py --datasets DIS-VD HRSOD-TE ... [--models birefnet inspyrenet] [--max-side 1024]

Images, GT masks and the imported SOD maps (output/<model>/<name>/sod/) are resized so that the longest
side is --max-side (INTER_AREA; masks and maps stay 8-bit, so load_gt's > 128 threshold applies to the
area-averaged mask). Output:
    DATA_ROOT/<name>-1k/images/<id>.jpg (quality 95) and masks/<id>.png
    output/<model>/<name>-1k/sod/<id>.png
The "-1k" names are registered in sodsam.data.DATASETS, so run_sam.py / tune2.py / fuse_eval.py run
unchanged on them; scripts/hr_eval.py scores their maps against the full-resolution GT.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sodsam import data


def target_size(h, w, max_side):
    s = max_side / max(h, w)
    if s >= 1:
        return h, w
    return max(1, round(h * s)), max(1, round(w * s))


def shrink(a, hw):
    if a.shape[:2] == tuple(hw):
        return a
    return cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=data.HR_DATASETS)
    ap.add_argument("--models", nargs="*", default=[], help="imported SOD models whose maps are downsized too")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality of the downsized images")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids (smoke test)")
    args = ap.parse_args()

    for ds in args.datasets:
        if ds.endswith(data.HR_SUFFIX):
            raise ValueError(f"{ds}: give the full-resolution name")
        name = ds + data.HR_SUFFIX
        img_out, msk_out = data.image_dir(name), data.mask_dir(name)
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)
        ids = data.list_ids(ds)[:args.limit]
        map_dirs = {m: (data.run_dir(m, ds, "sod"), data.run_dir(m, name, "sod")) for m in args.models}
        n_maps = {m: 0 for m in args.models}
        for id_ in tqdm(ids, desc=f"{ds} -> {name}"):
            ip, mp = img_out / f"{id_}.jpg", msk_out / f"{id_}.png"
            img = data.load_image(ds, id_)
            hw = target_size(*img.shape[:2], args.max_side)
            if args.overwrite or not ip.exists():
                Image.fromarray(shrink(img, hw)).save(ip, quality=args.quality)
            if args.overwrite or not mp.exists():
                gt = np.array(Image.open(data.mask_dir(ds) / f"{id_}.png"))
                if gt.ndim == 3:
                    gt = gt[..., 0]
                gt = gt.astype(np.uint8) * 255 if gt.dtype == bool else gt
                if gt.dtype != np.uint8:   # 16-bit or similar: map to 0..255 by the > 128 convention
                    gt = (gt > 128).astype(np.uint8) * 255
                data.write_png(mp, shrink(gt, hw))
            for m, (src, dst) in map_dirs.items():
                sp, dp = src / f"{id_}.png", dst / f"{id_}.png"
                if not sp.exists():
                    continue
                if args.overwrite or not dp.exists():
                    data.write_png(dp, shrink(data.read_png(sp), hw))
                n_maps[m] += 1
        print(f"{name}: {len(data.list_ids(name))} ids in {img_out.parent}" +
              "".join(f", {m}: {n} maps" for m, n in n_maps.items()))


if __name__ == "__main__":
    main()
