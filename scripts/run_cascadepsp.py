"""CascadePSP baseline (Cheng et al., CVPR 2020): refine the thresholded SOD mask with the published model.

The coarse mask is the SOD map max-normalized and thresholded at 127/255, as for SAMRefiner
(run_samrefiner.py); an empty coarse mask keeps the SOD map. CascadePSP is used through its pip package
(segmentation-refinement), with the published weights and the settings of its README (fast=False, L=900),
and the image read with OpenCV (BGR), as in the README. Writes <OUT_ROOT>/<sod>/<ds>/cascadepsp/<id>.png and
results/eval_<sod>_cascadepsp.json (metrics on the datasets).

  python scripts/run_cascadepsp.py --sod-model samba --datasets DUTS-TE ECSSD DUT-OMRON PASCAL-S HKU-IS
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tqdm import tqdm

from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


def coarse_mask(sod):
    """The SOD map max-normalized and thresholded at 127/255, as uint8 0/255."""
    if sod.max() == 0:
        return np.zeros(sod.shape, np.uint8)
    return ((255.0 * sod / sod.max()) >= 127).astype(np.uint8) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--L", type=int, default=900)
    args = ap.parse_args()

    import segmentation_refinement as refine
    refiner = refine.Refiner(device="cuda:0")
    res = {"sod_model": args.sod_model, "settings": {"fast": False, "L": args.L}, "datasets": {}}
    for ds in args.datasets:
        out_dir = data.run_dir(args.sod_model, ds, "cascadepsp")
        out_dir.mkdir(parents=True, exist_ok=True)
        scores = []
        for id_ in tqdm(data.list_ids(ds)[:args.limit], desc=ds):
            sod = data.read_png(data.run_dir(args.sod_model, ds, "sod") / f"{id_}.png")
            mask = coarse_mask(sod)
            if mask.any():
                img = cv2.imread(str(data.image_path(ds, id_)), cv2.IMREAD_COLOR)
                out = refiner.refine(img, mask, fast=False, L=args.L)
                out = np.asarray(out, dtype=np.uint8)
                if out.shape != sod.shape:
                    raise ValueError(f"{ds}/{id_}: CascadePSP output {out.shape} vs SOD map {sod.shape}")
            else:
                out = sod
            data.write_png(out_dir / f"{id_}.png", out)
            scores.append(metrics.image_scores(out, data.load_gt(ds, id_)))
        res["datasets"][ds] = {"cascadepsp": metrics.aggregate(scores)}
        print(metrics.format_row(f"{ds} cascadepsp", res["datasets"][ds]["cascadepsp"]), flush=True)
    RESULTS.mkdir(exist_ok=True)
    name = f"eval_{args.sod_model}_cascadepsp" + (f"_lim{args.limit}" if args.limit else "")
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
