"""Cut-in-the-loop prompting (cluster, GPU + PyMaxflow): alternate the graph cut and SAM.

  x_0 = graph cut of the SOD map and the zoom-prompt masks (the paper's model, --params json);
  for t = 1..T:  prompt_t = 255 * (s + x_{t-1}) / 2  (s = max-normalized SOD map; inside the cut
                 >= 128, outside <= 127), SAM with the zoom-in prompt on prompt_t (crop = its box);
                 x_t = graph cut of the SOD map and these masks, same parameters.
Writes the masks of every iteration as the prompt mask_loop<t> (run_sam.py format, with probabilities,
so the fusion can be retuned on them) and the cuts as <sod>/<ds>/cutloop/<id>_<t>.png (t = 0..T), and
prints F^max / MAE of every x_t per dataset (results/cutloop_<sod>.json).

  python scripts/cut_loop.py --sod-model samba --params results/tune2/samba_..._mask_zoom_linear_paper_binary_all.json
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
from sodsam.fusion2 import fuse2, prepare
from sodsam.sam_prompting import SamRunner


def cut(img, sod, masks, gt, model, th):
    x = prepare({"img": img, "sod": sod, "sam": list(masks), "gt": gt})
    return fuse2(x, *model, th) > 0


def scores_of(x, gt):
    """image_scores of a cut (bool or uint8), resized to the GT if the SOD map had another size."""
    p = x.astype(np.uint8) * 255 if x.dtype == bool else x
    if p.shape != gt.shape:
        p = cv2.resize(p, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return metrics.image_scores(p, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--params", required=True, help="tune2 result json (model and parameters)")
    ap.add_argument("--datasets", nargs="+", default=["MSOD", "MSRA-B1K"] + data.EVAL_DATASETS)
    ap.add_argument("--iters", type=int, default=2)
    args = ap.parse_args()
    res = json.loads(Path(args.params).read_text())
    model, th = tuple(res["model"]), res["params"]
    sam = SamRunner()
    out_json = Path(__file__).resolve().parents[1] / "results" / f"cutloop_{args.sod_model}.json"
    summary = json.loads(out_json.read_text()) if out_json.exists() else {}
    for ds in args.datasets:
        ids = data.list_ids(ds)
        dirs = [data.run_dir(args.sod_model, ds, f"sam_mask_loop{t}") for t in range(1, args.iters + 1)]
        cut_dir = data.run_dir(args.sod_model, ds, "cutloop")
        for d in dirs + [cut_dir]:
            d.mkdir(parents=True, exist_ok=True)
        scores = [json.loads((d / "scores.json").read_text()) if (d / "scores.json").exists() else {} for d in dirs]
        per = [[] for _ in range(args.iters + 1)]
        for n, id_ in enumerate(tqdm(ids, desc=f"cutloop/{ds}")):
            img, gt = data.load_image(ds, id_), data.load_gt(ds, id_)
            sod = data.read_png(data.run_dir(args.sod_model, ds, "sod") / f"{id_}.png")
            s = sod.astype(np.float64) / (sod.max() + 1e-13)
            if all(id_ in sc for sc in scores) and (cut_dir / f"{id_}_{args.iters}.png").exists():
                for t in range(args.iters + 1):
                    per[t].append(scores_of(data.read_png(cut_dir / f"{id_}_{t}.png"), gt))
                continue
            zoom = data.run_dir(args.sod_model, ds, "sam_mask_zoom")
            masks = [data.read_png(zoom / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
            x = cut(img, sod, masks, gt, model, th)
            data.write_png(cut_dir / f"{id_}_0.png", x.astype(np.uint8) * 255)
            per[0].append(scores_of(x, gt))
            for t in range(1, args.iters + 1):
                prompt = np.round(255 * (s + x) / 2).astype(np.uint8)
                masks, sc, probs = sam.from_zoom(img, prompt)
                for j, mk in enumerate(masks, 1):
                    data.write_png(dirs[t - 1] / f"{id_}_{j}.png", mk.astype(np.uint8) * 255)
                    data.write_png(dirs[t - 1] / f"{id_}_{j}_p.png", np.round(probs[j - 1] * 255))
                scores[t - 1][id_] = [float(v) for v in sc]
                x = cut(img, sod, masks, gt, model, th)
                data.write_png(cut_dir / f"{id_}_{t}.png", x.astype(np.uint8) * 255)
                per[t].append(scores_of(x, gt))
            if n % 200 == 199:
                for d, sc in zip(dirs, scores):
                    (d / "scores.json").write_text(json.dumps(sc))
        for d, sc in zip(dirs, scores):
            (d / "scores.json").write_text(json.dumps(sc))
        summary[ds] = {f"x{t}": metrics.aggregate(per[t]) for t in range(args.iters + 1)}
        for t in range(args.iters + 1):
            print(metrics.format_row(f"{ds} cutloop x{t}", summary[ds][f"x{t}"]), flush=True)
        out_json.parent.mkdir(exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
