"""SAM 2.1 outputs for the post-processing ideas (cluster, GPU):

  auto        SAM's automatic ("everything") masks, independent of the SOD model: 32 x 32 point grid,
              multimask, default thresholds; masks with 0.2% .. 95% of the image area kept, stored at
              half resolution as packed bits in <OUT_ROOT>/sam_auto/<ds>/<id>.npz with keys
              masks (K x h2 x ceil(w2/8) uint8, np.packbits along the last axis), shape (h2, w2),
              full_shape (H, W), iou (predicted IoU), stability, area (full-resolution pixels), box (xyxy)
  zoom_m<M>   the zoom-in prompt (sam_prompting.from_zoom) with the crop margin M percent instead of 25:
              <sod>/<ds>/sam_mask_zoom_m<M>/ in the run_sam.py format (masks, probabilities, scores)
  feat        4-neighbour cosine distances of the SAM 2.1 image features, independent of the SOD model:
              the predictor's image_embed (256 x 64 x 64) and high_res_feats (32 x 256 x 256,
              64 x 128 x 128) of the whole image, bilinearly resized to 256 x 256, concatenated (352
              channels) and L2-normalized per pixel; d_right[y, x] = 1 - <f[y, x], f[y, x+1]>,
              d_down[y, x] = 1 - <f[y, x], f[y+1, x]> (last column / row 0), resized to the image size,
              clipped to [0, 1] and stored as uint8 (round(d * 255)) in
              <OUT_ROOT>/sam_feat/<ds>/<id>_r.png and <id>_d.png (fusion.py pairwise "feat" / "both")

  python scripts/run_sam_extra.py --prompt auto --datasets DUTS-TE
  python scripts/run_sam_extra.py --prompt zoom_m0 zoom_m100 --sod-model samba --datasets DUTS-TE
  python scripts/run_sam_extra.py --prompt feat --datasets MSOD MSRA-B1K DUTS-TE
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tqdm import tqdm

from sodsam import data
from sodsam.sam_prompting import SamRunner

AUTO_MIN, AUTO_MAX = 0.002, 0.95


def auto_generator(sam):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    return SAM2AutomaticMaskGenerator(sam.predictor.model, points_per_side=32, points_per_batch=64,
                                      pred_iou_thresh=0.8, stability_score_thresh=0.95, multimask_output=True)


def run_auto(gen, img, path):
    h, w = img.shape[:2]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        anns = gen.generate(img)
    anns = [a for a in anns if AUTO_MIN * h * w <= a["area"] <= AUTO_MAX * h * w]
    anns.sort(key=lambda a: -a["area"])
    half = [a["segmentation"][::2, ::2] for a in anns]
    h2, w2 = (h + 1) // 2, (w + 1) // 2
    masks = np.packbits(np.stack(half) if half else np.zeros((0, h2, w2), bool), axis=-1)
    box = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                    for a in anns], np.int32).reshape(-1, 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, masks=masks, shape=np.array([h2, w2]), full_shape=np.array([h, w]),
                        iou=np.array([a["predicted_iou"] for a in anns], np.float32),
                        stability=np.array([a["stability_score"] for a in anns], np.float32),
                        area=np.array([a["area"] for a in anns], np.int64), box=box)
    return len(anns)


FEAT_SIZE = 256


def feat_distances(predictor, hw):
    """(d_right, d_down) uint8 H x W of the features the predictor holds after set_image."""
    import torch.nn.functional as F
    feats = predictor._features
    maps = [feats["image_embed"]] + list(feats["high_res_feats"])     # (1, C, h, w) each
    with torch.inference_mode():
        f = torch.cat([F.interpolate(m.float(), size=(FEAT_SIZE, FEAT_SIZE), mode="bilinear",
                                     align_corners=False) for m in maps], dim=1)
        f = F.normalize(f, dim=1)
        d_r = torch.zeros(1, 1, FEAT_SIZE, FEAT_SIZE, device=f.device)
        d_d = torch.zeros_like(d_r)
        d_r[..., :, :-1] = 1 - (f[..., :, :-1] * f[..., :, 1:]).sum(1, keepdim=True)
        d_d[..., :-1, :] = 1 - (f[..., :-1, :] * f[..., 1:, :]).sum(1, keepdim=True)
        out = []
        for d in (d_r, d_d):
            d = F.interpolate(d, size=tuple(hw), mode="bilinear", align_corners=False).clamp(0, 1)
            out.append((d[0, 0] * 255).round().to(torch.uint8).cpu().numpy())
    return out


def run_feat(sam, img, path_r, path_d):
    sam.set_image(img)
    d_r, d_d = feat_distances(sam.predictor, img.shape[:2])
    data.write_png(path_r, d_r)
    data.write_png(path_d, d_d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", nargs="+", required=True)
    ap.add_argument("--sod-model", default=None)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    sam = SamRunner()
    gen = auto_generator(sam) if "auto" in args.prompt else None
    zooms = [p for p in args.prompt if p.startswith("zoom_m")]
    if zooms and not args.sod_model:
        ap.error("zoom prompts need --sod-model")
    for ds in args.datasets:
        ids = data.list_ids(ds)
        if gen is not None:
            out = data.OUT_ROOT / "sam_auto" / ds
            counts = []
            for id_ in tqdm(ids, desc=f"auto/{ds}"):
                p = out / f"{id_}.npz"
                if p.exists():
                    continue
                counts.append(run_auto(gen, data.load_image(ds, id_), p))
            if counts:
                print(f"auto/{ds}: {len(counts)} images, {np.mean(counts):.1f} masks per image", flush=True)
        if "feat" in args.prompt:
            out = data.OUT_ROOT / "sam_feat" / ds
            n = 0
            for id_ in tqdm(ids, desc=f"feat/{ds}"):
                p_r, p_d = out / f"{id_}_r.png", out / f"{id_}_d.png"
                if p_r.exists() and p_d.exists():
                    continue
                run_feat(sam, data.load_image(ds, id_), p_r, p_d)
                n += 1
            print(f"feat/{ds}: {n} images computed", flush=True)
        for z in zooms:
            margin = int(z[len("zoom_m"):]) / 100
            d = data.run_dir(args.sod_model, ds, f"sam_mask_{z}")
            d.mkdir(parents=True, exist_ok=True)
            f = d / "scores.json"
            scores = json.loads(f.read_text()) if f.exists() else {}
            for n, id_ in enumerate(tqdm(ids, desc=f"{z}/{ds}")):
                if id_ in scores:
                    continue
                img = data.load_image(ds, id_)
                sod = data.read_png(data.run_dir(args.sod_model, ds, "sod") / f"{id_}.png")
                masks, sc, probs = sam.from_zoom(img, sod, margin=margin)
                for j, mk in enumerate(masks, 1):
                    data.write_png(d / f"{id_}_{j}.png", mk.astype(np.uint8) * 255)
                    data.write_png(d / f"{id_}_{j}_p.png", np.round(probs[j - 1] * 255))
                scores[id_] = [float(s) for s in sc]
                if n % 500 == 499:
                    f.write_text(json.dumps(scores))
            f.write_text(json.dumps(scores))


if __name__ == "__main__":
    main()
