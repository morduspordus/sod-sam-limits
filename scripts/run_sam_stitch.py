"""SAM2 on the native-resolution crops of run_sam_crops.py, stitched by three rules (sodsam.stitch):
uniform average, Gaussian-weighted average (nnU-Net blending) and the seam cut (exact graph cut where the
crops disagree, a grid of Potts parameters), all evaluated online so no per-crop map is stored.

  python scripts/run_sam_stitch.py --sod-model isnet --datasets DIS-VD DIS-TE1 DIS-TE2 DIS-TE3 DIS-TE4
  python scripts/run_sam_stitch.py --fake --sod-model isnet --datasets DIS-VD --limit 2   # CPU smoke test

Full-resolution dataset names. The crops are placed exactly as run_sam_crops.py (crops.boundary_band of
radius crops.band_radius, crops.place_crops with the same defaults), SAM is prompted the same way
(set_image on the crop, the soft SOD crop as the mask prompt), and the best-scoring of the three masks
(argmax score) gives each crop's probability map p_c. Output under output/<model>/<ds>/:
    stitch_avg/<id>.png, stitch_gauss/<id>.png, stitch_seam_l<lam>_s<sigma>/<id>.png   0/255, image size
                   (outside the covered area: the thresholded SOD map)
    meta_stitch/<id>.json   version (2) and prob_dtype (float32), crops, scores, best mask per crop, seam
                   size, seconds (SAM, stitching, cuts); a run that only writes missing counts keeps an
                   existing meta. The job passes --overwrite, so stitch maps of an older version are redone.
    sam_crops/<id>_n.png, <id>_cov.png, <id>.json   the count maps of run_sam_crops.py (same format), written
                   only when none of the three files exists yet, so one SAM pass serves the connectivity
                   cut too; existing sam_crops files are never overwritten (not even with --overwrite)
The crops' probability maps enter the stitcher as float32, accumulated online (no per-crop map is kept).
Ids whose stitch outputs all exist are skipped unless --overwrite. --fake uses run_sam_crops.FakeRunner.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from tqdm import tqdm

from run_sam_crops import FakeRunner
from sodsam import crops, data
from sodsam.crops import COUNT_SCALE, SUB as CROPS_SUB
from sodsam.stitch import LAMS, R_SEAM, SIGMAS, Stitcher, seam_name, stitch_all

META = "meta_stitch"
VERSION = 2   # meta_stitch "version": 2 = float32 probabilities into the stitcher (1: float16)


def variants(lams, sigmas):
    return ["avg", "gauss"] + [seam_name(l, s) for l in lams for s in sigmas]


def write_counts(out, id_, shape, boxes, masks, scores, r, args):
    """sam_crops/<id>_n.png, _cov.png, .json exactly as run_sam_crops.py."""
    count, covered = crops.stitch_counts(shape, boxes, masks)
    data.write_png(out / f"{id_}_n.png", np.round(COUNT_SCALE * count))
    data.write_png(out / f"{id_}_cov.png", covered.astype(np.uint8) * 255)
    (out / f"{id_}.json").write_text(json.dumps({"crops": [[int(v) for v in b] for b in boxes], "scores": scores,
                                                 "band_r": r, "shape": list(shape), "size": args.size,
                                                 "stride": args.stride, "fake": args.fake,
                                                 "writer": "run_sam_stitch"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True, choices=["birefnet", "isnet"])
    ap.add_argument("--datasets", nargs="+", default=["DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"],
                    help="full-resolution names")
    ap.add_argument("--size", type=int, default=crops.CROP_SIZE)
    ap.add_argument("--stride", type=int, default=crops.CROP_STRIDE)
    ap.add_argument("--min-frac", type=float, default=crops.MIN_FRAC)
    ap.add_argument("--max-crops", type=int, default=crops.MAX_CROPS)
    ap.add_argument("--r-seam", type=int, default=R_SEAM, help="dilation of the disagreement in pixels")
    ap.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    ap.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="redo the stitch outputs (never sam_crops)")
    ap.add_argument("--no-counts", action="store_true", help="do not write missing sam_crops count maps")
    ap.add_argument("--fake", action="store_true", help="FakeRunner instead of SAM (CPU, no torch)")
    args = ap.parse_args()

    names = variants(args.lams, args.sigmas)
    sam = None
    for ds in args.datasets:
        if ds.endswith(data.HR_SUFFIX):
            raise ValueError(f"{ds}: give the full-resolution name")
        sod_dir = data.run_dir(args.sod_model, ds, "sod")
        out_dirs = {k: data.run_dir(args.sod_model, ds, f"stitch_{k}") for k in names}
        meta_dir = data.run_dir(args.sod_model, ds, META)
        crop_dir = data.run_dir(args.sod_model, ds, CROPS_SUB)
        stats = []
        for id_ in tqdm(data.list_ids(ds)[:args.limit], desc=f"stitch/{ds}"):
            # never overwrite: any of the three files present -> the counts are not written
            need_counts = not args.no_counts and not any(
                (crop_dir / f"{id_}{suf}").exists() for suf in (".json", "_n.png", "_cov.png"))
            done = all((d / f"{id_}.png").exists() for d in out_dirs.values())
            if done and not args.overwrite and not need_counts:
                continue
            if sam is None:
                if args.fake:
                    sam = FakeRunner()
                else:
                    from sodsam.sam_prompting import SamRunner
                    sam = SamRunner()
            t0 = time.time()
            img = data.load_image(ds, id_)
            sod = data.read_png(sod_dir / f"{id_}.png")
            if sod.shape != img.shape[:2]:
                raise ValueError(f"{ds}/{id_}: SOD map {sod.shape} is not the image size {img.shape[:2]}")
            r = crops.band_radius(sod.shape)
            boxes = crops.place_crops(crops.boundary_band(sod, r), args.size, args.stride, args.min_frac,
                                      args.max_crops)
            st = Stitcher(sod.shape, boxes)
            masks, scores, best = [], [], []
            for y0, x0, y1, x1 in boxes:
                sam.set_image(np.ascontiguousarray(img[y0:y1, x0:x1]))
                mk, sc, pr = sam.from_mask(np.ascontiguousarray(sod[y0:y1, x0:x1]), "soft")
                k = int(np.argmax(sc))
                # float32: float16 would quantize the logits near p = 1 (9.21, 7.62, 6.93, ...)
                st.add((y0, x0, y1, x1), np.asarray(pr[k], np.float32))
                if need_counts:
                    masks.append(np.asarray(mk, bool))
                scores.append([float(s) for s in sc])
                best.append(k)
            t_sam = time.time() - t0
            if need_counts:
                write_counts(crop_dir, id_, sod.shape, boxes, masks, scores, r, args)
            del masks
            t1 = time.time()
            info = {}
            stitched = not done or args.overwrite
            if stitched:
                maps = stitch_all(img, sod, st, args.lams, args.sigmas, args.r_seam, info=info)
                for k, x in maps.items():
                    data.write_png(out_dirs[k] / f"{id_}.png", x.astype(np.uint8) * 255)
                del maps
            t_stitch = time.time() - t1
            rec = {"version": VERSION, "prob_dtype": "float32", "stitched": stitched,
                   "crops": [[int(v) for v in b] for b in boxes], "scores": scores, "best": best, "band_r": r,
                   "shape": list(sod.shape), "r_seam": args.r_seam, "sam_s": t_sam, "stitch_s": t_stitch,
                   "wrote_counts": need_counts, "fake": args.fake, **info}
            meta = meta_dir / f"{id_}.json"
            # only the counts were written: keep the existing meta (it holds the seam info of the stitching)
            if stitched or not meta.exists():
                meta_dir.mkdir(parents=True, exist_ok=True)
                meta.write_text(json.dumps(rec))
            stats.append(rec)
            if len(stats) <= 3:
                h, w = sod.shape
                print(f"  {id_[:40]:<40s} {w}x{h} crops={len(boxes)} sam={t_sam:.1f}s stitch={t_stitch:.1f}s"
                      f" seam={info.get('n_seam', 0)} px (disagree {info.get('n_disagree', 0)})", flush=True)
        if stats:
            print(f"{args.sod_model} stitch {ds}: {len(stats)} new, {np.mean([len(s['crops']) for s in stats]):.1f}"
                  f" crops per image, SAM {np.mean([s['sam_s'] for s in stats]):.1f} s, stitching "
                  f"{np.mean([s['stitch_s'] for s in stats]):.1f} s per image, seam mean "
                  f"{np.mean([s.get('n_seam', 0) for s in stats]):.0f} px, counts written for "
                  f"{sum(s['wrote_counts'] for s in stats)}" + (" [FAKE]" if args.fake else ""), flush=True)
        else:
            print(f"{args.sod_model} stitch {ds}: nothing to do", flush=True)


if __name__ == "__main__":
    main()
