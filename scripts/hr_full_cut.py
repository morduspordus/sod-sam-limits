"""The paper's graph cut at FULL resolution on the high-resolution benchmarks, restricted to the band
around the SOD boundary (sodsam.fusion_band.fuse_full), with the global 1K SAM masks or with the
native-resolution crops of scripts/run_sam_crops.py stitched by the cut.

  python scripts/hr_full_cut.py --sod-model isnet --datasets DIS-TE1 DIS-TE2 DIS-TE3 DIS-TE4 \
      --params results/tune2/isnet_DIS-VD-1k_mask_zoom_linear_paper_binary_all.json --workers 8

Rows (output/<model>/<ds>/<row>/<id>.png, 0/255, image size; scored by hr_eval.py --dataset <ds> --sub <row>):
    gc_full_global   the band cut with the counts of the 1K masks (output/<model>/<ds>-1k/sam_mask_zoom,
                     upsampled nearest) only
    gc_full_crops    the band cut with the crops' counts where covered, the global counts elsewhere
    crops_majority   no cut: inside the covered area object iff the combined count >= 1.5, outside it
                     the thresholded SOD map (the naive stitching baseline)
The parameters are the linear/paper ones of a tune2.py result json. Existing maps are skipped unless
--overwrite. Every worker holds one image's graph at a time; the band fraction, the seconds per row and
the worker's peak RSS are printed for the first --report images (to size the cluster job).
"""
import argparse
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import crops, data
from sodsam.crops import COUNT_SCALE, SUB as CROPS_SUB
from sodsam.fusion import sod_binary
from sodsam.fusion_band import fuse_full, load_linear_params

ROWS = ("gc_full_global", "gc_full_crops", "crops_majority")
PROMPT_1K = "mask_zoom"


def peak_rss_gb():
    """Peak resident set size of this process in GB (Linux: ru_maxrss in KB; Windows: PeakWorkingSetSize)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        fn = ctypes.windll.psapi.GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        if not fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return float("nan")
        return c.PeakWorkingSetSize / 2 ** 30
    except Exception:
        return float("nan")


def load_counts(sod_model, ds, id_, shape, with_crops):
    """Global (1K masks upsampled) and, with_crops, the combined counts and the covered map."""
    sam_dir = data.run_dir(sod_model, ds + data.HR_SUFFIX, f"sam_{PROMPT_1K}")
    masks_1k = [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
    global_count = crops.global_counts(masks_1k, shape)
    if not with_crops:
        return global_count, None, None
    crop_dir = data.run_dir(sod_model, ds, CROPS_SUB)
    p = crop_dir / f"{id_}_n.png"
    if not p.exists():
        raise FileNotFoundError(f"crop counts missing: {p} (scripts/run_sam_crops.py)")
    count = data.read_png(p).astype(np.float32) / COUNT_SCALE
    covered = data.read_png(crop_dir / f"{id_}_cov.png") > 0
    return global_count, crops.combined_counts(count, covered, global_count), covered


def _image(id_, sod_model, ds, params, rows, band_r, overwrite):
    todo = [r for r in rows if overwrite or not (data.run_dir(sod_model, ds, r) / f"{id_}.png").exists()]
    if not todo:
        return id_, None
    img = data.load_image(ds, id_)
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    if sod.shape != img.shape[:2]:
        raise ValueError(f"{ds}/{id_}: SOD map {sod.shape} is not the image size {img.shape[:2]}")
    r = band_r if band_r is not None else crops.band_radius(sod.shape)
    global_count, combined, covered = load_counts(sod_model, ds, id_, sod.shape,
                                                  any(t != "gc_full_global" for t in todo))
    info, times = {}, {}
    for row in todo:
        t0 = time.time()
        if row == "gc_full_global":
            x = fuse_full(img, sod, global_count, params, r, info=info)
        elif row == "gc_full_crops":
            x = fuse_full(img, sod, combined, params, r, info=info)
        else:
            x = np.where(covered, combined >= 1.5, sod_binary(sod))
        data.write_png(data.run_dir(sod_model, ds, row) / f"{id_}.png", x.astype(np.uint8) * 255)
        times[row] = time.time() - t0
    return id_, {"shape": sod.shape, "band_r": r, "times": times, "rss_gb": peak_rss_gb(),
                 "covered_frac": float(covered.mean()) if covered is not None else None, **info}


def report(id_, s):
    h, w = s["shape"]
    line = f"  {id_[:40]:<40s} {w}x{h} ({h * w / 1e6:.1f} Mpx) band_r={s['band_r']:.0f}"
    if "band_frac" in s:
        line += f" band={100 * s['band_frac']:.1f}% ({s['n_free'] / 1e6:.2f} Mpx)"
        if s["band_frac"] > 0.6:
            line += " [>60% of the pixels, solved exactly anyway]"
    if s["covered_frac"] is not None:
        line += f" crops_cover={100 * s['covered_frac']:.0f}%"
    line += "".join(f" {r}={t:.1f}s" for r, t in s["times"].items())
    return line + f" peak_rss={s['rss_gb']:.2f}GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True, choices=["birefnet", "isnet"])
    ap.add_argument("--datasets", nargs="+", default=["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"],
                    help="full-resolution names")
    ap.add_argument("--params", required=True, help="tune2.py result json of the linear/paper model")
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=ROWS)
    ap.add_argument("--band-r", type=float, default=None, help="band radius in pixels (default crops.band_radius)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", type=int, default=5, help="print band size, seconds and peak RSS of the first N images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    params = load_linear_params(args.params)
    print(f"params {args.params}: " + ", ".join(f"{k}={v:.3g}" for k, v in params.items()), flush=True)
    for ds in args.datasets:
        if ds.endswith(data.HR_SUFFIX):
            raise ValueError(f"{ds}: give the full-resolution name")
        ids = data.list_ids(ds)[:args.limit]
        f = partial(_image, sod_model=args.sod_model, ds=ds, params=params, rows=args.rows,
                    band_r=args.band_r, overwrite=args.overwrite)
        t0, done, stats = time.time(), 0, []
        # one image per task, ordered, so the first images report first; workers are recycled to return
        # the memory of the largest graphs to the system
        if args.workers > 1:
            pool = Pool(args.workers, maxtasksperchild=8)
            results = pool.imap(f, ids, chunksize=1)
        else:
            pool, results = None, map(f, ids)
        for n, (id_, s) in enumerate(results, 1):
            if s is None:
                continue
            done += 1
            stats.append(s)
            if done <= args.report:
                print(report(id_, s), flush=True)
            elif done % 50 == 0:
                print(f"  {ds}: {done} images, {(time.time() - t0) / n:.1f} s per image so far", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
        if stats:
            per_row = {r: np.mean([s["times"][r] for s in stats if r in s["times"]]) for r in args.rows}
            band = [s["band_frac"] for s in stats if "band_frac" in s]
            print(f"{args.sod_model} {ds}: {done} new of {len(ids)} ids in {time.time() - t0:.0f} s; "
                  + ", ".join(f"{r} {t:.1f} s" for r, t in per_row.items() if not np.isnan(t))
                  + (f"; band mean {100 * np.mean(band):.1f}% max {100 * max(band):.1f}%" if band else "")
                  + f"; peak RSS max {max(s['rss_gb'] for s in stats):.2f} GB", flush=True)
        else:
            print(f"{args.sod_model} {ds}: all {len(ids)} ids exist", flush=True)


if __name__ == "__main__":
    main()
