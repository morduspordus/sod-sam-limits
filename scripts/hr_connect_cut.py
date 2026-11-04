"""The full-resolution band cut of hr_full_cut.py (gc_full_crops) with the connectivity prior of
sodsam.connect (geodesic star convexity, exact).

  python scripts/hr_connect_cut.py --sod-model isnet --datasets DIS-VD \
      --params results/tune2/isnet_DIS-VD-1k_mask_zoom_linear_paper_binary_all.json --workers 6   # stage A
  python scripts/hr_connect_cut.py ... --rows $(python scripts/hr_connect_cut.py --stage-b gc_star_interior_g0_hard)

Rows (output/<model>/<ds>/<row>/<id>.png, 0/255, image size; scored by hr_eval.py --dataset <ds> --sub <row>):
    gc_star_<seed>_g<gamma>_hard      hard constraint (beta = inf)
    gc_star_<seed>_g<gamma>_b<beta>   soft constraint
seed: interior | strict | loose (sodsam.connect.star_parents); gamma: geodesic edge cost 1 + gamma (1 - mean
SOD of the two pixels). Two-stage selection on DIS-VD (jobs/hr_cs_cpu.sbatch): stage A = the hard rows of
{interior, strict} x gamma {0, 4} (the default --rows); stage B = beta in {128, 512, 2048, 8192} for the
selected (seed, gamma) (--stage-b prints those row names). The counts are the combined ones of
gc_full_crops (the crops' mean counts where covered, the upsampled 1K counts elsewhere; sam_crops from
run_sam_crops.py or run_sam_stitch.py); with beta = 0 the cut is gc_full_crops exactly. The parents are
computed once per image and (seed, gamma) and shared by the rows. Existing maps are skipped unless
--overwrite (the job passes --overwrite: the rows depend on the seed rule, sodsam.connect.SEED_RULE_VERSION,
printed in the log). Per image: the band components by kind (interior / band_seeds / own_strict /
unconstrained, sodsam.connect.star_parents), unconstrained and unreachable pixels (unreachable: 0 by
construction), violations of the hard rows (0 in theory; counted, not asserted), seconds and peak RSS for
the first --report images (and every image with violations); per set: the component counts and the totals.
"""
import argparse
import re
import sys
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from hr_full_cut import load_counts, peak_rss_gb
from sodsam import crops, data
from sodsam.connect import (C_SEED, HARD, SEED_RULE_VERSION, T_SEED, minimize_band_star, prepare_band,
                            star_parents)
from sodsam.fusion_band import load_linear_params

STAGE_A_SEEDS = ("interior", "strict")
GAMMAS = (0, 4)
BETAS = (128, 512, 2048, 8192)
STAGE_A = tuple(f"gc_star_{s}_g{g}_hard" for s in STAGE_A_SEEDS for g in GAMMAS)
NUM = r"[0-9]+(?:\.[0-9]+)?"
ROW_RE = re.compile(rf"gc_star_(interior|strict|loose)_g({NUM})_(hard|b({NUM}))")


def parse_row(row):
    """(seed mode, gamma, beta) of a row name; beta = HARD for the hard rows."""
    m = ROW_RE.fullmatch(row)
    if not m:
        raise ValueError(f"bad row name {row!r} (gc_star_<seed>_g<gamma>_hard|b<beta>)")
    return m.group(1), float(m.group(2)), (HARD if m.group(3) == "hard" else float(m.group(4)))


def stage_b(row):
    """The beta-grid rows of the (seed, gamma) of a row."""
    parse_row(row)
    base = row[:row.rindex("_")]
    return [f"{base}_b{b}" for b in BETAS]


def _image(id_, sod_model, ds, params, rows, band_r, t_s, c_s, overwrite):
    todo = [r for r in rows if overwrite or not (data.run_dir(sod_model, ds, r) / f"{id_}.png").exists()]
    if not todo:
        return id_, None
    img = data.load_image(ds, id_)
    sod = data.read_png(data.run_dir(sod_model, ds, "sod") / f"{id_}.png")
    if sod.shape != img.shape[:2]:
        raise ValueError(f"{ds}/{id_}: SOD map {sod.shape} is not the image size {img.shape[:2]}")
    r = band_r if band_r is not None else crops.band_radius(sod.shape)
    _, combined, covered = load_counts(sod_model, ds, id_, sod.shape, True)
    t0 = time.time()
    prep = prepare_band(img, sod, combined, params, r)
    times = {"prepare": time.time() - t0}
    stats = {"n_free": prep["n_free"], "band_frac": prep["n_free"] / sod.size, "seeds": {}, "violations": {}}
    groups = {}
    for row in todo:
        seed, gamma, _ = parse_row(row)
        groups.setdefault((seed, gamma), []).append(row)
    for (seed, gamma), grows in groups.items():
        parent, key = None, f"{seed}_g{gamma:g}"
        if prep["box"] is not None:
            box = prep["box"]
            t0 = time.time()
            s = sod[box].astype(np.float64) / (float(sod.max()) + 1e-13)
            info = {}
            parent = star_parents(s, combined[box], prep["free"], prep["fixed"], gamma, t_s, c_s, info, seed)
            del s
            times[f"parents_{key}"] = time.time() - t0
            stats["seeds"][key] = {k: info[k] for k in ("seed_source", "components", "n_seeds", "n_seed_band",
                                                        "n_unreachable", "n_unconstrained_px")}
        else:
            stats["seeds"][key] = {"seed_source": "no_band"}
        for row in grows:
            beta = parse_row(row)[2]
            t0 = time.time()
            x = prep["sod_mask"].copy()
            if prep["box"] is not None:
                info = {}
                x[prep["box"]] = minimize_band_star(prep["u"], prep["tables"], prep["free"], prep["fixed"],
                                                    parent, beta, info)
                if "violations" in info:
                    stats["violations"][row] = info["violations"]
            data.write_png(data.run_dir(sod_model, ds, row) / f"{id_}.png", x.astype(np.uint8) * 255)
            times[row] = time.time() - t0
        del parent
    return id_, {"shape": sod.shape, "band_r": r, "times": times, "rss_gb": peak_rss_gb(),
                 "covered_frac": float(covered.mean()), **stats}


def report(id_, s):
    h, w = s["shape"]
    line = (f"  {id_[:40]:<40s} {w}x{h} ({h * w / 1e6:.1f} Mpx) band_r={s['band_r']:.0f}"
            f" band={100 * s['band_frac']:.1f}% ({s['n_free'] / 1e6:.2f} Mpx) crops_cover={100 * s['covered_frac']:.0f}%")
    line += f" seed_rule=v{SEED_RULE_VERSION}"
    for key, v in s["seeds"].items():
        line += f" [{key}: components {v['seed_source']}"
        if "n_seeds" in v:
            line += (f" seeds={v['n_seeds']} (band {v['n_seed_band']}) unconstrained_px={v['n_unconstrained_px']}"
                     f" unreachable={v['n_unreachable']}")
        line += "]"
    if s["violations"]:
        line += " violations " + ",".join(f"{r}={n}" for r, n in s["violations"].items())
    line += "".join(f" {r}={t:.1f}s" for r, t in s["times"].items())
    return line + f" peak_rss={s['rss_gb']:.2f}GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", choices=["birefnet", "isnet"])
    ap.add_argument("--datasets", nargs="+", default=["DIS-VD"], help="full-resolution names")
    ap.add_argument("--params", help="tune2.py result json of the linear/paper model")
    ap.add_argument("--rows", nargs="+", default=list(STAGE_A), help="gc_star_<seed>_g<gamma>_hard|b<beta>")
    ap.add_argument("--stage-b", default=None, metavar="ROW",
                    help="only print the beta-grid rows of the (seed, gamma) of ROW and exit")
    ap.add_argument("--t-seed", type=float, default=T_SEED, help="loose seeds: SOD >= t_seed * max")
    ap.add_argument("--c-seed", type=float, default=C_SEED, help="loose seeds: mean SAM count >= c_seed")
    ap.add_argument("--band-r", type=float, default=None, help="band radius in pixels (default crops.band_radius)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", type=int, default=5, help="print sizes, seconds and peak RSS of the first N images")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.stage_b:
        print(" ".join(stage_b(args.stage_b)))
        return
    if not args.sod_model or not args.params:
        ap.error("--sod-model and --params are required")
    for row in args.rows:
        parse_row(row)

    params = load_linear_params(args.params)
    print(f"params {args.params}: " + ", ".join(f"{k}={v:.3g}" for k, v in params.items())
          + f"; rows {' '.join(args.rows)}; seed rule v{SEED_RULE_VERSION}", flush=True)
    for ds in args.datasets:
        if ds.endswith(data.HR_SUFFIX):
            raise ValueError(f"{ds}: give the full-resolution name")
        ids = data.list_ids(ds)[:args.limit]
        f = partial(_image, sod_model=args.sod_model, ds=ds, params=params, rows=args.rows, band_r=args.band_r,
                    t_s=args.t_seed, c_s=args.c_seed, overwrite=args.overwrite)
        t0, done, stats = time.time(), 0, []
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
            if done <= args.report or any(s["violations"].values()):
                print(report(id_, s), flush=True)
            elif done % 50 == 0:
                print(f"  {ds}: {done} images, {(time.time() - t0) / n:.1f} s per image so far", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
        if stats:
            keys = sorted({k for s in stats for k in s["times"]})
            per = {k: np.mean([s["times"][k] for s in stats if k in s["times"]]) for k in keys}
            # per (seed, gamma): components of each kind summed over the images, and the number of images
            # with at least one component of the kind
            comp, imgs = Counter(), Counter()
            for s in stats:
                for k, v in s["seeds"].items():
                    for kind, n in v.get("components", {}).items():
                        comp[f"{k}:{kind}"] += n
                        imgs[f"{k}:{kind}"] += n > 0
            unreach = sum(v.get("n_unreachable", 0) for s in stats for v in s["seeds"].values())
            viol = sum(n for s in stats for n in s["violations"].values())
            n_viol = sum(any(s["violations"].values()) for s in stats)
            print(f"{args.sod_model} {ds}: {done} new of {len(ids)} ids in {time.time() - t0:.0f} s; "
                  + ", ".join(f"{k} {t:.1f} s" for k, t in per.items())
                  + f"; peak RSS max {max(s['rss_gb'] for s in stats):.2f} GB", flush=True)
            print(f"{args.sod_model} {ds}: band components (seed rule v{SEED_RULE_VERSION}) "
                  + ", ".join(f"{k}={comp[k]} in {imgs[k]} images" for k in sorted(comp) if comp[k])
                  + f"; unreachable {unreach} px; hard-constraint violations {viol} px in {n_viol} images",
                  flush=True)
        else:
            print(f"{args.sod_model} {ds}: all {len(ids)} ids exist", flush=True)


if __name__ == "__main__":
    main()
