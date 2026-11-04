"""Multi-answer evaluation of the ordinal K-answer cut (tune_ideas.py --model ordinal), as nested_eval.py.

The K masks are computed on the fly from the tuned parameters (--tune-json; the protocol is cross-wise:
tune on MSOD, evaluate on PASCAL-S and vice versa). Per image and support level q, the admissible answers
are the benchmark GT and every subject answer with support >= q (nested_eval). Rows:
  ordinal K=k       best-of-K F: the best (mask, admissible answer) pair
  ordinal K=k [j]   mask j alone (1 = outermost)
  ordinal K=1       the single cut tuned with the same objective (--k1-json), sod_levels (--sod-levels-json)
  sod               the SOD map with all 256 thresholds (F^max per image, an upper bound of a soft map)
  graph cut         the paper's single cut (gc_zoom maps, or computed from a tune2 json with --gc-json)
and, for every row, recall and precision (tune_ideas.multi_scores) at the support levels GT, >= 0.25,
>= 0.15 and any (> 0). Output: results/multi_answer/ordinal_<sod>_<tuneset>.json; --save writes the masks to
output/<sod>/<ds>/ordinal_<tuneset>/<id>_<j>.png.

  python scripts/ordinal_eval.py --sod-model samba --tune-json results/tune2/samba_MSOD_idea_ordinal_K3_all.json \
      --k1-json results/tune2/samba_MSOD_idea_ordinal_K1_all.json --sod-levels-json results/tune2/samba_MSOD_idea_sod_levels_K3_all.json \
      --datasets PASCAL-S MSOD:all MSOD:top --workers 8
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np

from multi_answer_eval import LEVELS, level_name
from nested_eval import _best
from sodsam import data, ideas, metrics, pipeline
from sodsam.fusion2 import prepare
from tune_ideas import apply, f_matrix, load_answers, pack_answers, split_spec

RP_LEVELS = [None, 0.25, 0.15, 0.0]
_SCORES = {}


def tuneset(res):
    return "+".join(res["args"]["datasets"]).replace(":", "-")


def _rows_of(masks, ans, cands, gt, name):
    """best-of-K and per-mask F at every level, recall/precision at RP_LEVELS."""
    best = {level_name(q): float(f_matrix(masks, ans, q).max()) for q in LEVELS}
    rows = {name: best}
    if len(masks) > 1:
        for j, m in enumerate(masks, 1):
            rows[f"{name} [{j}]"] = {level_name(q): float(f_matrix([m], ans, q).max()) for q in LEVELS}
    rp = {}
    for q in RP_LEVELS:
        F = f_matrix(masks, ans, q)
        rp[level_name(q)] = [float(F.max(0).mean()), float(F.max(1).mean())]
    return rows, {name: rp}


def _image(id_, ds, key, sod, models, gc_res, save):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    cands = load_answers(ds, key, id_, gt)
    ans = pack_answers(gt, cands)
    if ds not in _SCORES:
        _SCORES[ds] = pipeline.load_scores(sod, ds, "mask_zoom")
    x = prepare(pipeline.load_inputs(sod, ds, "mask_zoom", id_, _SCORES[ds]))
    x["s8"] = np.round(x["s"] * 255).astype(np.uint8)

    def rd(p):
        m = data.read_png(p)
        if m.shape != gt.shape:
            m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m
    s = rd(data.run_dir(sod, ds, "sod") / f"{id_}.png")
    rows = {"sod": {level_name(q): _best(lambda a: float(metrics.f_curve(s, a).max()) if a.any() else 0.0, cands, gt, q)
                    for q in LEVELS}}
    rp = {}
    for name, (model, th, K, tag) in models.items():
        masks = apply(model, x, th, K)
        r, p = _rows_of(masks, ans, cands, gt, name)
        rows.update(r)
        rp.update(p)
        if save and model == "ordinal":
            for j, m in enumerate(masks, 1):
                data.write_png(data.run_dir(sod, ds, f"ordinal_{tag}") / f"{id_}_{j}.png", m.astype(np.uint8) * 255)
    gc = None
    if gc_res is not None:
        gc = ideas.gc_mask(x, gc_res)
    elif (data.run_dir(sod, ds, "gc_zoom") / f"{id_}.png").exists():
        gc = rd(data.run_dir(sod, ds, "gc_zoom") / f"{id_}.png") > 0
    if gc is not None:
        r, p = _rows_of([gc], ans, cands, gt, "graph cut")
        rows.update(r)
        rp.update(p)
    return {"best": rows, "rp": rp}


def _load(path, name):
    res = json.loads(Path(path).read_text())
    return name, (res["model"], res["params"], res.get("K") or 3, tuneset(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--tune-json", required=True, help="tune_ideas json of the ordinal cut (K from the json)")
    ap.add_argument("--k1-json", default=None, help="tune_ideas json of the K=1 ordinal cut")
    ap.add_argument("--sod-levels-json", default=None, help="tune_ideas json of the sod_levels baseline")
    ap.add_argument("--gc-json", default=None, help="tune2 json of the paper's cut (computed on the fly instead of gc_zoom maps)")
    ap.add_argument("--datasets", nargs="+", default=["PASCAL-S", "MSOD:all", "MSOD:top"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    models = dict([_load(args.tune_json, None)])
    K = models[None][2]
    models = {f"ordinal K={K}": models[None]}
    if args.k1_json:
        models.update([_load(args.k1_json, "ordinal K=1")])
    if args.sod_levels_json:
        name, m = _load(args.sod_levels_json, None)
        models[f"sod_levels K={m[2]}"] = m
    gc_res = json.loads(Path(args.gc_json).read_text()) if args.gc_json else None
    tag = models[f"ordinal K={K}"][3]
    out = {"sod_model": args.sod_model, "tune_json": args.tune_json, "k1_json": args.k1_json,
           "sod_levels_json": args.sod_levels_json, "gc_json": args.gc_json, "datasets": {}}
    with Pool(args.workers) as pool:
        for spec in args.datasets:
            ds, key = split_spec(spec)
            ids = data.list_ids(ds)
            ids = ids[:args.limit] if args.limit else ids
            fn = partial(_image, ds=ds, key=key, sod=args.sod_model, models=models, gc_res=gc_res, save=args.save)
            recs = [r for r in pool.imap(fn, ids, chunksize=4) if r]
            names = [n for n in recs[0]["best"] if all(n in r["best"] for r in recs)]
            best = {n: {lv: float(np.mean([r["best"][n][lv] for r in recs])) for lv in recs[0]["best"][n]} for n in names}
            for n in names:
                base = best[n]["GT"]
                best[n] = {lv: {"F": v, "err_recovered": (v - base) / max(1 - base, 1e-9)} for lv, v in best[n].items()}
            rp_names = [n for n in recs[0]["rp"] if all(n in r["rp"] for r in recs)]
            rp = {n: {lv: {"recall": float(np.mean([r["rp"][n][lv][0] for r in recs])),
                           "precision": float(np.mean([r["rp"][n][lv][1] for r in recs]))}
                      for lv in recs[0]["rp"][n]} for n in rp_names}
            print(f"\n{spec} ({args.sod_model}, parameters tuned on {tag}): {len(recs)} images   best-of-K F / error recovered")
            print(f"{'':18s}" + "".join(f"{level_name(q):>14s}" for q in LEVELS))
            for n in names:
                print(f"{n:18s}" + "".join(f"   {best[n][level_name(q)]['F']:.3f}" + (f" {best[n][level_name(q)]['err_recovered']:+.0%}"
                      if q is not None else "     ") for q in LEVELS))
            print(f"{'recall / precision':18s}" + "".join(f"{level_name(q):>16s}" for q in RP_LEVELS))
            for n in rp_names:
                print(f"{n:18s}" + "".join(f"     {rp[n][level_name(q)]['recall']:.3f}/{rp[n][level_name(q)]['precision']:.3f}"
                                            for q in RP_LEVELS))
            out["datasets"][spec] = {"n": len(recs), "best": best, "recall_precision": rp}
    p = Path(__file__).resolve().parents[1] / "results" / "multi_answer" / f"ordinal_{args.sod_model}_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwritten {p}")


if __name__ == "__main__":
    main()
