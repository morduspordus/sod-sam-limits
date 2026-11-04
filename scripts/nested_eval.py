"""Multi-answer evaluation of the nested two-answer cut (eval_ideas.py maps), as multi_answer_eval.py.

Per image and support level q, the admissible answers are the benchmark GT and every subject answer with
support >= q. A single mask scores the best admissible answer; the pair (inner, outer) scores the best
(member, admissible answer) combination, i.e. it is right when either answer matches what some people said.
Rows: SOD map (F^max over thresholds), inner, outer, the pair, and the paper's graph cut when its maps
exist (gc_zoom). Binary masks are scored by their F_beta. Output: results/multi_answer/nested_<sod>.json.

  python scripts/nested_eval.py --sod-model samba --datasets PASCAL-S MSOD:all MSOD:top --workers 2
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

from annotator_support import iou, sod_sessions
from multi_answer_eval import LEVELS, SAME, level_name, pascal_answers, support
from sodsam import data, ideas, metrics


def _best(f_of, cands, gt, q):
    """max of f_of over the admissible answers at level q."""
    best = f_of(gt)
    if q is not None:
        for a, s in cands:
            if s > q - 1e-9 and s > 0:
                best = max(best, f_of(a))
    return best


def _image(id_, ds, key, sod):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    if ds == "PASCAL-S":
        answers = pascal_answers(id_)
    else:
        answers = [(x["sj"], x["union" if key == "all" else key]) for x in sod_sessions(id_, gt.shape)]
    answers = [(sj, a) for sj, a in answers if a.shape == gt.shape]
    cands = [(a, support(a, answers)) for _, a in answers if a.any()]

    def rd(p):
        m = data.read_png(p)
        if m.shape != gt.shape:
            m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m
    s = rd(data.run_dir(sod, ds, "sod") / f"{id_}.png")
    inner = rd(data.run_dir(sod, ds, "nested") / f"{id_}_in.png") > 0
    outer = rd(data.run_dir(sod, ds, "nested") / f"{id_}_out.png") > 0
    rows = {"sod": lambda a: float(metrics.f_curve(s, a).max()) if a.any() else 0.0,
            "inner": lambda a: ideas.binary_f(inner, a), "outer": lambda a: ideas.binary_f(outer, a),
            "pair": lambda a: max(ideas.binary_f(inner, a), ideas.binary_f(outer, a))}
    p = data.run_dir(sod, ds, "gc_zoom") / f"{id_}.png"
    if p.exists():
        gc = rd(p) > 0
        rows["graph cut"] = lambda a: ideas.binary_f(gc, a)
    return {name: {level_name(q): _best(f, cands, gt, q) for q in LEVELS} for name, f in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--datasets", nargs="+", default=["PASCAL-S", "MSOD:all", "MSOD:top"])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    out = {}
    with Pool(args.workers) as pool:
        for spec in args.datasets:
            ds, key = (spec.split(":") + ["all"])[:2]
            recs = [r for r in pool.imap(partial(_image, ds=ds, key=key, sod=args.sod_model), data.list_ids(ds), chunksize=4) if r]
            names = [n for n in recs[0] if all(n in r for r in recs)]
            res = {n: {lv: float(np.mean([r[n][lv] for r in recs])) for lv in recs[0][n]} for n in names}
            for n in names:
                base = res[n]["GT"]
                res[n] = {lv: {"F": v, "err_recovered": (v - base) / max(1 - base, 1e-9)} for lv, v in res[n].items()}
            print(f"\n{spec} ({args.sod_model}): {len(recs)} images   mean F / error recovered")
            print(f"{'':12s}" + "".join(f"{level_name(q):>14s}" for q in LEVELS))
            for n in names:
                print(f"{n:12s}" + "".join(f"   {res[n][level_name(q)]['F']:.3f}" + (f" {res[n][level_name(q)]['err_recovered']:+.0%}"
                      if q is not None else "     ") for q in LEVELS))
            out[spec] = res
    p = Path(__file__).resolve().parents[1] / "results" / "multi_answer" / f"nested_{args.sod_model}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
