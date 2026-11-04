"""Multi-answer evaluation: score SOD maps against every answer people gave, not one ground truth.

Two datasets record each subject's own answer:
  PASCAL-S  12 subjects; a subject's answer = the union of the segments (isoCell) they selected
            (data/<subject>_sel.mat); the reconstruction matches the released graded masks exactly.
  SOD       7 subjects, several sessions each (one per BSDS segmentation); a session's answer = the
            union of the objects chosen ("all") or the object ranked most salient ("top").
Support of an answer A = mean over subjects of the share of that subject's answers with IoU >= 0.9
to A (PASCAL-S: 0 or 1 per subject). At level q the admissible answers are the benchmark GT plus
every answer with support >= q; each image is scored against the admissible answer that maximizes
its per-image F^max, and the usual dataset metrics (F^max of the mean F curve, MAE, E, S) are
computed against those chosen answers. q = inf is the standard benchmark.

Human row (control): each subject's answer scored the same way, with supports computed from the
other subjects only (leave-one-subject-out), so it shows how far a person is from the GT and how
much of that gap the other people's answers explain.

  python scripts/multi_answer_eval.py --workers 2
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
import h5py
import numpy as np
import scipy.io as sio

from annotator_support import iou, map_path, sod_sessions
from sodsam import data, metrics

PASCAL = data.DATA_ROOT / "downloads" / "PASCAL-S_full" / "datasets" / "segments" / "pascal"
OUT = Path(__file__).resolve().parents[1] / "results" / "multi_answer"
LEVELS = [None, 0.5, 1 / 3, 0.25, 0.15, 0.0]   # None = GT only; 0.0 = any single answer (>0 support)
MODELS = {"PASCAL-S": ["samba", "selfreformer", "mdsam", "vscode", "vstpp"],
          "MSOD": ["samba", "selfreformer", "vscode", "vstpp"]}
SAME = 0.9


def _load_sel(p):
    try:
        return [np.atleast_1d(x).astype(int) for x in sio.loadmat(p, squeeze_me=True)["selObj"][:, 0]]
    except NotImplementedError:                                  # two subjects' files are v7.3
        with h5py.File(p, "r") as h:
            s = h["selObj"]
            return [np.atleast_1d(h[s[0, k]][()]).ravel().astype(int) for k in range(s.shape[1])]


def pascal_answers(id_):
    """[(subject, answer mask)], one answer per subject."""
    global _SEL
    if "_SEL" not in globals():
        _SEL = [_load_sel(p) for p in sorted((PASCAL / "data").glob("*_sel.mat"))]
    k = int(id_) - 1
    with h5py.File(PASCAL / "isoCell.mat", "r") as f:
        L = f[f["isoCell"][0, k]][()].T
    return [(j, np.isin(L, s[k][s[k] > 0])) for j, s in enumerate(_SEL)]


def support(cand, answers, exclude=None):
    """Mean over subjects (except `exclude`) of the share of their answers that match cand."""
    per = {}
    for sj, a in answers:
        if sj != exclude:
            per.setdefault(sj, []).append(iou(a, cand) >= SAME)
    return float(np.mean([np.mean(v) for v in per.values()])) if per else 0.0


def choose(pred, gt, cands):
    """Per level: the admissible answer (GT or a candidate with enough support) maximizing per-image F^max."""
    fg = float(metrics.f_curve(pred, gt).max())
    fc = [(float(metrics.f_curve(pred, a).max()) if a.any() else 0.0, s, a) for a, s in cands]
    out = {}
    for q in LEVELS:
        best, ans = fg, gt
        if q is not None:
            for f, s, a in fc:
                if s > q - 1e-9 and (s > 0) and f > best:
                    best, ans = f, a
        out[q] = ans
    return out


def score_levels(pred, gt, cands):
    """image_scores against the chosen answer at every level (computed once per distinct answer)."""
    chosen = choose(pred, gt, cands)
    cache, res = {}, {}
    for q, a in chosen.items():
        key = id(a)
        if key not in cache:
            cache[key] = metrics.image_scores(pred, a)
        res[q] = cache[key]
    return res


def _image(id_, ds, key):
    gt = data.load_gt(ds, id_)
    if not gt.any():
        return None
    if ds == "PASCAL-S":
        answers = pascal_answers(id_)
    else:
        answers = [(x["sj"], x["union" if key == "all" else key]) for x in sod_sessions(id_, gt.shape)]
    answers = [(sj, a) for sj, a in answers if a.shape == gt.shape]
    cands = [(a, support(a, answers)) for _, a in answers if a.any()]
    distinct = []
    for a, _ in cands:
        if all(iou(a, d) < SAME for d in distinct):
            distinct.append(a)
    rec = {"id": id_, "models": {}, "human": [], "n_distinct": len(distinct)}
    for model in MODELS[ds]:
        p = map_path(model, ds, id_)
        if not p.exists():
            continue
        m = data.read_png(p)
        if m.shape != gt.shape:
            m = cv2.resize(m, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        rec["models"][model] = score_levels(m, gt, cands)
    for sj, a in answers:                                         # leave-one-subject-out human
        others = [(b, support(b, answers, exclude=sj)) for s2, b in answers if s2 != sj and b.any()]
        rec["human"].append(score_levels(a.astype(np.uint8) * 255, gt, others))
    return rec


def level_name(q):
    return "GT" if q is None else ("any" if q == 0 else f">={q:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--datasets", nargs="+", default=["PASCAL-S", "MSOD:all", "MSOD:top"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    with Pool(args.workers) as pool:
        for spec in args.datasets:
            ds, key = (spec.split(":") + ["all"])[:2]
            recs = [r for r in pool.imap(partial(_image, ds=ds, key=key), data.list_ids(ds), chunksize=4) if r]
            res = {}
            rows = {m: [r["models"][m] for r in recs if m in r["models"]] for m in MODELS[ds]}
            rows["human (LOSO)"] = [h for r in recs for h in r["human"]]
            print(f"\n{spec}: {len(recs)} images, mean distinct answers per image "
                  f"{np.mean([r['n_distinct'] for r in recs]):.1f}")
            print(f"{'':16s}" + "".join(f"{level_name(q):>16s}" for q in LEVELS) + "   (F^max / MAE; error recovered)")
            for name, rs in rows.items():
                if not rs:
                    continue
                agg = {level_name(q): metrics.aggregate([x[q] for x in rs]) for q in LEVELS}
                base = agg["GT"]["Fmax"]
                for v in agg.values():
                    v["F_err_recovered"] = (v["Fmax"] - base) / max(1 - base, 1e-9)
                res[name] = agg
                print(f"{name:16s}" + "".join(f"   {v['Fmax']:.3f}/{v['MAE']:.3f}" + (f" {v['F_err_recovered']:+.0%}"
                      if k != "GT" else "    ") for k, v in agg.items()))
            for q in LEVELS:
                ranking = sorted((m for m in MODELS[ds] if m in res), key=lambda m: -res[m][level_name(q)]["Fmax"])
                print(f"   ranking {level_name(q):>6s}: " + " > ".join(ranking))
            out[spec] = res
    (OUT / "multi_answer.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
