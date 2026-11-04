"""Tune a model of sodsam/ideas.py (nested, scales, decision, ordinal, sod_levels) by the random search of tune2.py.

Objective: mean per-image F_beta of the binary output (nested: the better of the two answers per image,
i.e. the pair should bracket the ground truth) over the tuning images. The images are sharded over the
workers (each worker holds 1/workers of them and returns sums). Result: results/tune2/<sod>_<datasets>_
idea_<model>_<half>.json with the parameters and the full metrics per dataset (nested: inner, outer,
best-of-two F).

Multi-answer models (ordinal, sod_levels; K answers, --levels): the objective is scored against the
answers people gave (PASCAL-S: pascal_answers; MSOD: sod_sessions, "MSOD:all" = union of the chosen
objects, "MSOD:top" = the top object), loaded once per image and stored as packed bits. Admissible answers
= the benchmark GT plus every distinct answer with support > 0. Per image, recall = mean over admissible
answers of the best F among the K masks, precision = mean over the K masks of their best F among the
answers; objective = mean over images of (recall + precision) / 2. Datasets without recorded answers
reduce to the GT alone (best-of-K F). Result: results/tune2/<sod>_<datasets>_idea_<model>_K<K>_<half>.json.

  python scripts/tune_ideas.py --model nested --sod-model samba --datasets MSRA-B1K --workers 2
  python scripts/tune_ideas.py --model scales --sod-model samba --datasets MSOD MSRA-B1K
  python scripts/tune_ideas.py --model decision --sod-model samba --gc-params results/tune2/samba_..._mask_zoom_linear_paper_binary_all.json
  python scripts/tune_ideas.py --model ordinal --levels 3 --sod-model samba --datasets MSOD --workers 16
  python scripts/tune_ideas.py --model sod_levels --levels 3 --sod-model samba --datasets PASCAL-S --workers 16
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from sodsam import data, ideas, metrics, pipeline
from sodsam.fusion2 import prepare

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tune2"
ANSWERS = ("PASCAL-S", "MSOD")          # datasets with per-subject answers


def split_spec(spec):
    """"MSOD:top" -> ("MSOD", "top"); a plain dataset name gets key "all"."""
    ds, key = (spec.split(":") + ["all"])[:2]
    return ds, key


def load_answers(ds, key, id_, gt):
    """[(answer mask, support)] of the people's answers of one image (nested_eval._image); [] without data."""
    if ds not in ANSWERS:
        return []
    from annotator_support import sod_sessions
    from multi_answer_eval import pascal_answers, support
    if ds == "PASCAL-S":
        answers = pascal_answers(id_)
    else:
        answers = [(x["sj"], x["union" if key == "all" else key]) for x in sod_sessions(id_, gt.shape)]
    answers = [(sj, a) for sj, a in answers if a.shape == gt.shape]
    return [(a, support(a, answers)) for _, a in answers if a.any()]


def pack_answers(gt, cands):
    """Distinct answers as packed bits: the GT (support inf) and the answers with support > 0."""
    bits, sup, seen = [], [], set()
    for a, s in [(gt, np.inf)] + list(cands):
        p = np.packbits(a.ravel())
        if s > 0 and p.tobytes() not in seen:
            seen.add(p.tobytes())
            bits.append(p)
            sup.append(s)
    return {"bits": np.stack(bits), "support": np.array(sup), "size": int(gt.size), "n_gt": int(gt.sum())}


def f_matrix(masks, ans, q=0.0):
    """F_beta of every mask (rows) against every admissible answer at support level q (None = GT only)."""
    keep = np.isinf(ans["support"]) if q is None else ans["support"] > q - 1e-9
    bits = ans["bits"][keep]
    M = np.stack([np.packbits(m.ravel()) for m in masks])
    tp = np.bitwise_count(M[:, None, :] & bits[None]).sum(-1, dtype=np.float64)
    nm = np.bitwise_count(M).sum(-1, dtype=np.float64)
    na = np.bitwise_count(bits).sum(-1, dtype=np.float64)
    return np.where(tp > 0, 1.3 * tp / (0.3 * na[None] + nm[:, None]), 0.0)


def multi_scores(masks, ans, q=0.0):
    """(recall, precision, best F) of K masks against the admissible answers at level q."""
    F = f_matrix(masks, ans, q)
    return float(F.max(0).mean()), float(F.max(1).mean()), float(F.max())


def load_record(model, sod, spec, id_, scores, prompt, layers, gc_res):
    """The per-image record a model needs (small: no image, no tree); None for an empty GT (multi-answer models)."""
    ds, key = split_spec(spec)
    inp = pipeline.load_inputs(sod, ds, prompt, id_, scores)
    x = prepare(inp)
    if model == "decision":
        return ideas.decision_prepare(x, ideas.gc_mask(x, gc_res), inp["sam"], scores[id_])
    if model in ideas.MULTI:
        if not x["gt"].any():
            return None
        rec = {"gt": x["gt"], "answers": pack_answers(x["gt"], load_answers(ds, key, id_, x["gt"]))}
        if model == "sod_levels":
            rec["s8"] = np.round(x["s"] * 255).astype(np.uint8)
        else:
            rec.update({k: x[k] for k in ("L", "n", "k", "sqdiff")})
        return rec
    rec = {k: x[k] for k in ("L", "n", "k", "sqdiff", "gt")}
    if model == "scales":
        rec["ns"] = []
        for p in layers:
            d = data.run_dir(sod, ds, f"sam_{p}")
            rec["ns"].append(sum((data.read_png(d / f"{id_}_{j}.png") > 0).astype(np.float64) for j in (1, 2, 3)))
    return rec


def apply(model, rec, th, K=3):
    """nested -> (inner, outer); ordinal, sod_levels -> K masks, outermost first; scales, decision -> mask."""
    if model == "nested":
        return ideas.nested_cut(rec, th)
    if model == "ordinal":
        return ideas.ordinal_cut(rec, th, K)
    if model == "sod_levels":
        return ideas.sod_levels(rec, th, K)
    if model == "scales":
        return ideas.scales_cut(rec, th)[0]
    return ideas.decision(rec, th)


def objective(model, rec, th, K=3):
    out = apply(model, rec, th, K)
    if model in ideas.MULTI:
        r, p, _ = multi_scores(out, rec["answers"])
        return (r + p) / 2
    if model == "nested":
        return max(ideas.binary_f(out[0], rec["gt"]), ideas.binary_f(out[1], rec["gt"]))
    return ideas.binary_f(out, rec["gt"])


def full_scores(model, rec, th, K=3):
    out = apply(model, rec, th, K)
    u8 = lambda m: m.astype(np.uint8) * 255
    if model in ideas.MULTI:
        r, p, _ = multi_scores(out, rec["answers"])
        fg = [ideas.binary_f(m, rec["gt"]) for m in out]
        return {"recall": r, "precision": p, "objective": (r + p) / 2, "gt_best": max(fg), "gt_levels": fg,
                "n_answers": int(len(rec["answers"]["support"]))}
    if model == "nested":
        return {"inner": metrics.image_scores(u8(out[0]), rec["gt"]), "outer": metrics.image_scores(u8(out[1]), rec["gt"]),
                "best": max(ideas.binary_f(out[0], rec["gt"]), ideas.binary_f(out[1], rec["gt"]))}
    return {"mask": metrics.image_scores(u8(out), rec["gt"])}


def aggregate(model, per):
    if model in ideas.MULTI:
        if not per:
            return {}
        out = {k: float(np.mean([p[k] for p in per])) for k in ("objective", "recall", "precision", "gt_best", "n_answers")}
        out["gt_levels"] = [float(v) for v in np.mean([p["gt_levels"] for p in per], axis=0)]
        out["n"] = len(per)
        return out
    if model == "nested":
        return {"inner": metrics.aggregate([p["inner"] for p in per]), "outer": metrics.aggregate([p["outer"] for p in per]),
                "best_of_two_F": float(np.mean([p["best"] for p in per]))}
    return metrics.aggregate([p["mask"] for p in per])


def _worker(conn, model, sod, prompt, layers, gc_path, K, keys):
    gc_res = json.loads(Path(gc_path).read_text()) if gc_path else None
    X, cache = {}, {}
    for spec, id_ in keys:
        ds = split_spec(spec)[0]
        if ds not in cache:
            cache[ds] = pipeline.load_scores(sod, ds, prompt)
        rec = load_record(model, sod, spec, id_, cache[ds], prompt, layers, gc_res)
        if rec is not None:
            X[spec, id_] = rec
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind, ths, ids = msg
        ids = [i for i in ids if i in X]
        if kind == "obj":
            conn.send([(sum(objective(model, X[i], th, K) for i in ids), len(ids)) for th in ths])
        else:
            conn.send([(i, full_scores(model, X[i], ths[0], K)) for i in ids])


class Shards:
    def __init__(self, n, worker_args, keys):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, *worker_args, keys[w::n]), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            if c.recv() != "ready":
                raise RuntimeError("worker failed to start")

    def objective(self, ths, ids):
        for c in self.conns:
            c.send(("obj", ths, ids))
        res = [c.recv() for c in self.conns]
        return [sum(r[j][0] for r in res) / max(sum(r[j][1] for r in res), 1) for j in range(len(ths))]

    def full(self, th, ids):
        for c in self.conns:
            c.send(("full", [th], ids))
        return dict(kv for c in self.conns for kv in c.recv())

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ideas.RANGES))
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_zoom", help="base SAM prompt (nested, decision; the 25% layer of scales)")
    ap.add_argument("--layers", nargs="+", default=ideas.SCALE_PROMPTS)
    ap.add_argument("--gc-params", default=None, help="tune2 json of the paper's graph cut (decision)")
    ap.add_argument("--datasets", nargs="+", default=["MSOD", "MSRA-B1K"],
                    help="tuning sets; MSOD:all / MSOD:top select the people's answers (ordinal, sod_levels)")
    ap.add_argument("--levels", type=int, default=3, help="K answers (ordinal, sod_levels)")
    ap.add_argument("--tune-half", default="all", choices=["A", "B", "all"])
    ap.add_argument("--limit", type=int, default=0, help="first N ids per dataset (smoke tests)")
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-refine", type=int, default=80)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.model == "decision" and not args.gc_params:
        ap.error("--model decision needs --gc-params")

    K = args.levels if args.model in ideas.MULTI else None
    ranges = ideas.param_ranges(args.model, args.levels)
    keys = list(ranges)
    lo0 = np.array([ranges[k][0] for k in keys], float)
    hi0 = np.array([ranges[k][1] for k in keys], float)
    lo, hi = lo0.copy(), hi0.copy()

    def ids_of(spec, half):
        ids = data.list_ids(split_spec(spec)[0])
        ids = ids[:args.limit] if args.limit else ids
        return {"A": ids[0::2], "B": ids[1::2], "all": ids}[half]
    other = {"A": "B", "B": "A", "all": "all"}[args.tune_half]
    tune_ids = [(ds, i) for ds in args.datasets for i in ids_of(ds, args.tune_half)]
    test_ids = [(ds, i) for ds in args.datasets for i in ids_of(ds, other)]
    all_keys = sorted(set(tune_ids) | set(test_ids))
    t0 = time.time()
    sh = Shards(args.workers, (args.model, args.sod_model, args.prompt, args.layers, args.gc_params, args.levels), all_keys)
    print(f"{len(all_keys)} images loaded in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(args.seed)
    best_v, best_f = None, -1.0
    try:
        for rnd in range(args.rounds + 1):
            n = args.n_random if rnd == 0 else args.n_refine
            cand = lo + (hi - lo) * rng.random((n, len(keys)))
            if best_v is not None:
                cand[0] = best_v
            for v, f in zip(cand, sh.objective([dict(zip(keys, v)) for v in cand], tune_ids)):
                if f > best_f:
                    best_v, best_f = v, f
            width = (hi - lo) * args.shrink
            lo, hi = np.maximum(best_v - width / 2, lo0), np.minimum(best_v + width / 2, hi0)
            print(f"round {rnd}: best objective {best_f:.4f} ({time.time() - t0:.0f} s)", flush=True)
        th = dict(zip(keys, map(float, best_v)))
        per = sh.full(th, test_ids)                 # no entry for images without a record (empty GT, MULTI)
        test_per_ds = {ds: aggregate(args.model, [per[k] for k in test_ids if k[0] == ds and k in per])
                       for ds in args.datasets}
        test_m = aggregate(args.model, [per[k] for k in test_ids if k in per])
    finally:
        sh.close()
    kname = f"_K{K}" if K else ""
    name = f"{args.sod_model}_{'+'.join(args.datasets).replace(':', '-')}_idea_{args.model}{kname}_{args.tune_half}{args.tag}"
    out = {"model": args.model, "K": K, "prompt": args.prompt, "layers": args.layers if args.model == "scales" else None,
           "gc_params": args.gc_params, "params": th, "tune_objective": best_f, "test": test_m,
           "test_per_dataset": test_per_ds, "args": vars(args), "seconds": time.time() - t0}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    if args.model in ideas.MULTI:
        print(f"{name}: objective {best_f:.4f} | recall {test_m['recall']:.4f} precision {test_m['precision']:.4f} "
              f"best-of-K F vs GT {test_m['gt_best']:.4f} per level " + " ".join(f"{v:.3f}" for v in test_m["gt_levels"])
              + f" (n={test_m['n']}, {test_m['n_answers']:.1f} answers/image, {time.time() - t0:.0f}s)", flush=True)
    elif args.model == "nested":
        print(f"{name}: objective {best_f:.4f} | held-out inner F={test_m['inner']['Fmax']:.4f} "
              f"outer F={test_m['outer']['Fmax']:.4f} best-of-two F={test_m['best_of_two_F']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    else:
        print(f"{name}: objective {best_f:.4f} | held-out F={test_m['Fmax']:.4f} MAE={test_m['MAE']:.4f} "
              f"E={test_m['E']:.4f} S={test_m['S']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
