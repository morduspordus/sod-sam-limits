"""Tune a fusion2 model on one half of the tuning set and evaluate it on the other half.

The halves are the even / odd positions of the sorted MSOD ids. Objective: F^max (toolbox) on the
tuning half. Random search, then rounds of finer random search around the best (as tune.py).

  python scripts/tune2.py --prompt mask_binary --unary linear --pairwise paper --output soft

--init <result json> ... adds their parameters as the first candidates (missing ones at their lower
bound), so the search cannot end below them on the tuning set. For --unary table_lin, table and linear
results are converted to the identical table_lin model (fusion2.as_table_lin).
--neg-prompt <prompt> also loads the masks of a complement prompt (for the +veto modifier); the
result file name then carries "<prompt>-neg-<neg prompt>".
--pairwise feat | both use the SAM feature distances of output/sam_feat/<ds>/ (run_sam_extra.py
--prompt feat; missing files are an error), tuning sigma_f; the workers hold them as float32.
--band adds the hard band constraint (fusion.clamp_band) with its width band_w tuned; the result
json carries "band": true and the file name "_band".
--bin-file results/bins_<sod>.json --bin k (scripts/bins.py) restricts the tuning and held-out images
to confidence bin k; the file name carries "_bin<k>" and args records bin_file / bin.
--unary logodds --lo-model table | gbt loads the learned log-odds maps of output/<sod>/<ds>/lo_<model>/
(scripts/learn_unary.py; missing files are an error); the file name carries "_lo<model>" and the result
json "lo_model".
--limit N uses only the first N ids of every dataset (smoke tests; the file name carries "_lim<N>").

Memory: every prepared image takes ~12 MB. By default each worker holds all images and evaluates
whole candidates; when that would exceed --mem-budget GB (or with --shard), each worker holds only a
1/workers shard of the images and returns sums of F curves per candidate (same candidates, same
selection, ~1/workers of the memory).
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, metrics, pipeline
from sodsam.fusion2 import as_table_lin, fuse2, param_ranges, prepare

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tune2"


def split_ids(ds, half, limit=None):
    ids = data.list_ids(ds)[:limit]
    return {"A": ids[0::2], "B": ids[1::2], "all": ids}[half]


def split_keys(datasets, half, limit=None):
    """(dataset, id) keys of a half, taken within each dataset (of the first `limit` ids)."""
    return [(ds, i) for ds in datasets for i in split_ids(ds, half, limit)]


MB_PER_IMAGE = 12


def _init(sod_model, datasets, prompt, model, neg_prompt=None, sod_model2=None, keys=None, band=False,
          lo_model=None):
    """Prepared images of the datasets (only those in keys, if given) into the global X; the feature
    distances are loaded for a feat / both pairwise, the log-odds maps of lo_model for a logodds unary."""
    global X, MODEL, BAND
    MODEL, BAND = model, band
    X = {}
    for ds in datasets:
        scores = None if sod_model2 else pipeline.load_scores(sod_model, ds, prompt)
        for i in data.list_ids(ds):
            if keys is None or (ds, i) in keys:
                inp = pipeline.load_inputs(sod_model, ds, prompt, i, scores, neg_prompt, sod_model2, lo_model)
                if model[1] in pipeline.FEAT_PAIRWISE:
                    inp["feat"] = pipeline.load_feat(ds, i)
                X[ds, i] = prepare(inp)


def _init_kw(kw):
    _init(**kw)


def _shard_worker(conn, init_args, keys):
    """Holds one shard of the images; answers ("curves", [th...], ids) with (sum of F curves, count)
    per th and ("scores", [th], ids) with (key, image_scores) pairs, over the ids it holds."""
    _init_kw({**init_args, "keys": set(keys)})
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind, ths, ids = msg
        ids = [i for i in ids if i in X]
        if kind == "curves":
            out = []
            for th in ths:
                cs = [metrics.f_curve(fuse2(X[i], *MODEL, th, BAND), X[i]["gt"]) for i in ids if X[i]["gt"].any()]
                out.append((np.sum(cs, axis=0) if cs else np.zeros(256), len(cs)))
            conn.send(out)
        else:
            conn.send([(i, metrics.image_scores(fuse2(X[i], *MODEL, ths[0], BAND), X[i]["gt"])) for i in ids])


class Shards:
    """Workers that each hold a 1/n shard of the keys (spawned processes, as Pool on Windows)."""

    def __init__(self, n, init_args, keys):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_shard_worker, args=(b, init_args, keys[w::n]), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            if c.recv() != "ready":
                raise RuntimeError("shard worker failed to start")

    def fmax(self, ths, ids):
        """F^max of each th over ids: mean F curve over all shards, as _fmax."""
        for c in self.conns:
            c.send(("curves", ths, ids))
        res = [c.recv() for c in self.conns]
        return [float((sum(r[j][0] for r in res) / sum(r[j][1] for r in res)).max()) for j in range(len(ths))]

    def full(self, th, ids):
        for c in self.conns:
            c.send(("scores", [th], ids))
        d = dict(kv for c in self.conns for kv in c.recv())
        return metrics.aggregate([d[i] for i in ids])

    def close(self):
        for c in self.conns:
            c.send(None)
        for p in self.procs:
            p.join(timeout=60)


def _fmax(args):
    th, ids = args
    curves = [metrics.f_curve(fuse2(X[i], *MODEL, th, BAND), X[i]["gt"]) for i in ids if X[i]["gt"].any()]
    return float(np.mean(curves, axis=0).max())


def _full(args):
    th, ids = args
    return metrics.aggregate([metrics.image_scores(fuse2(X[i], *MODEL, th, BAND), X[i]["gt"]) for i in ids])


def load_bins(path, k):
    """(dataset, id) keys of confidence bin k of a scripts/bins.py file."""
    b = json.loads(Path(path).read_text())["bins"]
    return {(ds, i) for ds, d in b.items() for i, v in d.items() if v == k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default="selfreformer")
    ap.add_argument("--prompt", default="mask_soft")
    ap.add_argument("--datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--unary", default="table")
    ap.add_argument("--pairwise", default="paper", choices=["none", "potts", "paper", "feat", "both", "paper_sb"])
    ap.add_argument("--output", default="binary", choices=["binary", "soft"])
    ap.add_argument("--band", action="store_true", help="hard band constraint, width band_w tuned")
    ap.add_argument("--bin-file", default=None, help="results/bins_<sod>.json of scripts/bins.py")
    ap.add_argument("--bin", type=int, default=None, help="restrict all images to this confidence bin")
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids per dataset (smoke tests)")
    ap.add_argument("--tune-half", default="A", choices=["A", "B", "all"])
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-refine", type=int, default=100)
    ap.add_argument("--shrink", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--init", nargs="*", default=[])
    ap.add_argument("--neg-prompt", default=None)
    ap.add_argument("--sod-model2", default=None, help="second SOD model for the duo unaries")
    ap.add_argument("--lo-model", default=None, choices=["table", "gbt"],
                    help="learned log-odds maps of learn_unary.py for --unary logodds")
    ap.add_argument("--shard", action="store_true", help="shard the images over the workers")
    ap.add_argument("--mem-budget", type=float, default=100.0,
                    help="GB; above this estimate the images are sharded automatically")
    args = ap.parse_args()
    if (args.bin_file is None) != (args.bin is None):
        ap.error("--bin-file and --bin go together")
    if (args.unary.split("+")[0] == pipeline.LOGODDS_UNARY) != (args.lo_model is not None):
        ap.error("--unary logodds and --lo-model go together")

    ranges = param_ranges(args.unary, args.pairwise, args.band)
    keys = list(ranges)
    lo0 = np.array([ranges[k][0] for k in keys], float)
    hi0 = np.array([ranges[k][1] for k in keys], float)
    lo, hi = lo0.copy(), hi0.copy()
    tune_ids = split_keys(args.datasets, args.tune_half, args.limit)
    other = {"A": "B", "B": "A", "all": "all"}[args.tune_half]
    test_ids = split_keys(args.datasets, other, args.limit)
    all_keys = split_keys(args.datasets, "all", args.limit)
    if args.bin_file:
        in_bin = load_bins(args.bin_file, args.bin)
        tune_ids, test_ids, all_keys = ([k for k in ks if k in in_bin] for ks in (tune_ids, test_ids, all_keys))
        if not tune_ids or not test_ids:
            sys.exit(f"bin {args.bin} of {args.bin_file} has no tuning or no held-out images")
    model = (args.unary, args.pairwise, args.output)
    if args.pairwise in pipeline.FEAT_PAIRWISE:
        # checked here: an exception in a Pool initializer only makes the pool respawn workers
        missing = [p for ds, i in all_keys for s in ("r", "d") if not (p := pipeline.feat_dir(ds) / f"{i}_{s}.png").exists()]
        if missing:
            sys.exit(f"{len(missing)} SAM feature files missing (run_sam_extra.py --prompt feat), e.g. {missing[0]}")
    if args.lo_model:
        missing = [p for ds, i in all_keys if not (p := pipeline.lo_dir(args.sod_model, ds, args.lo_model) / f"{i}.npy").exists()]
        if missing:
            sys.exit(f"{len(missing)} log-odds maps missing (learn_unary.py --model {args.lo_model}), e.g. {missing[0]}")

    rng = np.random.default_rng(args.seed)
    best_v, best_f, t0 = None, -1.0, time.time()
    init_v = []
    for path in args.init:
        res = json.loads(Path(path).read_text())
        # modifiers ("+veto", "+band") are off at the lower bound of their parameters, so a result of
        # the base model (converted to table_lin when that is the base) is an exact starting point
        init = res["params"]
        if args.unary.split("+")[0] == "table_lin":
            init = as_table_lin(res["model"][0].split("+")[0], res["params"])
        init_v.append(np.clip([init.get(k, ranges[k][0]) for k in keys], lo0, hi0))
    init_args = dict(sod_model=args.sod_model, datasets=args.datasets, prompt=args.prompt, model=model,
                     neg_prompt=args.neg_prompt, sod_model2=args.sod_model2, band=args.band,
                     lo_model=args.lo_model, keys=set(all_keys) if args.bin_file or args.limit else None)
    est_gb = len(all_keys) * MB_PER_IMAGE * args.workers / 1000
    shard = args.shard or est_gb > args.mem_budget
    print(f"{len(all_keys)} images, {args.workers} workers: "
          f"{'sharded' if shard else 'replicated'} (replicated estimate {est_gb:.0f} GB)", flush=True)

    def run_rounds(fmax_of):
        nonlocal best_v, best_f, lo, hi
        for rnd in range(args.rounds + 1):
            n = args.n_random if rnd == 0 else args.n_refine
            cand = lo + (hi - lo) * rng.random((n, len(keys)))
            if best_v is not None:
                cand[0] = best_v
            elif init_v:
                cand[:len(init_v)] = init_v
            for v, f in zip(cand, fmax_of([dict(zip(keys, v)) for v in cand])):
                if f > best_f:
                    best_v, best_f = v, f
            width = (hi - lo) * args.shrink
            lo, hi = np.maximum(best_v - width / 2, lo0), np.minimum(best_v + width / 2, hi0)

    if shard:
        sh = Shards(args.workers, init_args, all_keys)
        try:
            run_rounds(lambda ths: sh.fmax(ths, tune_ids))
            th = dict(zip(keys, map(float, best_v)))
            tune_m, test_m = sh.full(th, tune_ids), sh.full(th, test_ids)
            test_per_ds = {ds: sh.full(th, [k for k in test_ids if k[0] == ds]) for ds in args.datasets}
        finally:
            sh.close()
    else:
        with Pool(args.workers, initializer=_init_kw, initargs=(init_args,)) as pool:
            run_rounds(lambda ths: pool.imap(_fmax, [(t, tune_ids) for t in ths], chunksize=1))
            th = dict(zip(keys, map(float, best_v)))
            tune_m = pool.apply(_full, ((th, tune_ids),))
            test_m = pool.apply(_full, ((th, test_ids),))
            test_per_ds = {ds: pool.apply(_full, ((th, [k for k in test_ids if k[0] == ds]),))
                           for ds in args.datasets}

    tag = "+".join(args.datasets)
    ptag = args.prompt + (f"-neg-{args.neg_prompt}" if args.neg_prompt else "")
    ptag += f"-duo-{args.sod_model2}" if args.sod_model2 else ""
    name = f"{args.sod_model}_{tag}_{ptag}_{args.unary}_{args.pairwise}_{args.output}_{args.tune_half}"
    name += (f"_lo{args.lo_model}" if args.lo_model else "") + ("_band" if args.band else "")
    name += (f"_bin{args.bin}" if args.bin_file else "") + (f"_lim{args.limit}" if args.limit else "")
    name += f"_seed{args.seed}" if args.seed else ""     # seed 0 keeps the paper's file names
    out = {"model": model, "prompt": args.prompt, "neg_prompt": args.neg_prompt,
           "sod_model2": args.sod_model2, "lo_model": args.lo_model, "band": args.band, "params": th,
           "tune": tune_m, "test": test_m,
           "test_per_dataset": test_per_ds, "args": vars(args), "seconds": time.time() - t0}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"{name:<45s} tune F={tune_m['Fmax']:.4f} | held-out F={test_m['Fmax']:.4f} "
          f"MAE={test_m['MAE']:.4f} E={test_m['E']:.4f} S={test_m['S']:.4f} | "
          + " ".join(f"{ds}:F={m['Fmax']:.4f}" for ds, m in test_per_ds.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
