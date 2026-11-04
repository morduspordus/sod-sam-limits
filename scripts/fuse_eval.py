"""Run the fusion (and baselines / ablations) on cached SOD and SAM outputs and evaluate.

  python scripts/fuse_eval.py --sod-model selfreformer --prompt mask_soft --params results/params_selfreformer.json
Methods: sod, sod_crf, sam_conf, unary, potts, full, full_legacy (see sodsam/pipeline.py), and
"v2:<tune2 json>[:<output>]" (fusion2 models; feat/both pairwise, --band and logodds results load what
they need: a logodds result's "lo_model" names the log-odds maps of learn_unary.py).
"v2bins:<json bin 0>,<json bin 1>,<json bin 2>" with --bin-file results/bins_<sod>.json (scripts/bins.py)
picks the result of the image's confidence bin; the rest is the v2 method, so the numbers are comparable.
Zero-shot SAM maps (Table 1) are evaluated with --zero-shot <sam subdir>... .
--limit N evaluates only the first N ids of every dataset (smoke tests).
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm

from sodsam import data, metrics, pipeline
from sodsam.fusion import MATLAB_OPTIMAL_PARAMS, PAPER_PARAMS, FusionParams

RESULTS = Path(__file__).resolve().parents[1] / "results"


def load_params(spec):
    if spec == "paper":
        return PAPER_PARAMS
    if spec == "matlab":
        return MATLAB_OPTIMAL_PARAMS
    d = json.loads(Path(spec).read_text())
    return FusionParams.from_dict(d.get("params", d))


def _image(id_, sod_model, ds, prompt, methods, params, save, crf_params=None, neg_prompt=None,
           sod_model2=None, lo_model=None):
    scores, bins = _image.scores, _image.bins
    inp = pipeline.load_inputs(sod_model, ds, prompt, id_, scores, neg_prompt, sod_model2, lo_model)
    out = {}
    for m in methods:
        spec = m
        if m.startswith("v2bins:"):
            if bins is None:
                raise ValueError("v2bins methods need --bin-file")
            spec = "v2:" + m[len("v2bins:"):].split(",")[bins[ds][id_]]
        pred = pipeline.method_output(spec, inp, params, crf_params)
        if save:
            data.write_png(data.run_dir(sod_model, ds, f"{m}_{save}") / f"{id_}.png", pred)
        out[m] = metrics.image_scores(pred, inp["gt"])
    return out


def _init(scores, bins=None):
    _image.scores = scores
    _image.bins = bins


def lo_model_of(methods):
    """The "lo_model" of the v2 results with a logodds unary among methods (None without one; the
    maps of a second model are loaded per image by pipeline.method_output)."""
    models = set()
    for m in methods:
        if m.startswith("v2bins:"):
            specs = m[len("v2bins:"):].split(",")
        elif m.startswith("v2:"):
            specs = [m[3:]]
        else:
            continue
        for spec in specs:
            path = spec
            if ":" in spec and spec.rsplit(":", 1)[1] in pipeline.V2_OUTPUTS:
                path = spec.rsplit(":", 1)[0]
            res = pipeline._v2_model(path)
            if res["model"][0].split("+")[0] == pipeline.LOGODDS_UNARY:
                models.add(res["lo_model"])
    return sorted(models)[0] if models else None


def _zero_shot_image(id_, sod_model, ds, sub):
    pred = data.read_png(data.run_dir(sod_model, ds, sub) / f"{id_}_1.png")
    return metrics.image_scores(pred, data.load_gt(ds, id_))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_soft")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--methods", nargs="+", default=["sod", "sam_conf", "unary", "potts", "full"])
    ap.add_argument("--params", default="paper", help="paper | matlab | path to json")
    ap.add_argument("--zero-shot", nargs="+", help="evaluate SAM subdirs (mask _1) instead")
    ap.add_argument("--save", default="", help="if set, write maps to <method>_<save>/")
    ap.add_argument("--crf-params", help="json from tune_crf.py (default: DSS settings)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--neg-prompt", default=None, help="complement-prompt masks for v2 models with +veto")
    ap.add_argument("--sod-model2", default=None, help="second SOD model for v2 duo models")
    ap.add_argument("--bin-file", default=None, help="results/bins_<sod>.json for v2bins methods")
    ap.add_argument("--limit", type=int, default=None, help="only the first N ids per dataset")
    args = ap.parse_args()

    params = load_params(args.params)
    crf_params = json.loads(Path(args.crf_params).read_text())["crf_params"] if args.crf_params else None
    bins = json.loads(Path(args.bin_file).read_text())["bins"] if args.bin_file else None
    lo_model = None if args.zero_shot else lo_model_of(args.methods)
    results = {"sod_model": args.sod_model, "prompt": args.prompt, "params": params.to_dict(),
               "crf_params": crf_params, "bin_file": args.bin_file, "lo_model": lo_model, "datasets": {}}
    for ds in args.datasets:
        ids = data.list_ids(ds)[:args.limit]
        res = {}
        if args.zero_shot:
            with Pool(args.workers) as pool:
                for sub in args.zero_shot:
                    f = partial(_zero_shot_image, sod_model=args.sod_model, ds=ds, sub=sub)
                    per = list(tqdm(pool.imap(f, ids, chunksize=8), total=len(ids), desc=f"{ds}/{sub}"))
                    res[sub] = metrics.aggregate(per)
        else:
            scores = None if args.sod_model2 else pipeline.load_scores(args.sod_model, ds, args.prompt)
            f = partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt,
                        methods=args.methods, params=params, save=args.save, crf_params=crf_params,
                        neg_prompt=args.neg_prompt, sod_model2=args.sod_model2, lo_model=lo_model)
            with Pool(args.workers, initializer=_init, initargs=(scores, bins)) as pool:
                per = list(tqdm(pool.imap(f, ids, chunksize=4), total=len(ids), desc=ds))
            for m in args.methods:
                res[m] = metrics.aggregate([p[m] for p in per])
        results["datasets"][ds] = res
        for name, m in res.items():
            print(metrics.format_row(f"{ds} {name}", m), flush=True)

    RESULTS.mkdir(exist_ok=True)
    name = args.tag or f"{args.sod_model}_{args.prompt}_{'zeroshot' if args.zero_shot else Path(args.params).stem}"
    (RESULTS / f"eval_{name}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
