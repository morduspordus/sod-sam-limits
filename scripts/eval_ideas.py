"""Apply a tuned ideas model (tune_ideas.py result json) to datasets, save its maps and evaluate it.

Maps: <sod>/<ds>/nested/<id>_in.png and _out.png; <sod>/<ds>/scales/<id>.png; <sod>/<ds>/decision/<id>.png.
Results: results/eval_<sod>_idea_<model>.json ({"datasets": {ds: {...}}}; nested: inner, outer, best_of_two_F).

  python scripts/eval_ideas.py --sod-model samba --result results/tune2/samba_MSRA-B1K_idea_nested_all.json --workers 14
"""
import argparse
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from sodsam import data, pipeline
from tune_ideas import aggregate, apply, full_scores, load_record

_SCORES = {}


def _image(id_, ds, res, sod):
    if ds not in _SCORES:
        _SCORES[ds] = pipeline.load_scores(sod, ds, res["prompt"])
    gc_res = json.loads(Path(res["gc_params"]).read_text()) if res["gc_params"] else None
    rec = load_record(res["model"], sod, ds, id_, _SCORES[ds], res["prompt"], res["layers"], gc_res)
    out = apply(res["model"], rec, res["params"])
    d = data.run_dir(sod, ds, res["model"])
    if res["model"] == "nested":
        data.write_png(d / f"{id_}_in.png", out[0].astype(np.uint8) * 255)
        data.write_png(d / f"{id_}_out.png", out[1].astype(np.uint8) * 255)
    else:
        data.write_png(d / f"{id_}.png", out.astype(np.uint8) * 255)
    return full_scores(res["model"], rec, res["params"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    res = json.loads(Path(args.result).read_text())
    out_path = Path(__file__).resolve().parents[1] / "results" / f"eval_{args.sod_model}_idea_{res['model']}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"sod_model": args.sod_model, "datasets": {}}
    out.update({"model": res["model"], "params": res["params"], "result": args.result})
    with Pool(args.workers) as pool:
        for ds in args.datasets:
            ids = data.list_ids(ds)
            ids = ids[:args.limit] if args.limit else ids
            data.run_dir(args.sod_model, ds, res["model"]).mkdir(parents=True, exist_ok=True)
            per = pool.map(partial(_image, ds=ds, res=res, sod=args.sod_model), ids, chunksize=8)
            out["datasets"][ds] = aggregate(res["model"], per)
            r = out["datasets"][ds]
            if res["model"] == "nested":
                print(f"{ds:10s} inner F={r['inner']['Fmax']:.3f} MAE={r['inner']['MAE']:.3f} | outer F={r['outer']['Fmax']:.3f} "
                      f"MAE={r['outer']['MAE']:.3f} | best-of-two F={r['best_of_two_F']:.3f} (n={len(per)})", flush=True)
            else:
                print(f"{ds:10s} {res['model']} F={r['Fmax']:.3f} MAE={r['MAE']:.3f} E={r['E']:.3f} S={r['S']:.3f} (n={len(per)})", flush=True)
            out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
