"""Write a fusion method's output maps to output/<sod>/<dataset>/<name>/ (fuse_eval --save cannot
store v2 methods: their names hold a path).

  python scripts/save_maps.py --sod-model samba --datasets ECSSD --name linear \
      --method v2:results/tune2/samba_MSOD+MSRA-B1K_mask_soft_linear_paper_binary_all.json
"""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sodsam import data, pipeline
from sodsam.fusion import PAPER_PARAMS


def _image(id_, sod_model, ds, prompt, method, name):
    inp = pipeline.load_inputs(sod_model, ds, prompt, id_, _image.scores)
    data.write_png(data.run_dir(sod_model, ds, name) / f"{id_}.png", pipeline.method_output(method, inp, PAPER_PARAMS))


def _init(scores):
    _image.scores = scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--prompt", default="mask_soft")
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--method", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    for ds in args.datasets:
        scores = pipeline.load_scores(args.sod_model, ds, args.prompt)
        with Pool(args.workers, initializer=_init, initargs=(scores,)) as pool:
            list(pool.imap_unordered(partial(_image, sod_model=args.sod_model, ds=ds, prompt=args.prompt,
                                             method=args.method, name=args.name), data.list_ids(ds), chunksize=16))
        print(f"{args.sod_model} {ds}: wrote {data.run_dir(args.sod_model, ds, args.name)}", flush=True)


if __name__ == "__main__":
    main()
