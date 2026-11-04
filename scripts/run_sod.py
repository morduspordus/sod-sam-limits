"""Compute (or import) soft SOD maps.

  python scripts/run_sod.py --model selfreformer --datasets MSOD DUTS-TE ECSSD DUT-OMRON PASCAL-S HKU-IS
  python scripts/run_sod.py --model samba --import-dir <dir with DATASET/<id>.png> --datasets ...
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tqdm import tqdm

from sodsam import data
from sodsam.sod_models import MODELS, Precomputed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=[data.TUNE_DATASET] + data.EVAL_DATASETS)
    ap.add_argument("--import-dir", help="root with <dataset>/<id>.png maps computed elsewhere")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    model = None if args.import_dir else MODELS[args.model]()
    for ds in args.datasets:
        out = data.run_dir(args.model, ds, "sod")
        src = Precomputed(Path(args.import_dir) / ds) if args.import_dir else None
        for id_ in tqdm(data.list_ids(ds), desc=f"{args.model}/{ds}"):
            path = out / f"{id_}.png"
            if path.exists() and not args.overwrite:
                continue
            img = data.load_image(ds, id_)
            sal = src.load(id_, img.shape[:2]) if src else model(img)
            data.write_png(path, sal)


if __name__ == "__main__":
    main()
