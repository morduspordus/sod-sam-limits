"""Import downloaded prediction maps of a high-resolution SOD/DIS model as a pseudo SOD model, in the
layout of run_sod.py --import-dir: output/<model>/<dataset>/sod/<id>.png (uint8, image size).

  python scripts/hr_import.py --model birefnet --src <dir> --datasets DIS-VD DIS-TE1 ... [--map HRSOD-TE=TE-HRSOD]

For each dataset the source subdirectory is --map NAME=subdir if given, else the first existing
candidate among the usual spellings (DIS-VD, DIS_VD, TE-HRSOD, HRSOD-TE, HRSOD_TE, UHRSD_TE, ...),
searched up to two levels below --src. Files are matched to the dataset ids by stem (any extension;
a trailing _sal/_pred/_mask suffix is also tried). Maps whose size differs from the image are resized
bilinearly, as Precomputed.load does. Then scripts/hr_prepare.py makes the 1K copies.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sodsam import data

SUFFIXES = ("_sal", "_pred", "_mask", "_gt", "_sod")
EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def candidates(name):
    """Usual spellings of a benchmark's directory in the authors' archives."""
    base = name
    if base.endswith("-TE"):
        short = base[:-3]
        return [base, base.replace("-", "_"), f"TE-{short}", f"{short}_TE", f"{short}-TE", f"{short}_TE_2K",
                f"{short}_test", f"{short}-test", f"{short}/test", f"{short}/TE", short]
    return [base, base.replace("-", "_"), base.replace("DIS-", "DIS5K-"), base.replace("DIS-", "")]


def find_src(root, name, explicit=None):
    root = Path(root)
    if explicit:
        p = root / explicit if not Path(explicit).is_absolute() else Path(explicit)
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    for depth in ("", "*/", "*/*/"):
        for c in candidates(name):
            hits = sorted(p for p in root.glob(depth + c) if p.is_dir())
            if hits:
                return hits[0]
    raise FileNotFoundError(f"no directory for {name} under {root} (tried {candidates(name)}); use --map")


def index_files(d):
    """stem -> path, preferring png when a stem exists with several extensions; descends into
    gt/pred-like subdirectories only if the directory itself holds no images."""
    files = [p for p in d.iterdir() if p.suffix.lower() in EXTS]
    if not files:
        files = [p for p in d.rglob("*") if p.suffix.lower() in EXTS]
    idx = {}
    for p in sorted(files, key=lambda p: (p.suffix.lower() != ".png", str(p))):
        idx.setdefault(p.stem, p)
    return idx


def lookup(idx, id_):
    if id_ in idx:
        return idx[id_]
    for s in SUFFIXES:
        if id_ + s in idx:
            return idx[id_ + s]
    # the source may have dropped a prefix/suffix of the id (e.g. "#1#..." names of DIS)
    stripped = re.sub(r"[^0-9A-Za-z]+", "", id_)
    for k, p in idx.items():
        if re.sub(r"[^0-9A-Za-z]+", "", k) == stripped:
            return p
    return None


def read_gray(path):
    a = np.array(Image.open(path))
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        a = a.astype(np.uint8) * 255
    elif a.dtype != np.uint8:
        a = np.clip(a.astype(np.float64) / (65535.0 if a.max() > 255 else 1.0), 0, 255).astype(np.uint8)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pseudo SOD model name, e.g. birefnet")
    ap.add_argument("--src", required=True, help="root of the extracted prediction archive")
    ap.add_argument("--datasets", nargs="+", default=data.HR_DATASETS)
    ap.add_argument("--map", nargs="*", default=[], help="NAME=subdir overrides of the source directory")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict", action="store_true", help="fail on a missing map instead of skipping it")
    args = ap.parse_args()
    explicit = dict(m.split("=", 1) for m in args.map)

    for ds in args.datasets:
        try:
            src = find_src(args.src, ds, explicit.get(ds))
        except FileNotFoundError as e:
            print(f"{args.model}/{ds}: {e}")
            continue
        idx = index_files(src)
        out = data.run_dir(args.model, ds, "sod")
        ids = data.list_ids(ds)
        missing, resized = [], 0
        for id_ in tqdm(ids, desc=f"{args.model}/{ds} <- {src.name}"):
            path = out / f"{id_}.png"
            if path.exists() and not args.overwrite:
                continue
            p = lookup(idx, id_)
            if p is None:
                missing.append(id_)
                if args.strict:
                    raise FileNotFoundError(f"{ds}/{id_} not in {src}")
                continue
            a = read_gray(p)
            with Image.open(data.image_path(ds, id_)) as im:
                w, h = im.size
            if a.shape != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                resized += 1
            data.write_png(path, a)
        n = len(list(out.glob("*.png")))
        print(f"{args.model}/{ds}: {n}/{len(ids)} maps in {out} (source {src}, {len(idx)} files, "
              f"{resized} resized, {len(missing)} missing{': ' + ' '.join(missing[:5]) if missing else ''})")


if __name__ == "__main__":
    main()
