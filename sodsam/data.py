"""Benchmark datasets and the on-disk layout of intermediate results.

Layout under OUT_ROOT:
    <sod_model>/<dataset>/sod/<id>.png            soft saliency map, uint8, image size
    <sod_model>/<dataset>/sam_<prompt>/<id>_<k>.png  SAM mask k (1..3), 0/255
    <sod_model>/<dataset>/sam_<prompt>/scores.json   {id: [s1, s2, s3]}
    <sod_model>/<dataset>/<method>/<id>.png        final map evaluated by the metrics
SAM zero-shot runs that do not depend on an SOD model use <sod_model> = "none".
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(os.environ.get("SALIENCY_DATA", "data/saliency"))
OUT_ROOT = Path(os.environ.get("SODSAM_OUT", Path(__file__).resolve().parents[1] / "output"))

# name -> (image dir, mask dir), relative to DATA_ROOT
DATASETS = {
    "DUTS-TE": ("DUTS/benchmark/DUTSTE/images", "DUTS/benchmark/DUTSTE/masks"),
    "DUT-OMRON": ("DUT-OMRON/benchmark/test/images", "DUT-OMRON/benchmark/test/masks"),
    "ECSSD": ("ECSSD/benchmark/test/images", "ECSSD/benchmark/test/masks"),
    "PASCAL-S": ("PascalS/benchmark/Test/images", "PascalS/benchmark/Test/masks"),
    "HKU-IS": ("HKU_IS/benchmark/Test/images", "HKU_IS/benchmark/Test/masks"),
    # Movahedi & Elder (2010), 300 images; used only for parameter selection
    "MSOD": ("SOD/images", "SOD/annotations/labels"),
    "MSRA-B": ("MSRA_B/images", "MSRA_B/annotations/labels"),
    # the SOD models' training set; used only to train the refinement head (item 7)
    "DUTS-TR": ("DUTS/DUTSTR/images", "DUTS/DUTSTR/masks"),
}
# High-resolution benchmarks (2K-6K images; scripts/hr_prepare.py makes the "<name>-1k" copies with the
# longest side 1024 that the pipeline runs on, scripts/hr_eval.py scores them against the full-res GT).
# Layout under DATA_ROOT: <name>/images, <name>/masks (junctions to the extracted archives).
HR_DATASETS = ["DIS-VD", "DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4", "HRSOD-TE", "UHRSD-TE"]
HR_SUFFIX = "-1k"
for _n in HR_DATASETS:
    DATASETS[_n] = (f"{_n}/images", f"{_n}/masks")
    DATASETS[_n + HR_SUFFIX] = (f"{_n}{HR_SUFFIX}/images", f"{_n}{HR_SUFFIX}/masks")


def hr_parent(name):
    """Full-resolution dataset of a "-1k" derived one (the name itself if it is not derived)."""
    return name[:-len(HR_SUFFIX)] if name.endswith(HR_SUFFIX) else name


# name -> (parent dataset, size, seed): fixed random subsets, used only for parameter selection;
# a negative size is the complement of the subset of that size (same seed)
SUBSETS = {"MSRA-B1K": ("MSRA-B", 1000, 0), "MSRA-B4K": ("MSRA-B", -1000, 0)}
DATASETS.update({k: DATASETS[v[0]] for k, v in SUBSETS.items()})
EVAL_DATASETS = ["DUTS-TE", "ECSSD", "DUT-OMRON", "PASCAL-S", "HKU-IS"]
TUNE_DATASET = "MSOD"
TUNE_DATASETS = ["MSOD", "MSRA-B1K"]   # selection set: MSOD + 1000 MSRA-B images (unseen by the SOD models)
TUNE_DATASETS_ALL = ["MSOD", "MSRA-B1K", "MSRA-B4K"]   # the same plus the other 4000 MSRA-B images


def image_dir(name):
    return DATA_ROOT / DATASETS[name][0]


def mask_dir(name):
    return DATA_ROOT / DATASETS[name][1]


def _excluded():
    p = Path(__file__).with_name("excluded_ids.json")
    if not p.exists():
        return {}
    return {k: set(v) for k, v in json.loads(p.read_text()).items() if not k.startswith("_")}


EXCLUDED = _excluded()


def list_ids(name):
    """Sorted ids of images that have a ground-truth mask, without the tuning/training images that
    duplicate an evaluation image (EXCLUDED; subsets are drawn first, so they keep their members)."""
    gts = {p.stem for p in mask_dir(name).glob("*.png")}
    ids = sorted(p.stem for p in image_dir(name).iterdir()
                 if p.suffix.lower() in (".jpg", ".png") and p.stem in gts)
    if not ids:
        raise FileNotFoundError(f"no images with masks for {name} in {image_dir(name)}")
    if name in SUBSETS:
        _, size, seed = SUBSETS[name]
        chosen = np.random.default_rng(seed).choice(ids, abs(size), replace=False).tolist()
        ids = sorted(chosen) if size > 0 else sorted(set(ids) - set(chosen))
    drop = EXCLUDED.get(SUBSETS[name][0] if name in SUBSETS else name, set())
    return [i for i in ids if i not in drop]


def image_path(name, id_):
    for ext in (".jpg", ".png"):
        p = image_dir(name) / (id_ + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}/{id_}")


def load_image(name, id_):
    """RGB uint8 HxWx3 (gray images are replicated to 3 channels)."""
    return np.array(Image.open(image_path(name, id_)).convert("RGB"))


def load_gt(name, id_):
    """Boolean HxW; 1-bit masks are kept, otherwise foreground is first channel > 128,
    as in the MATLAB toolbox."""
    g = np.array(Image.open(mask_dir(name) / (id_ + ".png")))
    if g.ndim == 3:
        g = g[..., 0]
    return g if g.dtype == bool else g > 128


def run_dir(sod_model, dataset, sub):
    return OUT_ROOT / sod_model / dataset / sub


def read_png(path):
    a = np.array(Image.open(path))
    return a[..., 0] if a.ndim == 3 else a


def write_png(path, a):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a.astype(np.uint8)).save(path)
