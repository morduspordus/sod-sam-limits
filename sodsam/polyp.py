"""Polyp segmentation datasets (Kvasir-SEG, CVC-ClinicDB) with the PraNet train/test split.

Layout under DATA_ROOT (env POLYP_DATA):
    Kvasir-SEG/extracted/Kvasir-SEG/images/<id>.jpg, masks/<id>.jpg
        (https://datasets.simula.no/downloads/kvasir-seg.zip, extracted with zipfile)
    CVC-ClinicDB/download/Original/<n>.png, "Ground Truth"/<n>.png, n = 1..612
        (PNG file mirror of the official release, huggingface.co/datasets/berkaytrhn/cvc-clinicdb)
Split: polyp_split.json next to this file holds the test ids of PraNet (Fan et al. 2020): the image names
in TestDataset/{Kvasir,CVC-ClinicDB} of the PraNet data release (taken from the Zenodo copy
https://zenodo.org/records/5579392, PolypDataset.zip); train = every other id (Kvasir 900 / 100,
CVC-ClinicDB 550 / 62).
Masks: foreground = gray level > 127 (Kvasir's masks are JPEGs with values near 0 and 255).
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(os.environ.get("POLYP_DATA", "data/polyp"))

# name -> (image dir, mask dir, image suffix, mask suffix), relative to DATA_ROOT
DATASETS = {
    "Kvasir-SEG": ("Kvasir-SEG/extracted/Kvasir-SEG/images", "Kvasir-SEG/extracted/Kvasir-SEG/masks", ".jpg", ".jpg"),
    "CVC-ClinicDB": ("CVC-ClinicDB/download/Original", "CVC-ClinicDB/download/Ground Truth", ".png", ".png"),
}
SPLIT_FILE = Path(__file__).with_name("polyp_split.json")
SPLITS = ("train", "test", "all")


def _sort_key(id_):
    return (0, int(id_), "") if id_.isdigit() else (1, 0, id_)


def all_ids(name):
    """Sorted ids with both an image and a mask (numeric ids in numeric order)."""
    img_dir, msk_dir, isuf, msuf = DATASETS[name]
    imgs = {p.stem for p in (DATA_ROOT / img_dir).glob("*" + isuf)}
    msks = {p.stem for p in (DATA_ROOT / msk_dir).glob("*" + msuf)}
    ids = sorted(imgs & msks, key=_sort_key)
    if not ids:
        raise FileNotFoundError(f"no images with masks for {name} under {DATA_ROOT / img_dir}")
    return ids


def test_ids(name):
    return sorted(json.loads(SPLIT_FILE.read_text())[name]["test"], key=_sort_key)


def list_ids(name, split="all"):
    """Ids of a split; raises if a PraNet test id is missing from the data."""
    ids = all_ids(name)
    if split == "all":
        return ids
    test = set(test_ids(name))
    missing = test - set(ids)
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} test ids missing from the data, e.g. {sorted(missing)[0]}")
    return [i for i in ids if (i in test) == (split == "test")]


def image_path(name, id_):
    img_dir, _, isuf, _ = DATASETS[name]
    return DATA_ROOT / img_dir / (id_ + isuf)


def mask_path(name, id_):
    _, msk_dir, _, msuf = DATASETS[name]
    return DATA_ROOT / msk_dir / (id_ + msuf)


def load_image(name, id_):
    """RGB uint8 HxWx3."""
    return np.array(Image.open(image_path(name, id_)).convert("RGB"))


def load_mask(name, id_):
    """Boolean HxW, gray level > 127."""
    return np.array(Image.open(mask_path(name, id_)).convert("L")) > 127


def gt_box(mask):
    """xyxy box (inclusive pixel coordinates) of the foreground of a boolean mask; None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def dilate_box(box, shape, frac=0.1):
    """Box grown by frac of its width / height on every side (at least 1 pixel), clipped to the image."""
    x0, y0, x1, y1 = box
    h, w = shape
    mx, my = max(1, int(round(frac * (x1 - x0 + 1)))), max(1, int(round(frac * (y1 - y0 + 1))))
    return [max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my)]


def pad_box(box, shape, pad=1):
    """Box grown by pad pixels, clipped to the image (the crop that holds the dilated box and its ring)."""
    x0, y0, x1, y1 = box
    h, w = shape
    return [max(0, x0 - pad), max(0, y0 - pad), min(w - 1, x1 + pad), min(h - 1, y1 + pad)]
