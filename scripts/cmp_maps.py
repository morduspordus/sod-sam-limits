"""Compare our saliency maps with the authors' released maps: mean absolute difference and Fmax of both.
  python -I scripts/cmp_maps.py <ours_dir> <released_dir> <gt_dir>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(p, shape=None):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    if shape is not None and a.shape != shape:
        a = np.asarray(Image.open(p).convert("L").resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)
    return a


def fmax(preds, gts):
    th = np.arange(256)
    tp = np.zeros(256); pp = np.zeros(256); gp = 0.0
    prec = np.zeros(256); rec = np.zeros(256)
    for p, g in zip(preds, gts):
        g = g > 127
        hist_fg = np.histogram(p[g], bins=256, range=(0, 256))[0]
        hist_all = np.histogram(p, bins=256, range=(0, 256))[0]
        tp_i = np.cumsum(hist_fg[::-1])[::-1]       # pixels >= t that are fg
        pp_i = np.cumsum(hist_all[::-1])[::-1]
        prec += tp_i / np.maximum(pp_i, 1)
        rec += tp_i / max(g.sum(), 1)
    prec /= len(preds); rec /= len(preds)
    f = 1.3 * prec * rec / np.maximum(0.3 * prec + rec, 1e-8)
    return f.max()


ours, rel, gt = (Path(a) for a in sys.argv[1:4])
names = sorted(p.stem for p in ours.glob("*.png"))
diffs, P1, P2, G = [], [], [], []
for n in names:
    g = load(gt / f"{n}.png")
    a = load(ours / f"{n}.png", g.shape)
    rp = rel / f"{n}.png"
    if not rp.exists():
        continue
    b = load(rp, g.shape)
    diffs.append(np.abs(a - b).mean() / 255)
    P1.append(a); P2.append(b); G.append(g)
d = np.array(diffs)
print(f"{len(d)} images; mean |ours - released| {d.mean():.4f}, median {np.median(d):.4f}, max {d.max():.4f}")
print(f"Fmax ours {fmax(P1, G):.4f}  released {fmax(P2, G):.4f}")
