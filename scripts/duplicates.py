"""Free two-annotator data: images that appear in more than one SOD dataset, each with its own mask
drawn by a different lab. How often do the two masks disagree, and how (error_types categories,
reading mask B as the "prediction" and mask A as the GT)? And where an evaluation image has a
duplicate, does the SOD model's mistake agree with the other lab's mask?

Duplicates: 8x8 gray thumbnails (zero mean, unit norm) -> nearest neighbours, verified by the
correlation of 32x32 thumbnails (>= 0.97) and the aspect ratio (within 3%). Masks are compared at
the smaller image's size.

  python scripts/duplicates.py --workers 2 --sod-model samba
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
from PIL import Image

from error_types import CATS, classify
from sodsam import data, metrics

R = data.DATA_ROOT
SETS = {name: (R / img, R / msk) for name, (img, msk) in data.DATASETS.items()}
SETS.update({"MSRA10K": (R / "MSRA10K/images", R / "MSRA10K/annotations/labels"),
             "MSRA1K": (R / "MSRA1K/images", R / "MSRA1K/labels")})
OUT = Path(__file__).resolve().parents[1] / "results" / "duplicates"


def mask_file(ds, id_):
    d = SETS[ds][1]
    for ext in (".png", ".bmp", ".jpg"):
        if (d / (id_ + ext)).exists():
            return d / (id_ + ext)
    return None


def _thumb(path):
    try:
        g = np.asarray(Image.open(path).convert("L"), np.float32)
    except Exception:
        return None
    h, w = g.shape
    t8 = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    t32 = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).ravel()
    norm = lambda t: (t - t.mean()) / (np.linalg.norm(t - t.mean()) + 1e-6)
    return norm(t8), norm(t32), w / h


def load_mask(ds, id_, shape=None):
    p = mask_file(ds, id_)
    m = np.asarray(Image.open(p))
    m = (m[..., 0] if m.ndim == 3 else m) > 128 if m.dtype != bool else m
    if shape is not None and m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sod-model", default="samba")
    args = ap.parse_args()
    from sklearn.neighbors import NearestNeighbors
    OUT.mkdir(parents=True, exist_ok=True)

    items = []
    for ds, (img_dir, _) in SETS.items():
        if not img_dir.exists():
            continue
        ids = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".bmp"))
        items += [(ds, p.stem, p) for p in ids if mask_file(ds, p.stem)]
        print(f"{ds}: {sum(1 for x in items if x[0] == ds)} images with masks", flush=True)
    with Pool(args.workers) as pool:
        th = pool.map(_thumb, [p for _, _, p in items], chunksize=64)
    keep = [i for i, t in enumerate(th) if t is not None]
    items, th = [items[i] for i in keep], [th[i] for i in keep]
    X8 = np.stack([t[0] for t in th])
    nn = NearestNeighbors(n_neighbors=8).fit(X8)
    _, idx = nn.kneighbors(X8)
    pairs = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            a, b = sorted((i, int(j)))
            if items[a][0] != items[b][0] and abs(th[a][2] / th[b][2] - 1) < 0.03 \
                    and float(th[a][1] @ th[b][1]) >= 0.97:
                pairs.add((a, b))
    print(f"{len(pairs)} cross-dataset duplicate pairs", flush=True)

    # MSRA-B is the parent of the MSRA-B1K/4K subsets; MSRA1K and MSRA10K overlap MSRA-B by design
    stats = defaultdict(lambda: {"n": 0, "iou": [], "cats": Counter(), "examples": []})
    model_rows = []
    eval_sets = set(data.EVAL_DATASETS) | {"MSOD"}
    for a, b in sorted(pairs):
        (da, ia, pa), (db, ib, pb) = items[a], items[b]
        ga = load_mask(da, ia)
        gb = load_mask(db, ib)
        shape = min(ga.shape, gb.shape, key=lambda s: s[0] * s[1])
        ga, gb = load_mask(da, ia, shape), load_mask(db, ib, shape)
        if not ga.any() or not gb.any():
            continue
        u = np.logical_or(ga, gb).sum()
        iou = float(np.logical_and(ga, gb).sum() / u)
        cat, _ = classify(gb, ga)
        key = f"{da}|{db}"
        s = stats[key]
        s["n"] += 1
        s["iou"].append(iou)
        s["cats"][cat] += 1
        if cat != "boundary" and len(s["examples"]) < 12:
            s["examples"].append([ia, ib, cat, round(iou, 3)])
        # an evaluation image with a duplicate: is the model closer to the other lab's mask?
        for (de, ie, ge), (do, io, go) in (((da, ia, ga), (db, ib, gb)), ((db, ib, gb), (da, ia, ga))):
            if de not in eval_sets:
                continue
            mp = data.run_dir(args.sod_model, de, "sod") / f"{ie}.png"
            if not mp.exists():
                continue
            m = data.read_png(mp)
            if m.shape != ge.shape:
                m = cv2.resize(m, (ge.shape[1], ge.shape[0]), interpolation=cv2.INTER_LINEAR)
            f_own = float(metrics.f_curve(m, ge).max())
            f_other = float(metrics.f_curve(m, go).max())
            model_rows.append({"eval": de, "id": ie, "other": do, "other_id": io, "masks_iou": iou,
                               "f_own": f_own, "f_other": f_other})
    out = {}
    print(f"\n{'pair':28s} {'n':>5s} {'IoU<.85':>8s} {'meanIoU':>8s}  categories (B read against A)")
    for key, s in sorted(stats.items(), key=lambda kv: -kv[1]["n"]):
        iou = np.array(s["iou"])
        out[key] = {"n": s["n"], "mean_iou": float(iou.mean()), "share_iou_below_085": float((iou < 0.85).mean()),
                    "categories": dict(s["cats"]), "examples": s["examples"]}
        print(f"{key:28s} {s['n']:5d} {(iou < 0.85).mean():8.0%} {iou.mean():8.3f}  "
              + " ".join(f"{c}={s['cats'][c]}" for c in CATS if s["cats"][c]))
    if model_rows:
        dis = [r for r in model_rows if r["masks_iou"] < 0.85]
        print(f"\n{args.sod_model}: {len(model_rows)} evaluation images with a duplicate; on the {len(dis)} where "
              f"the two labs disagree (IoU<.85), the model is closer to the other lab's mask on "
              f"{np.mean([r['f_other'] > r['f_own'] for r in dis]):.0%}" if dis else "")
        for r in sorted(dis, key=lambda r: r["f_own"] - r["f_other"])[:10]:
            print(f"   {r['eval']}/{r['id']} vs {r['other']}/{r['other_id']}: F own {r['f_own']:.3f}, "
                  f"other {r['f_other']:.3f}, masks IoU {r['masks_iou']:.2f}")
    (OUT / "duplicates.json").write_text(json.dumps({"pairs": out, "model": model_rows}, indent=1))


if __name__ == "__main__":
    main()
