"""Check 3: are an SOD model's "wrong object" errors on PASCAL-S objects that some annotators chose?

PASCAL-S (Li et al., CVPR 2014) records, per pixel, the fraction of 12 subjects who clicked the object
segment containing it (graded masks, values k/12); the binary GT is fraction >= 0.5. For each image
and model, with S the thresholded model mask and G the binary GT:
  extra = S \\ G   (what the model added beyond the GT)
  support(extra) = share of the extra pixels whose object was chosen by >= 1, >= 3 (25%), >= 4 (1/3)
                   of the 12 subjects
Reported per error type of error_types.py, plus the share of the model's total loss that sits on
images whose extra region is mostly (>= 50% of its pixels) chosen by at least 3 of 12 subjects, i.e.
the model picked an object that a sizable minority of people also found salient.

  python scripts/pascal_agreement.py --models samba selfreformer --maps mdsam=<root> ...
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np

from error_types import CATS, classify
from sodsam import data, metrics

GRADED = data.DATA_ROOT / "downloads" / "PASCAL-S_full" / "datasets" / "masks" / "pascal"
OUT = Path(__file__).resolve().parents[1] / "results" / "error_types"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="maps under output/<model>/PASCAL-S/sod")
    ap.add_argument("--maps", nargs="*", default=[], help="name=root with <root>/PASCAL-S/<id>.png")
    args = ap.parse_args()
    sources = {m: data.run_dir(m, "PASCAL-S", "sod") for m in args.models}
    sources.update({kv.split("=", 1)[0]: Path(kv.split("=", 1)[1]) / "PASCAL-S" for kv in args.maps})
    ids = data.list_ids("PASCAL-S")
    graded, mismatch = {}, 0
    for i in ids:
        a = data.read_png(GRADED / f"{i}.png").astype(np.float64) / 255.0
        g = data.load_gt("PASCAL-S", i)
        if a.shape != g.shape or not np.array_equal(a >= 0.5 - 1e-6, g):
            mismatch += 1
        graded[i] = a
    print(f"PASCAL-S: {len(ids)} images; graded >= 0.5 differs from the benchmark GT on {mismatch}", flush=True)
    out = {}
    for name, root in sources.items():
        per_cat = defaultdict(list)
        loss_tot, loss_minority = 0.0, 0.0
        for i in ids:
            g = data.load_gt("PASCAL-S", i)
            if not g.any():
                continue
            m = data.read_png(root / f"{i}.png")
            if m.shape != g.shape:
                m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
            loss = 1.0 - float(metrics.f_curve(m, g).max())
            S = m.astype(np.float64) / (m.max() + 1e-13) >= 127 / 255 if m.max() > 0 else np.zeros_like(g)
            cat, _ = classify(S, g)
            extra = S & ~g
            loss_tot += loss
            if extra.sum() < 0.005 * g.size:              # nothing substantial added
                continue
            a = graded[i][extra]
            sup = {k: float((a >= k / 12 - 1e-6).mean()) for k in (1, 3, 4)}
            per_cat[cat].append((loss, sup))
            if sup[3] >= 0.5:
                loss_minority += loss
        res = {"minority_supported_loss_share": loss_minority / max(loss_tot, 1e-9), "by_category": {}}
        print(f"{name}: {loss_minority / max(loss_tot, 1e-9):.0%} of the PASCAL-S loss is on images whose extra "
              f"region was mostly chosen by >= 3 of 12 subjects", flush=True)
        for c in CATS:
            if per_cat[c]:
                sups = [s for _, s in per_cat[c]]
                r = {f">={k}/12": float(np.mean([s[k] for s in sups])) for k in (1, 3, 4)}
                r["n"] = len(sups)
                res["by_category"][c] = r
                print(f"   {c:13s} n={len(sups):4d}  extra pixels chosen by >=1/12: {r['>=1/12']:.0%}  "
                      f">=3/12: {r['>=3/12']:.0%}  >=4/12: {r['>=4/12']:.0%}", flush=True)
        out[name] = res
    (OUT / "pascal_agreement.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
