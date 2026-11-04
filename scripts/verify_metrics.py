"""Cross-check our metrics against PySODMetrics (pip package py_sod_metrics) on saved maps.

Runs in the separate venv .venv_metrics (so the main venv's packages are not changed):
  .venv_metrics/Scripts/python.exe scripts/verify_metrics.py --maps sod=samba:sod head=samba_hz:sod

Each --maps entry is name=<run>:<subdir>, read from <OUT_ROOT>/<run>/<dataset>/<subdir>/<id>.png. Reports
our F^max / MAE / E / S (sodsam.metrics) next to PySODMetrics' max F (beta^2 = 0.3), MAE, adaptive
E-measure, and S-measure, and the largest absolute difference.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import py_sod_metrics as psm

from sodsam import data, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--out", default="results/verify_metrics.json")
    args = ap.parse_args()
    res, worst = {}, 0.0
    for spec in args.maps:
        name, src = spec.split("=", 1)
        run, sub = src.split(":", 1)
        for ds in args.datasets:
            fm, mae, em, sm = psm.Fmeasure(), psm.MAE(), psm.Emeasure(), psm.Smeasure()
            ours = []
            for i in data.list_ids(ds):
                g = data.load_gt(ds, i)
                p = data.read_png(data.run_dir(run, ds, sub) / f"{i}.png")
                if p.shape != g.shape:
                    p = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
                gt = g.astype(np.uint8) * 255
                for m in (fm, mae, em, sm):
                    m.step(pred=p, gt=gt)
                ours.append(metrics.image_scores(p, g))
            o = metrics.aggregate(ours)
            theirs = {"Fmax": float(fm.get_results()["fm"]["curve"].max()), "MAE": float(mae.get_results()["mae"]),
                      "E": float(em.get_results()["em"]["adp"]), "S": float(sm.get_results()["sm"])}
            diff = {k: abs(o[k] - theirs[k]) for k in theirs}
            worst = max(worst, max(diff.values()))
            res[f"{name}/{ds}"] = {"ours": {k: o[k] for k in theirs}, "pysodmetrics": theirs, "absdiff": diff}
            print(f"{name:10s} {ds:10s} " + " ".join(f"{k} {o[k]:.4f}/{theirs[k]:.4f}" for k in theirs), flush=True)
    print(f"largest absolute difference: {worst:.5f}")
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
