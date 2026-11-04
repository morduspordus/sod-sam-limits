"""ECSSD metrics with and without the ~130 test images that also appear (with other masks) in the
tuning/training pools MSOD and MSRA-B (found by duplicates.py): does the overlap change any
conclusion? Maps: SOD, graph cut table/linear (save_maps.py --name gc_table / gc_linear), heads.

  python scripts/ecssd_clean_check.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2

from sodsam import data, metrics

RES = Path(__file__).resolve().parents[1] / "results" / "duplicates"
POOLS = ("MSOD", "MSRA-B", "MSRA-B1K", "MSRA-B4K")


def main():
    dup = json.loads((RES / "duplicates.json").read_text())
    bad = sorted({r["id"] for r in dup["model"] if r["eval"] == "ECSSD" and r["other"] in POOLS})
    ids = data.list_ids("ECSSD")
    clean = [i for i in ids if i not in set(bad)]
    print(f"{len(bad)} ECSSD images also in the tuning/training pools; {len(clean)} clean", flush=True)
    gts = {i: data.load_gt("ECSSD", i) for i in ids}
    out = {}
    for s in ("samba", "selfreformer"):
        for name, sub in (("sod", data.run_dir(s, "ECSSD", "sod")), ("gc_table", data.run_dir(s, "ECSSD", "gc_table")),
                          ("gc_linear", data.run_dir(s, "ECSSD", "gc_linear")),
                          ("head", data.run_dir(f"{s}_head", "ECSSD", "sod")),
                          ("head4k", data.run_dir(f"{s}_head4k", "ECSSD", "sod"))):
            per = {}
            for i in ids:
                m, g = data.read_png(sub / f"{i}.png"), gts[i]
                if m.shape != g.shape:
                    m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
                per[i] = metrics.image_scores(m, g)
            r = {k: metrics.aggregate([per[i] for i in sel]) for k, sel in (("full", ids), ("clean", clean), ("overlap", bad))}
            out[f"{s}/{name}"] = r
            print(f"{s:12s} {name:9s} full F={r['full']['Fmax']:.3f} MAE={r['full']['MAE']:.3f} | clean "
                  f"F={r['clean']['Fmax']:.3f} MAE={r['clean']['MAE']:.3f} | overlap only F={r['overlap']['Fmax']:.3f}",
                  flush=True)
    (RES / "ecssd_clean_check.json").write_text(json.dumps({"overlap_ids": bad, "results": out}, indent=1))


if __name__ == "__main__":
    main()
