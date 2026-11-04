"""CPU runtime of the graph cut for the paper: one process, the first --n DUTS-TE images, time of the
fusion step only (inputs loaded beforehand), with a tuned result json.

  python scripts/time_gc.py --sod-model samba --method v2:results/tune2/<...>.json --prompt mask_zoom
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import data, pipeline
from sodsam.fusion import PAPER_PARAMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default="samba")
    ap.add_argument("--method", required=True)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    ids = data.list_ids("DUTS-TE")[:args.n]
    scores = pipeline.load_scores(args.sod_model, "DUTS-TE", args.prompt)
    times, sizes = [], []
    for id_ in ids:
        inp = pipeline.load_inputs(args.sod_model, "DUTS-TE", args.prompt, id_, scores)
        t = time.perf_counter()
        pipeline.method_output(args.method, inp, PAPER_PARAMS)
        times.append(time.perf_counter() - t)
        sizes.append(inp["sod"].shape)
    try:
        cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    res = {"mean_s": float(np.mean(times)), "median_s": float(np.median(times)), "n": len(ids), "cpu": cpu,
           "mean_size": [float(np.mean([s[0] for s in sizes])), float(np.mean([s[1] for s in sizes]))]}
    print(json.dumps(res, indent=1))
    (Path(__file__).resolve().parents[1] / "results" / "timing_gc.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
