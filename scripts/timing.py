"""Runtime for the paper: SAM 2.1 (image encoding + mask prompt, plain and zoom-in), and the fusion head
forward pass, per image on the GPU, over the first --n DUTS-TE images (after --warmup images).

  python scripts/timing.py --sod-model samba --head results/head/samba_hz.pt --n 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from sodsam import data, pipeline
from sodsam.sam_prompting import SamRunner
from train_head import Head, features


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", default="samba")
    ap.add_argument("--head", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    ids = data.list_ids("DUTS-TE")[:args.n + args.warmup]
    sam = SamRunner()
    net = Head().cuda().eval()
    net.load_state_dict(torch.load(args.head, map_location="cuda"))
    t = {"sam_plain": [], "sam_zoom": [], "head": [], "head_features": []}
    for i, id_ in enumerate(ids):
        img = data.load_image("DUTS-TE", id_)
        sod = data.read_png(data.run_dir(args.sod_model, "DUTS-TE", "sod") / f"{id_}.png")
        _, a = timed(lambda: (sam.set_image(img), sam.from_mask(sod, "soft")))
        _, b = timed(lambda: sam.from_zoom(img, sod))
        inp = pipeline.load_inputs(args.sod_model, "DUTS-TE", "mask_zoom", id_)
        x, c = timed(lambda: torch.from_numpy(features(inp, 352))[None].cuda())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, d = timed(lambda: net(x))
        if i >= args.warmup:
            t["sam_plain"].append(a)
            t["sam_zoom"].append(b)
            t["head"].append(d)
            t["head_features"].append(c)
    res = {k: {"mean_ms": 1000 * float(np.mean(v)), "median_ms": 1000 * float(np.median(v))} for k, v in t.items()}
    res["gpu"] = torch.cuda.get_device_name(0)
    res["n"] = args.n
    print(json.dumps(res, indent=1))
    Path("results/timing_gpu.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
