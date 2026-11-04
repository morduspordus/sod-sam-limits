"""Set up the pseudo SOD model "<sod>_hzcut": the refinement head's maps in place of the SOD maps, with
the SAM masks of the original model, so that tune2.py / fuse_eval.py run the paper's cut on the head's output
without changes (item: graph cut on the head).

Links under OUT_ROOT (directory junctions on Windows, symlinks elsewhere), idempotent:
    <sod>_hzcut/<ds>/sod           -> <sod>_hz/<ds>/sod          (the head's soft maps)
    <sod>_hzcut/<ds>/sam_<prompt>  -> <sod>/<ds>/sam_<prompt>    (masks _1.._3, _p probabilities, scores.json)
pipeline.load_inputs / load_scores read nothing else from the model's directory; images and GT come from DATA_ROOT.
A missing source directory is an error (nothing is created for that dataset).

  python scripts/head_cut_setup.py --sod-models selfreformer samba
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sodsam import data


def is_link(p):
    return os.path.islink(p) or (hasattr(os.path, "isjunction") and os.path.isjunction(p))


def link_dir(link, target):
    """Create link -> target (absolute); skip an existing link to the same target, refuse anything else."""
    link, target = Path(link), Path(target)
    if not target.is_dir():
        raise FileNotFoundError(f"missing source directory {target}")
    if link.exists() or is_link(link):
        if is_link(link) and os.path.realpath(link) == os.path.realpath(target):
            return "kept"
        raise FileExistsError(f"{link} exists and is not a link to {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        os.symlink(target, link)
    return "linked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-models", nargs="+", default=["selfreformer", "samba"])
    ap.add_argument("--datasets", nargs="+", default=data.TUNE_DATASETS + data.EVAL_DATASETS)
    ap.add_argument("--prompt", default="mask_zoom")
    ap.add_argument("--head-suffix", default="hz", help="the head's maps are <sod>_<suffix>/<ds>/sod")
    ap.add_argument("--suffix", default="hzcut", help="name of the pseudo model: <sod>_<suffix>")
    args = ap.parse_args()

    for sod in args.sod_models:
        pseudo = f"{sod}_{args.suffix}"
        for ds in args.datasets:
            for sub, src in (("sod", data.run_dir(f"{sod}_{args.head_suffix}", ds, "sod")),
                             (f"sam_{args.prompt}", data.run_dir(sod, ds, f"sam_{args.prompt}"))):
                link = data.run_dir(pseudo, ds, sub)
                print(f"{link_dir(link, src):<7s} {link} -> {src}", flush=True)


if __name__ == "__main__":
    main()
