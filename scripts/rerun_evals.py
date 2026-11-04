"""Re-evaluate results/eval_<tag>.json files with the current metrics (sodsam/metrics.py), rerunning
fuse_eval.py with the arguments recorded in each file: SOD model, prompt, dense CRF parameters, and the
method keys; --neg-prompt / --sod-model2 come from the tune2 json of the v2 methods. The old file is kept in
results/eval_toolbox/ (once). Zero-shot files (no fusion methods) are not handled here.

  python scripts/rerun_evals.py --tags samba_zoom_5300 selfreformer_zoom_5300 --workers 14
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def command(tag, workers):
    d = json.loads((RES / f"eval_{tag}.json").read_text())
    methods = list(next(iter(d["datasets"].values())).keys())
    datasets = list(d["datasets"].keys())
    cmd = [sys.executable, str(ROOT / "scripts" / "fuse_eval.py"), "--sod-model", d["sod_model"], "--prompt",
           d["prompt"], "--workers", str(workers), "--tag", tag, "--datasets"] + datasets + ["--methods"] + methods
    extra = {}
    for m in methods:
        if m.startswith("v2:"):
            t = json.loads((ROOT / m[3:].split(":")[0]).read_text())
            a = t.get("args", t)
            for k in ("neg_prompt", "sod_model2"):
                if a.get(k):
                    if extra.get(k, a[k]) != a[k]:
                        raise ValueError(f"{tag}: methods disagree on {k}")
                    extra[k] = a[k]
    for k, v in extra.items():
        cmd += ["--" + k.replace("_", "-"), v]
    if d.get("crf_params"):
        p = RES / "eval_toolbox" / f"crf_{tag}.json"
        p.write_text(json.dumps({"crf_params": d["crf_params"]}))
        cmd += ["--crf-params", str(p)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    (RES / "eval_toolbox").mkdir(exist_ok=True)
    for tag in args.tags:
        cmd = command(tag, args.workers)
        print(" ".join(cmd[1:]), flush=True)
        if args.dry_run:
            continue
        old = RES / "eval_toolbox" / f"eval_{tag}.json"
        if not old.exists():
            shutil.copy(RES / f"eval_{tag}.json", old)
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        rows = [l for l in out.stdout.splitlines() if "F=" in l]
        print("\n".join(rows[-len(rows):]), flush=True)
        if out.returncode:
            print(f"FAILED {tag}:\n{out.stderr[-2000:]}", flush=True)


if __name__ == "__main__":
    main()
