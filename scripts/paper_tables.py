"""LaTeX tables of the rewritten manuscript, generated from the result files (so no number is typed
by hand). Missing inputs give "--" cells (or "pending" in the negative-results table) and a warning;
rerun when the runs finish.

  python scripts/paper_tables.py --out <paper>/tables \
      --results results

--results is the results directory of the main repository (eval_*.json, heads/, bootstrap/, timing_*.json,
oracle_*.json, error_types/, multi_answer/, selector/; its parent holds import/ for the zero-shot table).
--dev-results (default: results/ next to this script) holds the pixel oracle (pixel_oracle_*.json) and the
cluster evaluations of the post-processing variants (cluster/eval_*_ranker.json, _hzcut_retuned.json,
_soft.json). --print lists every number placed in the new tables with its source file and key.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"           # overridden by --results
DEV = ROOT / "results"           # overridden by --dev-results
# heads re-scored with the current metrics from their saved maps (rescore_maps.py --heads): plain prompt
# (headc_results) and zoom-in prompt (zoom_heads.sbatch, hz_results)
HEADS = RES / "heads"
HZ = RES / "heads"
DS = ["DUTS-TE", "ECSSD", "DUT-OMRON", "PASCAL-S", "HKU-IS"]
USED = []                        # (table, row, column, value, source) for --print
METRICS = [("Fmax", r"$\Fmax \uparrow$", True), ("MAE", r"$\MAE \downarrow$", False),
           ("E", r"$E_{s} \uparrow$", True), ("S", r"$S_{\alpha} \uparrow$", True)]
SOD_NAME = {"selfreformer": "SelfReformer~\\cite{yun2022selfreformer}", "samba": "Samba~\\cite{he2025samba}"}
MISSING = []


def load(path):
    p = Path(path)
    if not p.exists():
        MISSING.append(str(p))
        return None
    return json.loads(p.read_text())


def ev(name, method):
    """{dataset: metrics} of a method in results/eval_<name>.json (method: exact key or regex)."""
    d = load(RES / f"eval_{name}.json")
    if d is None:
        return {}
    out = {}
    for ds, x in d["datasets"].items():
        keys = [k for k in x if k == method] or [k for k in x if re.search(method, k)]
        if keys:
            out[ds] = x[keys[0]]
    if not out:
        MISSING.append(f"eval_{name}.json:{method}")
    return out


def head(tag, root=HEADS):
    d = load(root / f"{tag}.json")
    return d["test"] if d else {}


def head_mean(sod):
    """The paper's fusion head: the mean over its three training seeds (<sod>_hz, _hz_s1, _hz_s2) of each metric."""
    runs = [head(f"{sod}_hz{s}", HZ) for s in ("", "_s1", "_s2")]
    if not all(runs):
        return {}
    return {ds: {m: float(np.mean([r[ds][m] for r in runs])) for m in runs[0][ds] if m != "n"} for ds in runs[0]}


def fmt(v, key):
    return "--" if v is None else f"{v:.3f}"[1:] if v < 1 else f"{v:.3f}"


def cell(rows, name, ds, key):
    v = rows[name].get(ds, {}).get(key)
    return None if v is None else float(v)


def main_table(sod, rows, label, caption):
    """rows: {method name: {dataset: metrics}}; bold the best per metric and dataset."""
    lines = [r"\begin{table*}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             "Method & Metric & " + " & ".join(DS) + r" \\", r"\midrule"]
    names = list(rows)
    for n_i, name in enumerate(names):
        for m_i, (key, label_m, up) in enumerate(METRICS):
            vals = []
            for ds in DS:
                v = cell(rows, name, ds, key)
                col = [cell(rows, n, ds, key) for n in names]
                col = [c for c in col if c is not None]
                best = (max(col) if up else min(col)) if col else None
                s = fmt(v, key)
                if v is not None and best is not None and round(v, 3) == round(best, 3):
                    s = rf"\textbf{{{s}}}"
                vals.append(s)
            first = rf"\multirow{{4}}{{*}}{{\shortstack[l]{{{name}}}}}" if m_i == 0 else ""
            lines.append(f"{first} & {label_m} & " + " & ".join(vals) + r" \\")
        if n_i < len(names) - 1:
            lines.append(r"\midrule")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def f_table(rows, label, caption, key="Fmax", datasets=DS, groups=None):
    """Rows of F^max (or another metric) over datasets; groups: [(title, [row names])]."""
    groups = groups or [(None, list(rows))]
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}l" + "c" * len(datasets) + "@{}}", r"\toprule",
             "Variant & " + " & ".join(d.replace("DUT-OMRON", "OMRON").replace("PASCAL-S", "PASCAL") for d in datasets)
             + r" \\"]
    for title, names in groups:
        lines.append(r"\midrule")
        if title:
            lines.append(rf"\multicolumn{{{len(datasets) + 1}}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for n in names:
            shown = re.sub(r" \((SelfReformer|Samba)\)$", "", n)          # the group title names the model
            lines.append(shown + " & " + " & ".join(fmt(cell(rows, n, ds, key), key) for ds in datasets) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def error_table(models, label, caption):
    cats = [("boundary", ["boundary"]), ("part/whole", ["part_of_gt", "whole_of_gt"]),
            ("extent", ["under_extent", "over_extent"]), ("other object", ["other_object"]), ("mixed", ["mixed"])]
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}ll" + "c" * len(cats) + "@{}}", r"\toprule",
             "Dataset & SOD model & " + " & ".join(c for c, _ in cats) + r" \\"]
    for ds in ("DUTS-TE", "DUT-OMRON"):
        lines.append(r"\midrule")
        for i, (key, name) in enumerate(models):
            d = load(RES / "error_types" / f"{key}.json")
            if d is None:
                continue
            sh = d["summary"][ds]["share"]
            first = rf"\multirow{{{len(models)}}}{{*}}{{{ds}}}" if i == 0 else ""
            lines.append(f"{first} & {name} & " + " & ".join(f"{100 * sum(sh[c] for c in cs):.0f}\\%" for _, cs in cats)
                         + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def multi_table(label, caption):
    d = load(RES / "multi_answer" / "multi_answer.json")
    if d is None:
        return ""
    levels = [("GT", "GT only"), (">=0.25", r"$\geq\frac14$"), (">=0.15", r"$\geq 15\%$"), ("any", "any")]
    names = {"samba": "Samba", "selfreformer": "SelfReformer", "mdsam": "MDSAM", "vscode": "VSCode",
             "vstpp": "VST++", "human (LOSO)": "one person"}
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}ll" + "c" * len(levels) + "@{}}", r"\toprule",
             "Dataset & Predictor & " + " & ".join(t for _, t in levels) + r" \\"]
    for spec, title in (("PASCAL-S", "PASCAL-S"), ("MSOD:all", "SOD (all)"), ("MSOD:top", "SOD (top)")):
        lines.append(r"\midrule")
        rows = d[spec]
        for i, (k, v) in enumerate(rows.items()):
            first = rf"\multirow{{{len(rows)}}}{{*}}{{{title}}}" if i == 0 else ""
            cells = []
            for lv, _ in levels:
                x = v[lv]
                cells.append(fmt(x["Fmax"], "Fmax") + ("" if lv == "GT" else f" ({100 * x['F_err_recovered']:.0f}\\%)"))
            lines.append(f"{first} & {names.get(k, k)} & " + " & ".join(cells) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def zero_shot_table(label, caption):
    """SAM 2.1 alone on DUTS-TE (cluster job sam_zeroshot.sbatch, re-scored by zs2.sbatch -> import/zeroshot_v2/)."""
    zs = {s: load(RES.parent / "import" / "zeroshot_v2" / f"eval_{s}_zeroshot.json") for s in ("samba", "selfreformer")}
    get = lambda s, sub: (zs[s] or {}).get("datasets", {}).get("DUTS-TE", {}).get(sub, {})
    rows = [(f"Box, margin {m}", get("samba", f"sam_box_margin{m}")) for m in (0, 20, 40, 60, 80, 100)]
    rows += [("Center point", get("samba", "sam_center")), None,
             ("SelfReformer box", get("selfreformer", "sam_box_sod")), ("Samba box", get("samba", "sam_box_sod"))]
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Prompt & " + " & ".join(m for _, m, _ in METRICS) + r" \\", r"\midrule"]
    for r in rows:
        if r is None:
            lines.append(r"\midrule")
            continue
        name, x = r
        lines.append(name + " & " + " & ".join(fmt(x.get(k), k) for k, _, _ in METRICS) + r" \\")
    lines += [r"\botrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


SHORT = {"DUT-OMRON": "OMRON", "PASCAL-S": "PASCAL"}
MODELS = (("selfreformer", "SelfReformer"), ("samba", "Samba"))


def use(table, row, col, value, source):
    """Record a placed number for --print; returns the value."""
    USED.append((table, row, col, value, source))
    return value


def get(d, *keys):
    """Nested lookup that returns None for any missing level."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def table_head(cols, label, caption, star=False, first="Variant", sep="3pt"):
    env = "table*" if star else "table"
    return [rf"\begin{{{env}}}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
            rf"\setlength{{\tabcolsep}}{{{sep}}}", r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
            f"{first} & " + " & ".join(cols) + r" \\"], env


def table_foot(env):
    return [r"\botrule", r"\end{tabular}", rf"\end{{{env}}}"]


def samrefiner_table(gc_key, label, caption):
    """F^max of SAMRefiner's mask, of our methods on SAMRefiner's masks, and of our methods on SAM's masks."""
    lines, env = table_head([SHORT.get(d, d) for d in DS], label, caption)
    for sod, nm in MODELS:
        srf = load(RES / f"eval_maps_{sod}_samrefiner.json")
        rows = [("SOD map", ev(f"{sod}_final", "sod"), f"eval_{sod}_final.json:sod"),
                ("SAMRefiner", {ds: get(srf, "datasets", ds, "map") for ds in DS} if srf else {},
                 f"eval_maps_{sod}_samrefiner.json:map"),
                ("graph cut on SAMRefiner's masks", ev(f"{sod}_srf", r"mask_srf_linear_paper_binary_all"),
                 f"eval_{sod}_srf.json:v2:...mask_srf_linear_paper_binary_all"),
                ("fusion head on SAMRefiner's masks", head(f"{sod}_hsrf", HZ), f"heads/{sod}_hsrf.json:test"),
                ("graph cut", ev(f"{sod}_zoom_5300", gc_key), f"eval_{sod}_zoom_5300.json:{gc_key}"),
                ("fusion head (mean of 3 seeds)", head_mean(sod), f"heads/{sod}_hz*.json:test (mean)")]
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{len(DS) + 1}}}{{@{{}}l}}{{\emph{{{nm}}}}} \\")
        for name, r, src in rows:
            vals = [use("samrefiner", f"{name} ({nm})", ds, get(r, ds, "Fmax"), src) for ds in DS]
            lines.append(name + " & " + " & ".join(fmt(v, "Fmax") for v in vals) + r" \\")
    return "\n".join(lines + table_foot(env))


HR_DS = ["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4", "HRSOD-TE", "UHRSD-TE"]
# sod model -> {test set: tuning set}; see jobs/hr_cut.sbatch
HR_RUNS = {"birefnet": {d: "DIS-VD" for d in HR_DS},
           "isnet": {d: "DIS-VD" for d in HR_DS[:4]},
           "pgnet": {"HRSOD-TE": "UHRSD-TE", "UHRSD-TE": "HRSOD-TE"}}
HR_NAME = {"birefnet": "BiRefNet~\\cite{zheng2024birefnet}", "isnet": "IS-Net~\\cite{qin2022dis}",
           "pgnet": "PGNet~\\cite{xie2022pgnet}"}


HR_HEAD = {"Fmax": r"$\Fmax$", "MAE": r"$\MAE$", "E": r"$E_s$", "S": r"$S_\alpha$"}


def hr_table(label, caption, keys=("Fmax", "MAE")):
    """Two metrics (F^max and MAE by default) on the high-resolution benchmarks, scored at full resolution
    (results of jobs/hr_cut.sbatch, unpacked under <dev-results>/cluster/hr/results)."""
    hr = DEV / "cluster" / "hr" / "results"
    cols = [f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in HR_DS]
    lines, env = table_head(cols, label, caption, star=True, first="", sep="2pt")
    lines[5] = r"\begin{tabular}{@{}l" + "cc" * len(HR_DS) + "@{}}"
    lines.insert(4, r"\footnotesize")
    lines.append(" & " + " & ".join(" & ".join(HR_HEAD[k] for k in keys) for _ in HR_DS) + r" \\")
    for sod, runs in HR_RUNS.items():
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{2 * len(HR_DS) + 1}}}{{@{{}}l}}{{\emph{{{HR_NAME[sod]}}}}} \\")
        for name, sub in (("SOD map", lambda t: "sod"), ("SAM (most conf.)", lambda t: "sam_conf"),
                          ("graph cut", lambda t: f"gc_hr_{t}-1k")):
            cells = []
            for ds in HR_DS:
                if ds not in runs:
                    cells += ["", ""]
                    continue
                f = f"eval_hr_{sod}_{ds}-1k_{sub(runs[ds])}.json"
                d = load(hr / f)
                for key in keys:
                    v = use("hr", f"{name} ({sod})", f"{ds} {key}", get(d, key), f"dev:cluster/hr/results/{f}")
                    cells.append(fmt(v, key))
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + table_foot(env))


def bootstrap_table(label, caption):
    """Paired bootstrap of the F^max differences (results/bootstrap/<sod>.json: '<pair>/<dataset>')."""
    pairs = [("gc-sod", "graph cut $-$ SOD"), ("head-sod", "fusion head $-$ SOD"), ("head-gc", "fusion head $-$ graph cut")]
    # one row per (model, benchmark), one column per difference: narrow enough for the text width
    lines, env = table_head([n for _, n in pairs], label, caption, star=True, first="Benchmark", sep="5pt")
    lines.insert(4, r"\small")
    for sod, nm in MODELS:
        b = load(RES / "bootstrap" / f"{sod}.json") or {}
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{len(pairs) + 1}}}{{@{{}}l}}{{\emph{{{nm}}}}} \\")
        for ds in DS:
            cells = []
            for key, name in pairs:
                x = b.get(f"{key}/{ds}")
                if x is None:
                    MISSING.append(f"bootstrap/{sod}.json:{key}/{ds}")
                    cells.append("--")
                    continue
                lo, hi = x["F_ci"]
                use("bootstrap", f"{name} ({nm})", ds, (x["F_diff"], lo, hi), f"bootstrap/{sod}.json:{key}/{ds}:F_diff,F_ci")
                s = f"{sgn(x['F_diff'])} [{sgn(lo)}, {sgn(hi)}]"
                if lo > 0 or hi < 0:
                    s += r"$^{*}$"
                cells.append(s)
            lines.append(ds + " & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + table_foot(env))


def sgn(v):
    """Signed three-decimal number without the leading zero: +.007, -.003, .000."""
    s = f"{abs(v):.3f}"[1:]
    return ("$-$" if v < -0.0005 else "+" if v > 0.0005 else "") + s


def seeds_table(label, caption):
    """Mean +- sample standard deviation of F^max of the fusion head over three training seeds."""
    tags = ["hz", "hz_s1", "hz_s2"]
    lines, env = table_head([SHORT.get(d, d) for d in DS], label, caption, first="Fusion head")
    for sod, nm in MODELS:
        runs = [(t, head(f"{sod}_{t}", HZ)) for t in tags]
        runs = [(t, r) for t, r in runs if r]
        lines.append(r"\midrule")
        cells = []
        for ds in DS:
            vals = [get(r, ds, "Fmax") for _, r in runs]
            vals = [v for v in vals if v is not None]
            if len(vals) < 2:
                cells.append("--")
                continue
            m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
            use("seeds", f"{nm}", ds, (m, s, vals), f"heads/{sod}_{{hz,hz_s1,hz_s2}}.json:test:{ds}:Fmax")
            cells.append(f"{fmt(m, 'Fmax')} $\\pm$ {fmt(s, 'Fmax')}")
        lines.append(f"{nm} ({len(runs)} seeds) & " + " & ".join(cells) + r" \\")
    return "\n".join(lines + table_foot(env))


def runtime_table(label, caption_fmt):
    """Seconds per image from results/timing_gc.json (CPU) and results/timing_gpu.json (GPU)."""
    gc = load(RES / "timing_gc.json") or {}
    gpu = load(RES / "timing_gpu.json") or {}
    cpu_name = (gc.get("cpu") or "").split("\n")[0].strip() or r"\TODO{CPU model}"
    gpu_name = (gpu.get("gpu") or "").strip() or r"\TODO{GPU model}"
    cpu_name = re.sub(r"\(R\)|\(TM\)", "", cpu_name).replace("  ", " ").replace("CPU @", "@")
    size = gc.get("mean_size")
    size_s = f"{size[0]:.0f} $\\times$ {size[1]:.0f}" if size else r"\TODO{size}"
    rows = [("SAM~2.1, Hiera-L, plain prompt", "GPU", get(gpu, "sam_plain", "mean_ms"), get(gpu, "sam_plain", "median_ms"),
             "timing_gpu.json:sam_plain"),
            ("SAM~2.1, Hiera-L, zoom-in prompt", "GPU", get(gpu, "sam_zoom", "mean_ms"), get(gpu, "sam_zoom", "median_ms"),
             "timing_gpu.json:sam_zoom"),
            ("graph cut (max-flow)", "CPU", None if "mean_s" not in gc else gc["mean_s"] * 1000,
             None if "median_s" not in gc else gc["median_s"] * 1000, "timing_gc.json:mean_s,median_s"),
            ("fusion head, input assembly", "CPU$\\to$GPU", get(gpu, "head_features", "mean_ms"),
             get(gpu, "head_features", "median_ms"), "timing_gpu.json:head_features"),
            ("fusion head, forward pass", "GPU", get(gpu, "head", "mean_ms"), get(gpu, "head", "median_ms"),
             "timing_gpu.json:head")]
    caption = caption_fmt.format(cpu=cpu_name, gpu=gpu_name, n=gc.get("n", gpu.get("n", "?")), size=size_s)
    lines, env = table_head(["Device", "mean", "median"], label, caption, first="Stage")
    lines.append(r"\midrule")
    for name, dev, mean, med, src in rows:
        use("runtime", name, "mean_ms", mean, src)
        use("runtime", name, "median_ms", med, src)
        ms = lambda v: "--" if v is None else f"{v:.1f}"
        lines.append(f"{name} & {dev} & {ms(mean)} & {ms(med)} \\\\")
    return "\n".join(lines + table_foot(env))


def pixel_oracle_table(gc_key, label, caption):
    """Pixel-level oracle on DUTS-TE (dev results/pixel_oracle_<sod>_mask_zoom.json): F^max and wrong pixels."""
    cols = []
    for _, nm in MODELS:
        cols += [rf"\multicolumn{{2}}{{c}}{{{nm}}}"]
    lines = [r"\begin{table}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             "Candidates & " + " & ".join(cols) + r" \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             " & " + " & ".join([r"$\Fmax$", "wrong px"] * len(MODELS)) + r" \\", r"\midrule"]
    rows = [("SOD map, binarized", "sod"), ("SAM's three masks", "oracle_sam"), ("SOD map and SAM's masks", "oracle"),
            None,   # region-level oracles: the same four candidates, labels chosen per rule or per cell
            ("one fixed rule (16 cases) for the dataset", "oracle_rule_dataset"),
            ("one rule per image", "oracle_rule_image"),
            (r"cells $\geq 1\%$ of the image, else the rule", "oracle_region_1"),
            (r"cells $\geq 0.1\%$ of the image, else the rule", "oracle_region_0.1")]
    data = {sod: (load(DEV / f"pixel_oracle_{sod}_mask_zoom.json") or {}).get("DUTS-TE", {}) for sod, _ in MODELS}
    thr = {sod: get(data[sod], "sod_threshold") for sod, _ in MODELS}
    soft = {sod: get(data[sod], "sod_soft_Fmax") for sod, _ in MODELS}
    for row in rows:
        if row is None:
            lines.append(r"\midrule")
            continue
        name, key = row
        cells = []
        for sod, nm in MODELS:
            f = use("pixel_oracle", f"{name} ({nm})", "Fmax", get(data[sod], key, "Fmax"),
                    f"dev:pixel_oracle_{sod}_mask_zoom.json:DUTS-TE:{key}:Fmax")
            w = use("pixel_oracle", f"{name} ({nm})", "wrong_pixels", get(data[sod], key, "wrong_pixels"),
                    f"dev:pixel_oracle_{sod}_mask_zoom.json:DUTS-TE:{key}:wrong_pixels")
            cells += [fmt(f, "Fmax"), "--" if w is None else f"{100 * w:.2f}\\%"]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    cells = []
    for sod, nm in MODELS:
        g = ev(f"{sod}_zoom_5300", gc_key).get("DUTS-TE", {})
        f = use("pixel_oracle", f"graph cut ({nm})", "Fmax", g.get("Fmax"), f"eval_{sod}_zoom_5300.json:{gc_key}:DUTS-TE:Fmax")
        w = use("pixel_oracle", f"graph cut ({nm})", "wrong_pixels=MAE", g.get("MAE"), f"eval_{sod}_zoom_5300.json:{gc_key}:DUTS-TE:MAE")
        cells += [fmt(f, "Fmax"), "--" if w is None else f"{100 * w:.2f}\\%"]
    lines.append("graph cut (binary) & " + " & ".join(cells) + r" \\")
    for sod, nm in MODELS:
        use("pixel_oracle", f"SOD threshold ({nm})", "sod_threshold", thr[sod], f"dev:pixel_oracle_{sod}_mask_zoom.json:DUTS-TE:sod_threshold")
        use("pixel_oracle", f"SOD soft Fmax ({nm})", "sod_soft_Fmax", soft[sod], f"dev:pixel_oracle_{sod}_mask_zoom.json:DUTS-TE:sod_soft_Fmax")
    return "\n".join(lines + table_foot("table")), thr, soft


def negative_table(gc_key, label, caption, key="Fmax"):
    """Post-processing variants that did not help (`key`, F^max by default, on DUTS-TE / DUT-OMRON / PASCAL-S,
    both models). Missing values give 'pending' cells ('--' for the other metrics, which some rows lack)."""
    dss = ["DUTS-TE", "DUT-OMRON", "PASCAL-S"]

    def ds_key(path, *sub):
        d = load(path)
        if d is None:
            return None
        return {ds: get(d, "datasets", ds, *sub) for ds in dss}

    def plain(path):
        d = load(path)
        return None if d is None else {ds: get(d, ds) for ds in dss}

    def cut_of(path):
        """fuse_eval json: the one method that is not "sod" (its key holds the tune2 json path)."""
        d = load(path)
        if d is None:
            return None
        return {ds: next((v for k, v in (get(d, "datasets", ds) or {}).items() if k != "sod"), None) for ds in dss}

    variants = [
        ("Graph cut (reference)", lambda s: ev(f"{s}_zoom_5300", gc_key), lambda s: f"eval_{s}_zoom_5300.json:{gc_key}"),
        ("Expected-$F$ decision (SOD / SAM / cut)", lambda s: ds_key(RES / f"eval_{s}_idea_decision.json"), lambda s: f"eval_{s}_idea_decision.json:datasets"),
        ("Nested two-answer cut, outer mask", lambda s: ds_key(RES / f"eval_{s}_idea_nested.json", "outer"), lambda s: f"eval_{s}_idea_nested.json:datasets:outer"),
        ("Label-cost $\\alpha$-expansion (SAM auto masks)", lambda s: ds_key(RES / f"eval_{s}_labelcost.json", "labelcost"), lambda s: f"eval_{s}_labelcost.json:datasets:labelcost"),
        ("Scale-consistent cut (4 zoom levels)", lambda s: ds_key(RES / f"eval_{s}_idea_scales.json"), lambda s: f"eval_{s}_idea_scales.json:datasets"),
        ("Cut-in-the-loop, 2 rounds", lambda s: {ds: get(load(RES / f"cutloop_{s}.json"), ds, "x2") for ds in dss} if (RES / f"cutloop_{s}.json").exists() else None, lambda s: f"cutloop_{s}.json:x2"),
        ("Learned candidate ranker", lambda s: ds_key(DEV / "cluster" / f"eval_{s}_ranker.json", "ranker"), lambda s: f"dev:cluster/eval_{s}_ranker.json:datasets:ranker"),
        ("Graph cut on the head's map (retuned)", lambda s: cut_of(DEV / "cluster" / f"eval_{s}_hzcut_retuned.json"), lambda s: f"dev:cluster/eval_{s}_hzcut_retuned.json:datasets:v2"),
        ("Calibrated soft output", lambda s: ds_key(DEV / "cluster" / f"eval_{s}_soft.json", "soft_band"), lambda s: f"dev:cluster/eval_{s}_soft.json:datasets:soft_band"),
        ("Boundary-band constraint", lambda s: cut_of(DEV / "cluster" / f"eval_{s}_band.json"), lambda s: f"dev:cluster/eval_{s}_band.json:datasets:v2"),
        ("Confidence-binned parameters", lambda s: cut_of(DEV / "cluster" / f"eval_{s}_bins.json"), lambda s: f"dev:cluster/eval_{s}_bins.json:datasets:v2bins"),
        ("SAM-feature pairwise term", lambda s: cut_of(DEV / "cluster" / f"eval_{s}_feat.json"), lambda s: f"dev:cluster/eval_{s}_feat.json:datasets:v2"),
        ("Learned unary (table)", lambda s: cut_of(DEV / f"eval_{s}_lotable.json"), lambda s: f"dev:eval_{s}_lotable.json:datasets:v2"),
        ("Learned unary (GBT)", lambda s: cut_of(DEV / f"eval_{s}_logbt.json"), lambda s: f"dev:eval_{s}_logbt.json:datasets:v2"),
    ]
    cols = [rf"\multicolumn{{3}}{{c}}{{{nm}}}" for _, nm in MODELS]
    lines = [r"\begin{table*}[t]", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\centering", r"\footnotesize",
             r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             "Variant & " + " & ".join(cols) + r" \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             " & " + " & ".join([SHORT.get(d, d) for d in dss] * len(MODELS)) + r" \\", r"\midrule"]
    ref = {}
    pending = []
    worst = (-1.0, "", "", "")
    for i, (name, fn, src) in enumerate(variants):
        cells = []
        for sod, nm in MODELS:
            r = fn(sod)
            if r is None:
                r = {}
            if i == 0:
                ref[sod] = r
            for ds in dss:
                v = get(r, ds, key)
                if v is None:
                    cells.append("pending" if key == "Fmax" else "--")
                    pending.append(f"{name} ({nm}, {ds})")
                else:
                    use(f"negative_{key}", f"{name} ({nm})", ds, v, src(sod))
                    cells.append(fmt(v, key))
                    if key == "Fmax" and i > 0 and get(ref[sod], ds, "Fmax") is not None:
                        gain = round(v, 3) - round(ref[sod][ds]["Fmax"], 3)      # as the reader sees it
                        if gain > worst[0]:
                            worst = (gain, name, nm, ds)
        lines.append(name + " & " + " & ".join(cells) + r" \\")
        if i == 0:
            lines.append(r"\midrule")
    if key == "Fmax":
        print(f"negative table: largest gain over the graph cut is {worst[0]:+.4f} ({worst[1]}, {worst[2]}, {worst[3]})")
    short = sorted({re.sub(r" \(.*", "", p) for p in pending})
    if short:
        print("negative table pending rows:", "; ".join(sorted({p for p in pending})))
    return "\n".join(lines + table_foot("table*")), worst, short


def main():
    global RES, DEV, HEADS, HZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=str(RES), help="results/ of the main repository")
    ap.add_argument("--dev-results", default=str(DEV), help="results/ with pixel_oracle_*.json and cluster/")
    ap.add_argument("--print", action="store_true", help="list every number placed in the new tables with its source")
    args = ap.parse_args()
    RES = Path(args.results)
    DEV = Path(args.dev_results)
    HEADS = HZ = RES / "heads"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {}

    # the methods of the paper: zoom-in prompt; graph cut with the linear unary and the paper's pairwise
    # terms tuned on 5300 images; head trained on MSRA-B4K with the zoom masks
    gc = r"_mask_zoom_linear_paper_binary_all"
    for sod in ("selfreformer", "samba"):
        crf = ev("selfreformer_sod_crf" if sod == "selfreformer" else "samba_sod_crf", "sod_crf")
        rows = {"SOD": ev(f"{sod}_final", "sod"), "SOD + dense CRF": crf,
                r"SOD + SAM\\(most conf.\ mask)": ev(f"{sod}_zoom_naive", "sam_conf"),
                r"Graph cut\\(training-free)": ev(f"{sod}_zoom_5300", gc),
                r"Fusion head\\(learned)": head_mean(sod)}
        tabs[f"main_{sod}"] = main_table(
            sod, rows, f"tab:main_{sod}",
            f"Improving {SOD_NAME[sod]} with SAM~2.1. The graph cut is tuned and the fusion head is trained on "
            "images not used by the SOD model and disjoint from the benchmarks (Sect.~\\ref{sec:params}); the "
            "head's values are the means over three training seeds. The best result for each metric and dataset "
            "is in bold.")

    # head ablations: inputs and training set (F^max on all benchmarks)
    rows, groups = {}, []
    for sod, nm in (("selfreformer", "SelfReformer"), ("samba", "Samba")):
        names = []
        for tag, label, root in ((f"{sod}_hz", "full input", HZ),
                                 (f"{sod}_hz_nosam", "no SAM masks", HZ),
                                 (f"{sod}_hz_noimg", "no image", HZ),
                                 (f"{sod}_hz_nosod", "no SOD map", HZ),
                                 (f"{sod}_hz_dutstr", "trained on DUTS-TR", HZ),
                                 (f"{sod}_hsoft_s0", "plain prompt", HZ)):
            rows[f"{label} ({nm})"] = head(tag, root)
            names.append(f"{label} ({nm})")
        rows[f"SOD alone ({nm})"] = ev(f"{sod}_final", "sod")
        groups.append((nm, [f"SOD alone ({nm})"] + names))
    tabs["abl_head"] = f_table(rows, "tab:abl_head",
                               r"Fusion head ablations ($\Fmax$). Inputs removed from the full head (image, SOD map, "
                               "SAM masks and probabilities), the head trained on the SOD models' own training set "
                               "DUTS-TR instead of MSRA-B4K, and the head with the plain prompt "
                               "instead of the zoom-in prompt.", groups=groups)

    # graph-cut ablations: tuned on MSOD + MSRA-B1K (zoom-in prompt unless stated otherwise)
    rows, groups = {}, []
    for sod, nm in (("selfreformer", "SelfReformer"), ("samba", "Samba")):
        plain = f"{sod}_clean_2sets" if (RES / f"eval_{sod}_clean_2sets.json").exists() else f"{sod}_final"
        abl = f"{sod}_zoom_abl"
        r = {f"no pairwise terms ({nm})": ev(abl, "_mask_zoom_linear_none_"),
             f"Potts terms only ({nm})": ev(abl, "_mask_zoom_linear_potts_"),
             f"lookup-table unary ({nm})": ev(abl, "_mask_zoom_table_paper_"),
             f"full model ({nm})": ev(abl, "_mask_zoom_linear_paper_"),
             f"plain prompt ({nm})": ev(plain, "_linear_paper_binary_all"),
             f"plain prompt, HQ-SAM 2 ({nm})": ev(f"{sod}_hq", "_linear_paper_binary_all"),
             f"full model, tuned on 5167 ({nm})": ev(f"{sod}_zoom_5300", gc)}
        rows.update(r)
        groups.append((nm, list(r)))
    tabs["abl_gc"] = f_table(rows, "tab:abl_gc", r"Graph cut ablations ($\Fmax$). The full model has the linear "
                             "unary, the Potts and repulsive pairwise terms, and the zoom-in prompt; all rows are "
                             "tuned on MSOD + MSRA-B1K, except the last.", groups=groups)

    # oracle: how much could a per-image choice between SOD and SAM gain?
    rows, groups = {}, []
    for sod, nm in (("selfreformer", "SelfReformer"), ("samba", "Samba")):
        o = load(RES / f"oracle_{sod}_mask_zoom.json") or {}
        r = {f"SOD ({nm})": {ds: o[ds]["sod"] for ds in o},
             f"oracle: best SAM mask ({nm})": {ds: o[ds]["oracle_sam"] for ds in o},
             f"oracle: best of SOD and SAM ({nm})": {ds: o[ds]["oracle_all"] for ds in o},
             f"graph cut ({nm})": ev(f"{sod}_zoom_5300", gc), f"fusion head ({nm})": head_mean(sod)}
        rows.update(r)
        groups.append((nm, list(r)))
    tabs["oracle"] = f_table(rows, "tab:oracle", r"Oracle per-image selection ($\Fmax$): with the ground truth, "
                             "choose per image the SOD map or one of the three SAM masks.", groups=groups)

    tabs["errors"] = error_table([("selfreformer", "SelfReformer"), ("samba", "Samba"), ("birefnet", "BiRefNet"),
                                  ("mdsam", "MDSAM"), ("vscode", "VSCode"), ("vstpp", "VST++")], "tab:errors",
                                 "Where the remaining error of six SOD models lies: share of the total per-image "
                                 "$F$ loss by error type (Sect.~\\ref{sec:errors}).")
    tabs["multi"] = multi_table("tab:multi", r"Multi-answer evaluation ($\Fmax$, and the share of the error "
                                "explained in parentheses): each image is scored against the ground truth or any "
                                "answer given by at least the stated share of the subjects.")
    # supplementary: variants that did not help (plain prompt, tuned on MSOD + MSRA-B1K before the duplicate
    # removal, the protocol under which they were tried)
    for key, title in (("Fmax", r"$\Fmax$"), ("MAE", r"$\MAE$")):
        rows, groups = {}, []
        for sod, nm in (("selfreformer", "SelfReformer"), ("samba", "Samba")):
            sel = load(RES / "selector" / f"{sod}.json")
            r = {f"reference: linear unary ({nm})": ev(f"{sod}_final", "_linear_paper_binary_all"),
                 f"reference: table unary ({nm})": ev(f"{sod}_final", "_table_paper_binary_all"),
                 f"table + SOD confidence ({nm})": ev(f"{sod}_table_lin", "table_lin_paper"),
                 f"+ complement-prompt veto ({nm})": ev(f"{sod}_table_lin+veto", "veto"),
                 f"+ boundary band ({nm})": ev(f"{sod}_table_lin+band", "band"),
                 f"level-set tree ({nm})": ev(f"{sod}_tree", "tree"),
                 f"two SOD models ({nm})": ev(f"{sod}_duo", r"duo_paper_binary_all\.json$") if sod == "samba" else {},
                 f"two SOD models, no referee ({nm})": ev(f"{sod}_duo_plain", r"duo_plain_paper_binary_all\.json$")
                 if sod == "samba" else {},
                 f"soft output ({nm})": ev(f"{sod}_outputs_fixed", r"linear_paper_binary_all\.json:soft$"),
                 f"soft output + dense CRF ({nm})": ev(f"{sod}_outputs_fixed", r"linear_paper_binary_all\.json:soft_crf$"),
                 f"soft output + guided filter ({nm})": ev(f"{sod}_outputs", r"linear_paper_binary_all\.json:soft_gf$"),
                 f"per-image selector ({nm})": {ds: v["selector"] for ds, v in (sel or {}).get("datasets", {}).items()}}
            r = {k: v for k, v in r.items() if v}
            rows.update(r)
            groups.append((nm, list(r)))
        tabs[f"suppl_negative_{key}"] = f_table(
            rows, f"tab:suppl_negative_{key}", f"Variants that did not help ({title}). All graph cut variants use "
            "the plain mask prompt and are tuned on MSOD + MSRA-B1K.", key=key, groups=groups)

    tabs["zero_shot"] =zero_shot_table("tab:zero_shot", "SAM~2.1 alone on DUTS-TE with different prompts: the "
                                        "whole image shrunk by a margin (in pixels) as a box prompt, the image "
                                        "center as a point prompt, and the bounding box of an SOD model's mask "
                                        "as a box prompt.")
    # ---- tables added for the revision ----
    tabs["samrefiner"] = samrefiner_table(
        gc, "tab:samrefiner",
        r"SAMRefiner~\cite{lin2025samrefiner} as an alternative to prompted SAM ($\Fmax$). SAMRefiner (SAM ViT-H, "
        "default settings) refines the SOD map binarized at half its maximum; the next two rows apply our graph cut "
        "and fusion head to its masks instead of SAM's prompted masks; the last two rows are our methods as in "
        r"Tables~\ref{tab:main_selfreformer} and~\ref{tab:main_samba}.")
    tabs["bootstrap"] = bootstrap_table(
        "tab:bootstrap",
        r"Paired bootstrap over images (2000 resamples, the same resample for all methods) of the $\Fmax$ "
        r"differences between methods: mean "
        r"difference and 95\% confidence interval. $^{*}$: the interval excludes zero.")
    tabs["seeds"] = seeds_table(
        "tab:seeds",
        r"Seed variation of the fusion head: mean $\pm$ sample standard deviation of $\Fmax$ over three training "
        "runs with different random seeds (same data, same epoch selection).")
    tabs["runtime"] = runtime_table(
        "tab:runtime",
        "Runtime per image in milliseconds (mean and median over {n} DUTS-TE images of about {size} pixels) "
        "after the SOD map is computed. CPU: {cpu}; GPU: {gpu}.")
    tabs["pixel_oracle"], thr, soft = pixel_oracle_table(
        gc, "tab:pixel_oracle",
        r"Pixel-level oracle on DUTS-TE: each pixel takes the label of any correct candidate, so the result is wrong "
        r"only where every candidate is wrong. $\Fmax$ of the oracle masks and the fraction of wrong pixels "
        r"(for a binary mask, equal to $\MAE$). Lower block: the same four candidates, but the label is chosen "
        r"per rule from the four candidate labels (16 cases), per connected cell of constant candidate labels, "
        r"or per pixel. The SOD map is binarized at its best dataset threshold "
        "(" + ", ".join(f"{nm}: {thr_s}" for (_, nm), thr_s in zip(MODELS, ["{thr_sr}", "{thr_sa}"])) + ").")
    neg, worst, pending_rows = negative_table(
        gc, "tab:negative",
        r"Post-processing variants that did not help ($\Fmax$). All variants use the zoom-in prompt and are "
        r"tuned on MSOD + MSRA-B1K; the reference graph cut is tuned on all tuning images (Table~\ref{tab:abl_gc}); "
        r"none exceeds the graph cut by more than {worst} on any benchmark.")
    tabs["negative"] = neg.replace("{worst}", f"{worst[0]:.3f}"[1:] if worst[0] >= 0 else "0")
    tabs["hr"] = hr_table(
        "tab:hr",
        r"High-resolution benchmarks: DIS-TE1--4 of DIS5K~\cite{qin2022dis}, HRSOD-TE~\cite{zeng2019hrsod} and "
        r"UHRSD-TE~\cite{xie2022pgnet}. All methods run at a longest side of 1024 pixels and are scored at full "
        r"resolution. The graph cut is tuned on DIS-VD for BiRefNet and IS-Net; for PGNet, which has no "
        r"validation split with released maps, it is tuned on HRSOD-TE and tested on UHRSD-TE, and the other way "
        r"round. Empty cells: model not evaluated on that benchmark.")
    # supplementary: the other metrics of the two tables above (no \cite there: the supplement has no bibliography)
    tabs["suppl_hr_ES"] = hr_table(
        "tab:suppl_hr_ES",
        r"High-resolution benchmarks, $E_s$ and $S_\alpha$ (the other metrics of Table~7 of the paper; same "
        r"protocol). Empty cells: model not evaluated on that benchmark.", keys=("E", "S"))
    for key, name in (("MAE", r"$\MAE$, lower is better"), ("E", r"$E_s$"), ("S", r"$S_\alpha$")):
        tabs[f"suppl_variants_{key}"] = negative_table(
            gc, f"tab:suppl_variants_{key}",
            rf"Post-processing variants of Sect.~6.2 of the paper ({name}). Rows and tuning as in Table~13 of the "
            r"paper; --: not recorded for this variant.", key=key)[0]
    # fill the thresholds of the pixel-oracle caption
    for (sod, _), k in zip(MODELS, ["{thr_sr}", "{thr_sa}"]):
        t = thr.get(sod)
        tabs["pixel_oracle"] = tabs["pixel_oracle"].replace(k, "--" if t is None else f"{int(t)}/255")

    for k, v in tabs.items():
        (out / f"{k}.tex").write_text(v + "\n", encoding="utf-8")
    print(f"wrote {len(tabs)} tables to {out}")
    for m in sorted(set(MISSING)):
        print("MISSING", m)
    if args.print:
        print("---- numbers placed in the new tables (table | row | column | value | source) ----")
        for t, r, c, v, s in USED:
            if isinstance(v, tuple):
                vs = " ".join(f"{x:.4f}" if isinstance(x, float) else str([round(y, 4) for y in x]) for x in v)
            else:
                vs = "None" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            print(f"{t} | {r} | {c} | {vs} | {s}")


if __name__ == "__main__":
    main()
