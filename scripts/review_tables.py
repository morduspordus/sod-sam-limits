"""Supplementary tables of the second review, from the review_eval.py reports (import/review/report_<sod>.json)
and the error_robust.py results (results/error_robust/<model>.json). Writes LaTeX to --out.

  suppl_boundary.tex   Fmax, trimap IoU (5 px) and boundary F of the SOD map, SAM's most confident mask, the rule
                       "SOD mask or >= 2 SAM masks", the graph cut and the head, on the five benchmarks
  suppl_gain_type.tex  per error type of the SOD map: its share of the SOD map's loss (1 - F at each map's
                       Fmax threshold) and of the graph cut's and the head's reduction of that loss
  suppl_prompt.tex     SAM and the graph cut for each encoding of the mask prompt
  suppl_partc.tex      the graph cut for MDSAM, VSCode-S and VST++-B, with bootstrap intervals of its gain
  suppl_oracle_auto.tex per-image oracles with SAM's automatic masks (scripts/oracle_auto.py)
  suppl_fixed_f.tex    F at the fixed threshold 0.5 and adaptive F of the SOD map, the cut and the head (fixed_f.py)
  suppl_err_robust.tex the error decomposition of the six SOD models under other weightings, thresholds, and
                       at the pixel level

  python scripts/review_tables.py --out <paper>/tables
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_RED = 0.02
BENCH = ["DUTS-TE", "ECSSD", "DUT-OMRON", "PASCAL-S", "HKU-IS"]
SODS = [("selfreformer", "SelfReformer"), ("samba", "Samba")]
TYPES = {"Boundary": ("boundary",), "Part/whole": ("part_of_gt", "whole_of_gt"),
         "Extent": ("under_extent", "over_extent"), "Other object": ("other_object",), "Mixed": ("mixed",)}
MODELS = [("selfreformer", "SelfReformer"), ("samba", "Samba"), ("birefnet", "BiRefNet"), ("mdsam", "MDSAM"),
          ("vscode", "VSCode"), ("vstpp", "VST++")]


def f3(x):
    return f"{x:.3f}"[1:] if 0 <= x < 1 else f"{x:.3f}"


def sgn(x):
    """Signed difference as in tables/bootstrap.tex: +.006, $-$.001, four decimals where three round to .000."""
    s = f"{abs(x):.3f}" if round(abs(x), 3) > 0 else f"{abs(x):.4f}"
    return ("$-$" if x < 0 else "+") + s[1:]


def pct(x):
    return f"{100 * x:.0f}"


def boundary_table(reps):
    rows = [("sod", "SOD map"), ("sam_conf", "SAM, most conf.\\ mask"), ("rule_major", "SOD or $\\geq 2$ SAM masks"),
            ("gc", "Graph cut"), ("head", "Fusion head")]
    out = ["\\begin{table*}[t]",
           "\\caption{Boundary measures on the five benchmarks (Sect.~5.3 of the paper). Each map is binarized at "
           "its own $\\Fmax$ threshold; tIoU: IoU within 5 pixels of the ground-truth contour; BF: boundary "
           "F-measure with a 3-pixel tolerance. The rule ``SOD or $\\geq 2$ SAM masks'' uses the SOD mask "
           "$s \\geq 0.5$ (Table~8 of the paper).}",
           "\\label{tab:suppl_boundary}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2pt}",
           "\\begin{tabular}{@{}ll" + "ccc" * len(BENCH) + "@{}}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{b}}}" for b in BENCH) + " \\\\",
           "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(BENCH))),
           "SOD model & Map & " + " & ".join(["$\\Fmax$ & tIoU & BF"] * len(BENCH)) + " \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        for i, (key, name) in enumerate(rows):
            cells = []
            for b in BENCH:
                m = reps[sod][b][key]
                cells += [f3(m["Fmax"]), f3(m["tiou5"]), f3(m["bf"])]
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {name} & " + " & ".join(cells) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def gain_table(reps):
    out = ["\\begin{table*}[t]",
           "\\caption{Where the gain comes from, by the error type of the SOD map (Sect.~5.3 of the paper; types "
           "as in Sect.~5.4). For each type: the share (\\%) of the SOD map's total loss $\\sum (1 - F)$, with $F$ "
           "at each map's $\\Fmax$ threshold, and the share of the graph cut's (GC) and the head's reduction of "
           "that loss. Negative: the method makes these images worse; --: the method reduces the total loss by "
           "less than 2\\%, too little for its shares to be meaningful.}",
           "\\label{tab:suppl_gain_type}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{tabular}{@{}ll" + "ccc" * len(BENCH) + "@{}}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{b}}}" for b in BENCH) + " \\\\",
           "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(BENCH))),
           "SOD model & SOD error type & " + " & ".join(["loss & GC & head"] * len(BENCH)) + " \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        for i, (tname, cats) in enumerate(TYPES.items()):
            cells = []
            for b in BENCH:
                lt = reps[sod][b]["_loss_by_sod_type"]
                tot = {m: sum(lt[c]["loss"][m] for c in lt) for m in ("sod", "gc", "head")}
                part = {m: sum(lt[c]["loss"][m] for c in cats if c in lt) for m in ("sod", "gc", "head")}
                cells.append(pct(part["sod"] / tot["sod"]))
                for m in ("gc", "head"):
                    red = tot["sod"] - tot[m]
                    # a reduction below 2% of the SOD map's loss leaves the shares to noise
                    cells.append(pct((part["sod"] - part[m]) / red) if red > MIN_RED * tot["sod"] else "--")
            first = f"\\multirow{{{len(TYPES)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {tname} & " + " & ".join(cells) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def robust_table(rob):
    groups = ["boundary", "part/whole", "extent", "other object", "mixed"]
    out = ["\\begin{table*}[t]",
           "\\caption{Robustness of the error decomposition (Table~9 of the paper). Per dataset and SOD model: "
           "the share (\\%) of the error of each type with the paper's loss weighting (1 $-$ per-image $F$), with "
           "each image weighted by its fraction of wrong pixels instead, and the range of the boundary share over "
           "a grid of the taxonomy's thresholds (boundary IoU 0.75--0.90, containment 0.8--0.95, part/whole ratio "
           "1.5--3, lower extent ratio 1.05--1.3; 108 settings). Last two columns: the share of all wrong pixels "
           "of the thresholded SOD mask (pooled over the dataset) within 5 and 10 pixels of the ground-truth "
           "contour, an upper bound on the pixel error that a better outline alone could remove.}",
           "\\label{tab:suppl_err_robust}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}ll ccccc ccccc c cc@{}}", "\\toprule",
           " & & \\multicolumn{5}{c}{Loss-weighted (paper)} & \\multicolumn{5}{c}{Wrong-pixel-weighted} & "
           "Boundary & \\multicolumn{2}{c}{Pixels near contour} \\\\",
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}\\cmidrule(lr){13-13}\\cmidrule(lr){14-15}",
           "Dataset & Model & bnd & p/w & ext & oth & mix & bnd & p/w & ext & oth & mix & grid range & $\\le$5 px "
           "& $\\le$10 px \\\\", "\\midrule"]
    for j, ds in enumerate(["DUTS-TE", "DUT-OMRON"]):
        for i, (key, name) in enumerate(MODELS):
            r = rob[key][ds]
            lo, hi = r["grid_range"]["boundary"]
            cells = [pct(r["image"]["loss"][g]) for g in groups] + [pct(r["image"]["wrong_frac"][g]) for g in groups]
            cells += [f"{pct(lo)}--{pct(hi)}", pct(r["pixel"]["<=5px"]["pooled"]), pct(r["pixel"]["<=10px"]["pooled"])]
            first = f"\\multirow{{{len(MODELS)}}}{{*}}{{{ds}}}" if i == 0 else ""
            out.append(f"{first} & {name} & " + " & ".join(cells) + " \\\\")
        if j == 0:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def stability_table():
    """The paper's graph cut against its tuning repeated with two other random seeds (import/review/
    eval_seed<k>_<sod>.json) and against two-fold tuning on the benchmarks themselves (import/ideas6/
    report_<sod>_cv.json, "conv": tuned on one half of each benchmark, evaluated on the other, both ways)."""
    out = ["\\begin{table*}[t]",
           "\\caption{Stability of the graph cut's tuning ($\\Fmax$, four decimals; Sect.~4.3 of the paper). "
           "Paper: tuned on MSOD + MSRA-B1K + MSRA-B4K (Tables~1 and~2 of the paper); seed 2, seed 3: the same "
           "random search from scratch with two other seeds; two-fold: tuned on one half of each benchmark and "
           "evaluated on the other half, and the other way round, so that the parameters follow the benchmark's "
           "own annotation conventions.}",
           "\\label{tab:suppl_stability}", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{tabular}{@{}ll" + "c" * len(BENCH) + "@{}}", "\\toprule",
           "SOD model & Tuning & " + " & ".join(BENCH) + " \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        seeds = [json.loads((ROOT / "import" / "review" / f"eval_seed{k}_{sod}.json").read_text())["datasets"]
                 for k in (1, 2)]
        conv = json.loads((ROOT / "import" / "ideas6" / f"report_{sod}_cv.json").read_text())["conv"]
        def pick(d, ds, suffix):
            return next(v["Fmax"] for k, v in d[ds].items() if k.endswith(suffix))
        rows = [("Paper", [conv[b]["cut"]["Fmax"] for b in BENCH]),
                ("Seed 2", [pick(seeds[0], b, "_seed1.json") for b in BENCH]),
                ("Seed 3", [pick(seeds[1], b, "_seed2.json") for b in BENCH]),
                ("Two-fold on benchmark", [conv[b]["cv"]["Fmax"] for b in BENCH])]
        for i, (name, vals) in enumerate(rows):
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {name} & " + " & ".join(f"{v:.4f}"[1:] for v in vals) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def prompt_table():
    """SAM's most confident mask and the graph cut (tuned on MSOD + MSRA-B1K) for each encoding of the mask
    prompt, from import/prompt_enc/eval_penc_<sod>_<prompt>.json (jobs/prompt_cut.sbatch)."""
    encs = [("mask_zoom", "Soft (paper): $0, \\dots, 255$"), ("mask_zoom_binary", "Binary: $\\pm 10$"),
            ("mask_zoom_logit", "Logit: $2 L(s)$"), ("mask_zoom_signed", "Signed: $10 (2s - 1)$"),
            ("mask_zoom_pts", "Soft + 2 points")]
    out = ["\\begin{table*}[t]",
           "\\caption{Encoding of the mask prompt ($\\Fmax$, zoom-in prompt; Sect.~3.1 of the paper). SAM: its "
           "most confident mask; graph cut: tuned on MSOD + MSRA-B1K for each encoding. Binary: logit $+10$ where "
           "$s \\geq 0.5$ and $-10$ elsewhere; logit: twice the logit $L(s)$ of the saliency, with $s$ clipped "
           "to $[0.01, 0.99]$; signed: a linear map of $s$ to $[-10, 10]$; soft + 2 points: the paper's "
           "prompt with a positive point at the interior point of the SOD mask farthest from its boundary and a "
           "negative point at the background point farthest from the mask.}",
           "\\label{tab:suppl_prompt}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}lll" + "c" * len(BENCH) + "@{}}", "\\toprule",
           "SOD model & Prompt & Map & " + " & ".join(BENCH) + " \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        for i, (enc, name) in enumerate(encs):
            d = json.loads((ROOT / "import" / "prompt_enc" / f"eval_penc_{sod}_{enc}.json").read_text())["datasets"]
            for k, (meth, mname) in enumerate([("sam_conf", "SAM"), ("v2:", "Graph cut")]):
                vals = [next(v["Fmax"] for key, v in d[b].items() if key.startswith(meth)) for b in BENCH]
                first = f"\\multirow{{{2 * len(encs)}}}{{*}}{{{label}}}" if i == k == 0 else ""
                enc_cell = f"\\multirow{{2}}{{*}}{{{name}}}" if k == 0 else ""
                out.append(f"{first} & {enc_cell} & {mname} & " + " & ".join(f3(v) for v in vals) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def partc_table():
    """The paper's graph cut for three more SOD models, tuned exactly as in the paper, from import/partc/
    eval_partc_<model>.json and the paired bootstrap of the cut against the SOD map, partc_<model>.json
    (jobs/partc_cut.sbatch)."""
    models = [("mdsam", "MDSAM"), ("vscode", "VSCode-S"), ("vstpp", "VST++-B")]
    out = ["\\begin{table*}[t]",
           "\\caption{The graph cut for three more SOD models ($\\Fmax$ and $\\MAE$, zoom-in prompt), tuned as in the paper "
           "on MSOD, MSRA-B1K, and MSRA-B4K. SAM: its most confident mask. Gain: the $\\Fmax$ gain of the graph "
           "cut over the SOD map, with its 95\\% paired bootstrap interval in the next row (as in Table~\\ref{tab:bootstrap}); "
           "$^{*}$: the interval excludes zero. Last row: $\\MAE$ of the SOD map and of the graph cut. The maps are "
           "computed with the released weights; those of VST++-B are not identical to the released maps "
           "(see the text).}",
           "\\label{tab:suppl_partc}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}ll" + "c" * len(BENCH) + "@{}}", "\\toprule",
           "SOD model & Map & " + " & ".join(BENCH) + " \\\\", "\\midrule"]
    for j, (m, label) in enumerate(models):
        d = json.loads((ROOT / "import" / "partc" / f"eval_partc_{m}.json").read_text())["datasets"]
        bs = json.loads((ROOT / "import" / "partc" / f"partc_{m}.json").read_text())
        for k, (meth, mname) in enumerate([("sod", "SOD map"), ("sam_conf", "SAM"), ("v2:", "Graph cut")]):
            vals = [next(v["Fmax"] for key, v in d[b].items() if key.startswith(meth)) for b in BENCH]
            first = f"\\multirow{{6}}{{*}}{{{label}}}" if k == 0 else ""
            out.append(f"{first} & {mname} & " + " & ".join(f3(v) for v in vals) + " \\\\")
        gains, cis = [], []
        for b in BENCH:
            r = bs[f"gc-sod/{b}"]
            lo, hi = r["F_ci"]
            gains.append(sgn(r["F_diff"]) + ("$^{*}$" if lo > 0 or hi < 0 else ""))
            cis.append(f"{{\\scriptsize [{sgn(lo)}, {sgn(hi)}]}}")
        out.append(" & Gain & " + " & ".join(gains) + " \\\\")
        out.append(" & {\\scriptsize 95\\% interval} & " + " & ".join(cis) + " \\\\")
        mae = [[next(v["MAE"] for key, v in d[b].items() if key.startswith(meth)) for b in BENCH]
               for meth in ("sod", "v2:")]
        out.append(" & $\\MAE$ SOD / cut & " + " & ".join(f"{f3(a)} / {f3(c)}" for a, c in zip(*mae)) + " \\\\")
        if j < len(models) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def oracle_auto_table():
    """Per-image oracles with SAM's automatic masks (scripts/oracle_auto.py, results/oracle_<auto>_<sod>_mask_zoom.json,
    auto = default and loose quality thresholds)."""
    res = {(a, s): json.loads((ROOT / "results" / f"oracle_{a}_{s}_mask_zoom.json").read_text())
           for a in ("auto", "auto_loose") for s, _ in SODS}
    n_masks = {a: [res[(a, "samba")][b]["n_auto_mean"] for b in BENCH] for a in ("auto", "auto_loose")}
    out = ["\\begin{table*}[t]",
           "\\caption{Per-image oracles with SAM's automatic (``everything'') masks ($\\Fmax$): with the ground truth, "
           "choose per image the best candidate. Automatic masks: SAM~2.1 with a $32 \\times 32$ point grid, masks of "
           "0.2\\% to 95\\% of the image, with SAM's default quality thresholds (predicted IoU $\\geq 0.8$, stability "
           f"$\\geq 0.95$; {min(n_masks['auto']):.0f} to {max(n_masks['auto']):.0f} masks per image) or looser ones "
           f"(0.6, 0.85; {min(n_masks['auto_loose']):.0f} to {max(n_masks['auto_loose']):.0f} masks per image). Union: "
           "greedy union of automatic masks, adding the mask that raises $F$ most until none does. The second row of "
           "each SOD model is the oracle of Table~6 of the paper.}",
           "\\label{tab:suppl_oracle_auto}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}ll" + "c" * len(BENCH) + "@{}}", "\\toprule",
           " & Candidates & " + " & ".join(BENCH) + " \\\\", "\\midrule"]
    rows = [("auto", "auto_single", "best automatic mask (default)"), ("auto", "auto_union", "union of automatic masks (default)"),
            ("auto_loose", "auto_single", "best automatic mask (loose)"),
            ("auto_loose", "auto_union", "union of automatic masks (loose)")]
    for i, (a, k, name) in enumerate(rows):
        first = f"\\multirow{{{len(rows)}}}{{*}}{{SAM alone}}" if i == 0 else ""
        out.append(f"{first} & {name} & " + " & ".join(f3(res[(a, 'samba')][b][k]['Fmax']) for b in BENCH) + " \\\\")
    rows = [("auto", "sod", "SOD map"), ("auto", "oracle_prompt", "SOD map, prompted masks"),
            ("auto", "all_single", "$+$ automatic masks (default)"), ("auto", "all_union", "$+$ their union"),
            ("auto_loose", "all_single", "$+$ automatic masks (loose)"), ("auto_loose", "all_union", "$+$ their union")]
    for s, label in SODS:
        out.append("\\midrule")
        for i, (a, k, name) in enumerate(rows):
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {name} & " + " & ".join(f3(res[(a, s)][b][k]['Fmax']) for b in BENCH) + " \\\\")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def fixed_f_table():
    """F^max, F at the fixed threshold 128/255 and adaptive F of the SOD map, the graph cut and the head
    (scripts/fixed_f.py, results/fixed_f_<sod>.json), with paired bootstrap differences at the fixed threshold."""
    maps = [("sod", "SOD map"), ("gc", "Graph cut"), ("head", "Fusion head")]
    meas = [("Fmax", "$\\Fmax$"), ("F05", "$F_\\beta$ at 0.5"), ("Fadp", "adaptive $F_\\beta$")]
    diffs = [("gc-sod", "Graph cut $-$ SOD"), ("head-sod", "Head $-$ SOD"), ("head-gc", "Head $-$ graph cut")]
    out = ["\\begin{table*}[t]",
           "\\caption{$F_\\beta$ at thresholds that do not favor soft maps. $\\Fmax$: the best threshold for each "
           "map and dataset (the paper's measure); $F_\\beta$ at 0.5: every map binarized at half its range; "
           "adaptive $F_\\beta$: binarized at twice its mean, as in PySODMetrics. The graph cut is binary, so its three "
           "values agree up to the min-max normalization. Last rows: differences of $F_\\beta$ at 0.5; $^{*}$: the "
           "95\\% paired bootstrap interval excludes zero.}",
           "\\label{tab:suppl_fixed_f}", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}lll" + "c" * len(BENCH) + "@{}}", "\\toprule",
           "SOD model & Measure & Map & " + " & ".join(BENCH) + " \\\\"]
    for s, label in SODS:
        r = json.loads((ROOT / "results" / f"fixed_f_{s}.json").read_text())
        out.append("\\midrule")
        n_rows = len(meas) * len(maps) + len(diffs)
        for i, (mk, mname) in enumerate(meas):
            for j, (k, name) in enumerate(maps):
                first = f"\\multirow{{{n_rows}}}{{*}}{{{label}}}" if i == j == 0 else ""
                mcell = f"\\multirow{{{len(maps)}}}{{*}}{{{mname}}}" if j == 0 else ""
                out.append(f"{first} & {mcell} & {name} & " + " & ".join(f3(r[b][k][mk]) for b in BENCH) + " \\\\")
        for j, (k, name) in enumerate(diffs):
            mcell = f"\\multirow{{{len(diffs)}}}{{*}}{{at 0.5}}" if j == 0 else ""
            cells = []
            for b in BENCH:
                d, lo, hi = r[b]["ci_F05"][k]
                cells.append(sgn(d) + ("$^{*}$" if lo > 0 or hi < 0 else ""))
            out.append(f" & {mcell} & {name} & " + " & ".join(cells) + " \\\\")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    reps = {s: json.loads((ROOT / "import" / "review" / f"report_{s}.json").read_text()) for s, _ in SODS}
    rob = {m: json.loads((ROOT / "results" / "error_robust" / f"{m}.json").read_text()) for m, _ in MODELS}
    (out / "suppl_boundary.tex").write_text(boundary_table(reps))
    (out / "suppl_gain_type.tex").write_text(gain_table(reps))
    (out / "suppl_err_robust.tex").write_text(robust_table(rob))
    (out / "suppl_stability.tex").write_text(stability_table())
    (out / "suppl_prompt.tex").write_text(prompt_table())
    (out / "suppl_partc.tex").write_text(partc_table())
    (out / "suppl_oracle_auto.tex").write_text(oracle_auto_table())
    (out / "suppl_fixed_f.tex").write_text(fixed_f_table())
    print("wrote", [p.name for p in out.glob("suppl_*.tex")][:20])


if __name__ == "__main__":
    main()
