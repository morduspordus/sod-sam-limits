"""Supplementary tables that back numbers of Sect. 5 of the paper that appear only in the text (second review).
Writes LaTeX to --out, in the style of review_tables.py:

  suppl_oracle_stats.tex      the per-image oracle of Table "oracle" (scripts/oracle.py, results/
                              oracle_<sod>_mask_zoom.json): images where a SAM mask beats the SOD map by > .02 in
                              per-image F, images where the oracle takes a SAM mask; DUTS-TE statistics of the
                              pixel-level oracle paragraph (scripts/granularity.py, results/granularity_<sod>_
                              mask_zoom_DUTS-TE.json; scripts/sliver_stats.py, results/sliver_stats.json)
  suppl_type_reduction.tex    per fine error type of the SOD map, the relative reduction of that type's loss by the
                              graph cut and the head (import/review/report_<sod>.json, review_eval.py --report)
  suppl_gain_type_ci.tex      bootstrap intervals of the boundary share of the gain (scripts/gain_type_ci.py,
                              results/gain_type_ci.json)
  suppl_oracle_auto_types.tex the added oracle gain of SAM's automatic masks by the SOD map's error type
                              (scripts/oracle_auto_types.py, results/oracle_auto_types_<sod>.json)
Tables whose inputs are missing are skipped.

  python scripts/suppl_stats_tables.py --out <dir>
"""
import argparse
import json
from pathlib import Path

from review_tables import BENCH, SODS, TYPES, f3, pct

ROOT = Path(__file__).resolve().parents[1]
ORACLE_RES = ROOT / "results"     # oracle.py results of the paper's Table "oracle"
FINE = [("boundary", "Boundary"), ("part_of_gt", "Part of GT"), ("whole_of_gt", "Whole of GT"),
        ("under_extent", "Under-extent"), ("over_extent", "Over-extent"), ("other_object", "Other object"),
        ("mixed", "Mixed")]
MIN_N, MIN_SHARE = 10, 0.02       # a type with fewer images or less of the SOD map's loss gets "--"


def spct(x):
    """Signed integer percentage with a math minus."""
    s = pct(abs(x))
    return ("$-$" if round(100 * x) < 0 else "") + s


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def oracle_stats_table():
    orc = {s: load(ORACLE_RES / f"oracle_{s}_mask_zoom.json") for s, _ in SODS}
    gran = {s: load(ROOT / "results" / f"granularity_{s}_mask_zoom_DUTS-TE.json") for s, _ in SODS}
    sliv = load(ROOT / "results" / "sliver_stats.json")
    if None in orc.values() or None in gran.values():
        return None
    out = ["\\begin{table}[t]",
           "\\caption{Statistics of the oracles of Sect.~5.1 of the paper (zoom-in prompt). Top: per benchmark, "
           "the share (\\%) of the images on which the best of SAM's three masks beats the SOD map by more than "
           ".02 in per-image $F$ (the maximum over thresholds of the image's $F$ curve), and on which the "
           "per-image oracle of Table~6 takes a SAM mask. Bottom, DUTS-TE: SAM's masks, sorted by area, are "
           "nested if each lies at least 90\\% inside the next larger one; area ratios: medians of middle/smallest "
           "and largest/middle; cells: 4-connected regions of constant signature of the binarized SOD map and the "
           "three masks; one-candidate oracle: the best per image of the binarized SOD map, the three masks and "
           "the fixed rule, against the per-image rule of Table~7; last rows: the per-image rule's gain in summed "
           "per-image $F$ by the error type of the SOD map (Sect.~5.4).}",
           "\\label{tab:suppl_oracle_stats}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}l" + "c" * len(BENCH) + "@{}}", "\\toprule",
           " & " + " & ".join(BENCH) + " \\\\", "\\midrule",
           f"\\multicolumn{{{len(BENCH) + 1}}}{{@{{}}l}}{{SAM mask beats the SOD map by $>$ .02 (\\% of images)}} \\\\"]
    for s, label in SODS:
        out.append(f"\\quad {label} & " + " & ".join(pct(orc[s][b]["sam_wins"]) for b in BENCH) + " \\\\")
    out.append(f"\\multicolumn{{{len(BENCH) + 1}}}{{@{{}}l}}{{Oracle takes a SAM mask (\\% of images)}} \\\\")
    for s, label in SODS:
        out.append(f"\\quad {label} & " + " & ".join(pct(orc[s][b]["picked_sam"]) for b in BENCH) + " \\\\")
    out += ["\\midrule", "DUTS-TE & " + " & ".join(label for _, label in SODS) + " & & & \\\\", "\\midrule"]

    def row(name, vals):
        out.append(f"{name} & " + " & ".join(vals) + " & & & \\\\")

    n = {s: gran[s]["n"] for s, _ in SODS}
    row("SAM's masks nested (\\%)", [pct(gran[s]["nested_share"]) for s, _ in SODS])
    row("Area ratio middle/smallest", [f"{gran[s]['median_area_ratio'][0]:.2f}" for s, _ in SODS])
    row("Area ratio largest/middle", [f"{gran[s]['median_area_ratio'][1]:.2f}" for s, _ in SODS])
    if sliv:
        row("Cells per image (mean)", [f"{sliv[s]['cells_mean']:.0f}" for s, _ in SODS])
        row("Cells under 50 px (\\%)", [pct(sliv[s]["small_share_pooled"]) for s, _ in SODS])
        row("\\quad their area (\\% of the image)", [f"{100 * sliv[s]['small_area_share']:.1f}" for s, _ in SODS])
    row("One-candidate oracle, \\% of rule gain", [pct(gran[s]["choose5_share_of_rule_image_gain"]) for s, _ in SODS])
    row("\\quad takes a SAM mask (\\% of images)",
        [pct(sum(gran[s]["best5"].get(f"SAM {k}", 0) for k in (1, 2, 3)) / n[s]) for s, _ in SODS])
    sh = {s: gran[s]["rule_image_gain_share_by_sod_type"] for s, _ in SODS}
    out.append("Per-image rule's gain by SOD error type (\\%) & & & & & \\\\")
    for key, name in [("boundary", "boundary"), ("part/whole", "part/whole"), ("extent", "extent"),
                      ("other object", "other object"), ("mixed", "mixed")]:
        row(f"\\quad {name}", [pct(sh[s].get(key, 0.0)) for s, _ in SODS])
    row("\\quad part/whole $+$ extent", [pct(sh[s]["part/whole"] + sh[s]["extent"]) for s, _ in SODS])
    out += ["\\botrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out) + "\n"


def type_reduction_table(reps):
    out = ["\\begin{table*}[t]",
           "\\caption{Relative reduction (\\%) of the loss $\\sum (1 - F)$ ($F$ at each map's $\\Fmax$ threshold) "
           "of the images of each error type of the SOD map (Sect.~5.4 of the paper) by the graph cut (GC) and "
           "the head, with the type's share (\\%) of the SOD map's loss (as in Table~\\ref{tab:suppl_gain_type}). "
           "Negative: the method makes these images worse; --: fewer than 10 images or less than 2\\% of the SOD "
           "map's loss.}",
           "\\label{tab:suppl_type_reduction}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
           "\\begin{tabular}{@{}ll" + "ccc" * len(BENCH) + "@{}}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{b}}}" for b in BENCH) + " \\\\",
           "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(BENCH))),
           "SOD model & SOD error type & " + " & ".join(["loss & GC & head"] * len(BENCH)) + " \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        for i, (cat, name) in enumerate(FINE):
            cells = []
            for b in BENCH:
                lt = reps[sod][b]["_loss_by_sod_type"]
                tot = sum(lt[c]["loss"]["sod"] for c in lt)
                if cat not in lt:
                    cells += ["0", "--", "--"]
                    continue
                l = lt[cat]["loss"]
                cells.append(pct(l["sod"] / tot))
                small = lt[cat]["n"] < MIN_N or l["sod"] < MIN_SHARE * tot
                cells += ["--" if small else spct((l["sod"] - l[m]) / l["sod"]) for m in ("gc", "head")]
            first = f"\\multirow{{{len(FINE)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {name} & " + " & ".join(cells) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def gain_type_ci_table(ci):
    out = ["\\begin{table*}[t]",
           f"\\caption{{Boundary share of the gain with 95\\% bootstrap intervals ({ci['boot']} resamples of the "
           "images of each benchmark; every map's $\\Fmax$ threshold is re-estimated on each resample). Red.: the "
           "method's reduction of the SOD map's total loss $\\sum (1 - F)$ (\\%); share: the part (\\%) of this "
           "reduction on the images whose SOD map has a boundary error (Table~\\ref{tab:suppl_gain_type}); "
           "$^{\\dagger}$: the upper end of the interval is below 50\\%.}",
           "\\label{tab:suppl_gain_type_ci}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}ll ccc ccc@{}}", "\\toprule",
           " & & \\multicolumn{3}{c}{Graph cut} & \\multicolumn{3}{c}{Fusion head} \\\\",
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
           "SOD model & Benchmark & red. & share & 95\\% interval & red. & share & 95\\% interval \\\\", "\\midrule"]
    for j, (sod, label) in enumerate(SODS):
        for i, b in enumerate(BENCH):
            cells = []
            for m in ("gc", "head"):
                r = ci[sod][b][m]
                lo, hi = r["ci"]
                wide = max(abs(lo), abs(hi)) > 9.995
                interval = "unbounded" if wide else f"[{spct(lo)}, {spct(hi)}]"
                cells += [spct(r["rel_reduction"]), spct(r["share"]) + ("$^{\\dagger}$" if hi < 0.5 else ""),
                          interval]
            first = f"\\multirow{{{len(BENCH)}}}{{*}}{{{label}}}" if i == 0 else ""
            out.append(f"{first} & {b} & " + " & ".join(cells) + " \\\\")
        if j < len(SODS) - 1:
            out.append("\\midrule")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def oracle_auto_types_table(res):
    types = list(TYPES)
    rows = [("sod_loss", "SOD loss"), ("prompt_gain", "Prompted oracle gain"),
            ("auto_gain_single", "$+$ automatic masks"), ("auto_gain_union", "$+$ their union")]
    out = ["\\begin{table*}[t]",
           "\\caption{Where the room of SAM's automatic masks lies (loose quality thresholds; "
           "Table~\\ref{tab:suppl_oracle_auto}). Per image, $F$ is the maximum over thresholds of the image's $F$ "
           "curve; the gains are sums over the images of the per-image $F$ gain of an oracle over the previous "
           "row's candidates (prompted oracle over the SOD map; $+$ automatic masks: best of the SOD map, the "
           "prompted and the single automatic masks over the prompted oracle; $+$ their union: the same with the "
           "greedy union of automatic masks). Mean: the gain (or the SOD map's loss $1 - F$) per image; the other "
           "columns: its share (\\%) in the images of each error type of the SOD map (Sect.~5.4 of the paper).}",
           "\\label{tab:suppl_oracle_auto_types}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{@{}lll c" + "c" * len(types) + "@{}}", "\\toprule",
           "SOD model & Benchmark & & Mean & " + " & ".join(types) + " \\\\"]
    for sod, label in SODS:
        if res.get(sod) is None:
            continue
        out.append("\\midrule")
        for i, b in enumerate(BENCH):
            d = res[sod]["datasets"][b]
            if i:
                out.append("\\cmidrule(l){2-" + str(4 + len(types)) + "}")
            for k, (key, name) in enumerate(rows):
                g = d[key]
                cells = [f3(g["total"] / d["n"])]
                cells += [spct(g["coarse"].get(t, 0.0) / g["total"]) if g["total"] > 0 else "--" for t in types]
                first = f"\\multirow{{{len(BENCH) * len(rows)}}}{{*}}{{{label}}}" if i == k == 0 else ""
                bcell = f"\\multirow{{{len(rows)}}}{{*}}{{{b}}}" if k == 0 else ""
                out.append(f"{first} & {bcell} & {name} & " + " & ".join(cells) + " \\\\")
    out += ["\\botrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reps = {s: json.loads((ROOT / "import" / "review" / f"report_{s}.json").read_text()) for s, _ in SODS}
    tabs = {"suppl_oracle_stats.tex": oracle_stats_table(),
            "suppl_type_reduction.tex": type_reduction_table(reps)}
    ci = load(ROOT / "results" / "gain_type_ci.json")
    if ci:
        tabs["suppl_gain_type_ci.tex"] = gain_type_ci_table(ci)
    auto = {s: load(ROOT / "results" / f"oracle_auto_types_{s}.json") for s, _ in SODS}
    if any(auto.values()):
        tabs["suppl_oracle_auto_types.tex"] = oracle_auto_types_table(auto)
    for name, tex in tabs.items():
        if tex:
            (out / name).write_text(tex)
    print("wrote", [name for name, tex in tabs.items() if tex])


if __name__ == "__main__":
    main()
