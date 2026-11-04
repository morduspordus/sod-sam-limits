"""Analyses of the per-image records of scripts/ideas6_eval.py.

  gate   per-image switch between the SOD map and the cut: "use the cut when dir * feature > t", the
         feature, direction and threshold chosen on the tuning sets (pooled F^max), applied to the test
         sets; also the per-image oracle switch (the better of the two at the test set's SOD-optimal
         threshold) as an upper bound
  blend  sod, cut, soft and the blends on the test sets; the output chosen on the tuning sets: the best S
         among those within 0.002 of the best tuning F^max
  thin   per test set: Spearman correlation of the per-image F gain of the cut (both binarized at 128)
         with the thin fraction (r = 2), the mean gain per thin-fraction tercile, and the pixels the cut
         fixes / breaks inside the thin region T and outside it (per radius)
  conv   per test set with "cv" records: sod, cut (tuned on the tuning sets), cv (tuned on the other half
         of the test set itself), oracle, and the fraction of the oracle headroom (oracle - cut) that cv
         closes

  python scripts/ideas6_report.py --sod-model samba --tune MSOD MSRA-B1K MSRA-B4K --test DUTS-TE ECSSD ...
Writes results/ideas6/report_<sod>.json and prints the tables.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from sodsam import metrics

OUT = Path(__file__).resolve().parents[1] / "results" / "ideas6"
I128 = 255 - 128          # f_curve index of threshold 128 (index 0 = threshold 255)


def load(sod_model, ds, suffix=""):
    with open(OUT / f"per_image_{sod_model}_{ds}{suffix}.pkl", "rb") as f:
        return pickle.load(f)["records"]


def agg(recs, key):
    return metrics.aggregate([r["maps"][key] for r in recs])


def pooled_fmax(curves):
    return float(np.mean(curves, axis=0).max())


def gate_fmax(recs, use_cut):
    return pooled_fmax([r["maps"]["cut" if c else "sod"]["f_curve"] for r, c in zip(recs, use_cut)])


def gate_maps(recs, use_cut):
    return metrics.aggregate([r["maps"]["cut" if c else "sod"] for r, c in zip(recs, use_cut)])


def fit_gate(recs, features):
    """(feature, direction, threshold, tuning F^max) maximizing the pooled F^max of the gated maps;
    the thresholds are the 0, 2, ..., 100 percentiles of the feature (and +-inf: always SOD / cut)."""
    best = (None, 1, np.inf, gate_fmax(recs, [False] * len(recs)))
    allcut = gate_fmax(recs, [True] * len(recs))
    if allcut > best[3]:
        best = (None, 1, -np.inf, allcut)
    for f in features:
        v = np.array([r["feats"][f] for r in recs], float)
        for d in (1, -1):
            for t in np.unique(np.percentile(d * v, np.arange(0, 101, 2))):
                fm = gate_fmax(recs, d * v > t)
                if fm > best[3] + 1e-12:
                    best = (f, d, float(t), fm)
    return best


def apply_gate(recs, gate):
    f, d, t, _ = gate
    if f is None:
        return [bool(t < 0)] * len(recs)
    return [d * r["feats"][f] > t for r in recs]


def oracle_gate(recs):
    i = int(np.argmax(np.mean([r["maps"]["sod"]["f_curve"] for r in recs], axis=0)))
    return [r["maps"]["cut"]["f_curve"][i] > r["maps"]["sod"]["f_curve"][i] for r in recs]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def row(name, m):
    return f"  {name:<14s} F={m['Fmax']:.4f} MAE={m['MAE']:.4f} E={m['E']:.4f} S={m['S']:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--tune", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--suffix", default="", help="file name suffix of the records (e.g. _lim3)")
    ap.add_argument("--test-suffix", default="", help="extra suffix of the test records (_cv: with the cv maps)")
    args = ap.parse_args()
    tune = [r for ds in args.tune for r in load(args.sod_model, ds, args.suffix)]
    test = {ds: load(args.sod_model, ds, args.test_suffix + args.suffix) for ds in args.test}
    feats = list(tune[0]["feats"])
    out = {"sod_model": args.sod_model, "tune": args.tune}

    gate = fit_gate(tune, feats)
    out["gate"] = {"feature": gate[0], "dir": gate[1], "t": gate[2], "tune_F": gate[3],
                   "tune_F_sod": agg(tune, "sod")["Fmax"], "tune_F_cut": agg(tune, "cut")["Fmax"], "test": {}}
    print(f"== gate ({args.sod_model}): chosen on {'+'.join(args.tune)}: {gate[0]} dir {gate[1]} t {gate[2]:.4g} | "
          f"tune F sod {out['gate']['tune_F_sod']:.4f} cut {out['gate']['tune_F_cut']:.4f} gate {gate[3]:.4f}")
    for ds, recs in test.items():
        use = apply_gate(recs, gate)
        orc = oracle_gate(recs)
        res = {"sod": agg(recs, "sod"), "cut": agg(recs, "cut"), "gate": gate_maps(recs, use),
               "oracle_gate": gate_maps(recs, orc), "frac_cut": float(np.mean(use)), "frac_cut_oracle": float(np.mean(orc))}
        out["gate"]["test"][ds] = res
        print(f" {ds} (gate uses the cut on {res['frac_cut']:.0%}, the oracle on {res['frac_cut_oracle']:.0%})")
        for k in ("sod", "cut", "gate", "oracle_gate"):
            print(row(k, res[k]))

    keys = [k for k in tune[0]["maps"] if k in ("cut", "soft") or k.startswith("blend")]
    tm = {k: agg(tune, k) for k in keys}
    fbest = max(m["Fmax"] for m in tm.values())
    chosen = max((k for k in keys if tm[k]["Fmax"] >= fbest - 0.002), key=lambda k: tm[k]["S"])
    out["blend"] = {"chosen": chosen, "tune": tm, "test": {}}
    print(f"== blend/soft ({args.sod_model}): chosen on the tuning sets: {chosen}")
    for ds, recs in test.items():
        out["blend"]["test"][ds] = {k: agg(recs, k) for k in ["sod"] + keys}
        print(f" {ds}")
        for k, m in out["blend"]["test"][ds].items():
            print(row(k + (" *" if k == chosen else ""), m))

    out["thin"] = {}
    print(f"== thin structures ({args.sod_model}): gain = F(cut) - F(SOD), both at threshold 128")
    for ds, recs in test.items():
        recs = [r for r in recs if not r["gt_empty"]]
        gain = np.array([r["maps"]["cut"]["f_curve"][I128] - r["maps"]["sod"]["f_curve"][I128] for r in recs])
        frac = np.array([r["thin"][2]["frac"] for r in recs])
        # terciles by rank (ties broken by order), so each image is in exactly one
        groups = np.array_split(np.argsort(frac, kind="stable"), 3)
        terc = [float(gain[g].mean()) for g in groups]
        edges = [float(frac[g].max()) for g in groups]
        res = {"spearman_gain_thin2": spearman(gain, frac), "gain_by_tercile": terc, "tercile_max_frac": edges,
               "mean_gain": float(gain.mean()), "per_r": {}}
        print(f" {ds}: mean gain {gain.mean():+.4f}; Spearman(gain, thin fraction r=2) {res['spearman_gain_thin2']:+.3f}; "
              f"gain by thin tercile {' '.join(f'{v:+.4f}' for v in terc)} (tercile max fractions "
              f"{' '.join(f'{v:.3f}' for v in edges)})")
        for r in recs[0]["thin"]:
            s = {k: int(sum(x["thin"][r][k] for x in recs)) for k in ("T_px", "fix_in", "break_in", "fix_out", "break_out")}
            res["per_r"][r] = s
            print(f"   r={r}: inside T ({s['T_px']} px) fixes {s['fix_in']} breaks {s['break_in']} (net {s['fix_in'] - s['break_in']:+d}); "
                  f"outside fixes {s['fix_out']} breaks {s['break_out']} (net {s['fix_out'] - s['break_out']:+d})")
        out["thin"][ds] = res

    out["conv"] = {}
    for ds, recs in test.items():
        if "cv" not in recs[0]["maps"]:
            continue
        m = {k: agg(recs, k) for k in ("sod", "cut", "cv", "oracle")}
        head = m["oracle"]["Fmax"] - m["cut"]["Fmax"]
        m["closed"] = (m["cv"]["Fmax"] - m["cut"]["Fmax"]) / head if head > 0 else float("nan")
        out["conv"][ds] = m
        print(f"== convention {ds} ({args.sod_model}): cv closes {m['closed']:.1%} of the oracle headroom")
        for k in ("sod", "cut", "cv", "oracle"):
            print(row(k, m[k]))

    (OUT / f"report_{args.sod_model}{args.test_suffix}{args.suffix}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
