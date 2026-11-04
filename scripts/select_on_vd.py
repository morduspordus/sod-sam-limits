"""Select one row on DIS-VD (best F^max) to be reported on DIS-TE1..4, from hr_eval.py results.

  python scripts/select_on_vd.py --sod-model isnet --rows gc_star_interior_g0_hard gc_star_strict_g0_hard ... \
      --out results/select_isnet_star_A.json
  python scripts/select_on_vd.py --sod-model isnet --family stitch_seam_
  python scripts/select_on_vd.py --sod-model isnet --check DIS-VD DIS-TE1 --rows sod gc_full_crops   # check only

Result files: results/eval_hr_<model>_<set>_<row><tag>.json, matched by exact name. --rows: exactly these
rows, and every one must be complete on the selection set (else exit 1). --family: the rows whose names
start with the prefix AND fully match a known row pattern (so files of another --tag, e.g. *_smoke.json,
never leak in); incomplete rows are listed but not eligible. Complete = missing == 0 and n == the number of
ids of the set (data.list_ids, the first --limit). Ties in F^max (exactly equal floats): with --rows the
row listed first wins (the job lists the beta rows in increasing beta, then the stage-A hard row, so the
weakest constraint that reaches the best F wins; in stage A interior before strict, gamma 0 before 4);
with --family the alphabetically first name. Prints the ranking, the reference rows (--ref, default
gc_full_crops) next to the selected one, and the selected row's test-set scores when they exist (an
incomplete test result is an error); --no-test skips the test sets entirely (the stage-A/B selections, so
stale test results cannot interfere). Writes --out (default results/select_<model>_<name>.json) with
"selected"; exits 1 when nothing can be selected. --check: only verify that the --rows results of the
given sets are complete (exit 1 otherwise).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sodsam import data, metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"
NUM = r"[0-9]+(?:\.[0-9]+)?"
KNOWN = re.compile(rf"sod|crops_majority|gc_full_(global|crops)|stitch_(avg|gauss|seam_l{NUM}_s{NUM})"
                   rf"|gc_star_(interior|strict|loose)_g{NUM}_(hard|b{NUM})")
KEYS = ("Fmax", "MAE", "E", "S", "n", "missing")


def load(res_dir, model, ds, row, tag):
    p = res_dir / f"eval_hr_{model}_{ds}_{row}{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def complete(r, n_full):
    return r is not None and r.get("missing", 0) == 0 and r["n"] == n_full


def fail(msg):
    print("SELECTION FAILED: " + msg, flush=True)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rows", nargs="+", help="exactly these rows")
    g.add_argument("--family", help="row-name prefix, e.g. stitch_seam_ or gc_star_")
    ap.add_argument("--check", nargs="+", default=None, metavar="SET", help="only check --rows on these sets")
    ap.add_argument("--vd", default="DIS-VD", help="the selection set")
    ap.add_argument("--test", nargs="+", default=["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"])
    ap.add_argument("--ref", nargs="*", default=["gc_full_crops"], help="rows printed next to the selected one")
    ap.add_argument("--require-test", action="store_true", help="fail unless the selected row is complete on --test")
    ap.add_argument("--no-test", action="store_true", help="do not read any test-set result (stage selections)")
    ap.add_argument("--tag", default="", help="hr_eval.py --tag of the result files")
    ap.add_argument("--limit", type=int, default=None, help="as hr_eval.py --limit (smoke tests)")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.no_test:
        if args.require_test:
            ap.error("--no-test and --require-test exclude each other")
        args.test = []

    res_dir = Path(args.results)
    n_of = {}

    def n_full(ds):
        if ds not in n_of:
            n_of[ds] = len(data.list_ids(ds)[:args.limit])
        return n_of[ds]

    if args.check:
        if not args.rows:
            ap.error("--check needs --rows")
        bad = []
        for ds in args.check:
            for row in args.rows:
                r = load(res_dir, args.sod_model, ds, row, args.tag)
                if not complete(r, n_full(ds)):
                    bad.append(f"{ds}/{row}: " + ("no result" if r is None else
                                                  f"n={r['n']} of {n_full(ds)}, missing={r.get('missing')}"))
        if bad:
            fail("incomplete evaluations:\n  " + "\n  ".join(bad))
        print(f"check ok: {len(args.rows)} rows complete on {' '.join(args.check)}")
        return

    if args.rows:
        rows = list(dict.fromkeys(args.rows))
        name = "rows"
    else:
        head = f"eval_hr_{args.sod_model}_{args.vd}_"
        tail = f"{args.tag}.json"
        rows = []
        for p in sorted(res_dir.glob(f"{head}{args.family}*{tail}")):
            row = p.name[len(head):len(p.name) - len(tail)]
            if row.startswith(args.family) and KNOWN.fullmatch(row):
                rows.append(row)
        name = args.family.rstrip("_")
    found = {row: load(res_dir, args.sod_model, args.vd, row, args.tag) for row in rows}
    nv = n_full(args.vd)
    if args.rows:
        bad = [r for r in rows if not complete(found[r], nv)]
        if bad:
            fail(f"rows not complete on {args.vd} (n = {nv}): {' '.join(bad)}")
    found = {k: v for k, v in found.items() if v is not None}
    if not found:
        fail(f"no results for {args.family or rows} on {args.vd}{args.tag}")
    # ties: the order of --rows (family rows are in alphabetical order)
    ranked = sorted(found, key=lambda k: (-found[k]["Fmax"], rows.index(k)))
    eligible = [k for k in ranked if complete(found[k], nv)]
    if not eligible:
        fail(f"no row is complete on {args.vd} (n = {nv})")
    best = eligible[0]
    print(f"{args.sod_model} {args.vd}: {len(found)} rows, n = {nv}")
    for k in ranked:
        flag = " <- selected" if k == best else ("" if k in eligible else " (incomplete, not eligible)")
        print("  " + metrics.format_row(k, found[k]) + flag)
    refs = {}
    for ref in args.ref:
        r = load(res_dir, args.sod_model, args.vd, ref, args.tag)
        refs[ref] = {args.vd: {k: r.get(k) for k in KEYS} if r else None}
        print("  ref " + (metrics.format_row(ref, r) + ("" if complete(r, nv) else " (INCOMPLETE)")
                          if r else f"{ref}: no result"))
    test, bad = {}, []
    for ds in args.test:
        r = load(res_dir, args.sod_model, ds, best, args.tag)
        if r is None:
            if args.require_test:
                bad.append(f"{ds}/{best}: no result")
            continue
        if not complete(r, n_full(ds)):
            bad.append(f"{ds}/{best}: n={r['n']} of {n_full(ds)}, missing={r.get('missing')}")
            continue
        test[ds] = {k: r.get(k) for k in KEYS}
        print("  test " + metrics.format_row(f"{ds}/{best}", r))
        for ref in args.ref:
            rr = load(res_dir, args.sod_model, ds, ref, args.tag)
            refs[ref][ds] = {k: rr.get(k) for k in KEYS} if rr else None
            print("   ref " + (metrics.format_row(f"{ds}/{ref}", rr) + ("" if complete(rr, n_full(ds)) else
                                                                     " (INCOMPLETE)") if rr else f"{ref}: no result"))
    out = Path(args.out) if args.out else res_dir / f"select_{args.sod_model}_{name}{args.tag}.json"
    out.write_text(json.dumps({"sod_model": args.sod_model, "rows": rows, "family": args.family,
                               "selection_set": args.vd, "selected": best,
                               "vd": {k: {m: found[k].get(m) for m in KEYS} for k in ranked},
                               "ref": refs, "test": test}, indent=1))
    if bad:
        fail("selected row incomplete on the test sets: " + "; ".join(bad))
    print(f"SELECTED {args.sod_model}: {best} (F={found[best]['Fmax']:.4f} on {args.vd}) -> {out}")


if __name__ == "__main__":
    main()
