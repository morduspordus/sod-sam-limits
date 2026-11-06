"""Supplementary table that backs the multi-annotator numbers of Sect. 5.5 of the paper ("Are these errors choices
that people also make?"), which appear only in the text. Writes LaTeX in the style of review_tables.py:

  suppl_multi_stats.tex
    MSOD, people: part vs whole between pairs of different subjects whose sessions used the SAME BSDS segmentation
        (SES.index), most salient object (IMP 1) only; nested = smaller >= 90% inside larger and larger >= 2x smaller
        (annotator_support.nested_part_whole). Not saved by annotator_support.py, whose top-level keys pool all
        cross-subject pairs; computed here and cached in results/msod_same_seg_part_whole.json (--recompute redoes it).
        The all-pairs values of annotator_support_sod.json are shown in the table for reference.
    MSOD, GT vs session: annotator_support_sod.json, SOD.gt_iou_union_mean (all objects a session chose) and
        gt_iou_top_mean (most salient object).
    MSOD per model (four models with maps): by_category.part_of_gt.loss_share, .some_session_iou80_top and
        by_category.other_object.some_session_iou80_top.
    PASCAL-S per model (five models): pascal_agreement.json, by_category.<type>[">=3/12"] for whole_of_gt,
        other_object, boundary, and minority_supported_loss_share.
    DUT-OMRON per model: annotator_support_omron.json, by_category.whole_of_gt.extra_fixated_share.
  Inputs: results/error_types/{pascal_agreement,annotator_support_sod,annotator_support_omron}.json
  (written by scripts/pascal_agreement.py and scripts/annotator_support.py).

  python scripts/multi_stats_table.py --out <dir> [--workers 4] [--recompute]
"""
import argparse
import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.io as sio
from scipy.ndimage import binary_fill_holes

from annotator_support import SOD_RAW, _objects, nested_part_whole
from review_tables import pct
from sodsam import data

ROOT = Path(__file__).resolve().parents[1]
ERR = ROOT / "results" / "error_types"
CACHE = ROOT / "results" / "msod_same_seg_part_whole.json"
MODELS = [("selfreformer", "SelfReformer"), ("samba", "Samba"), ("birefnet", "BiRefNet"), ("mdsam", "MDSAM"),
          ("vscode", "VSCode"), ("vstpp", "VST++")]


# ---------------------------------------------------------------- MSOD, same-segmentation pairs

def _sessions(id_, shape):
    """Per session: subject, BSDS segmentation index and the most salient object's mask (as annotator_support)."""
    s = sio.loadmat(SOD_RAW / f"SO{id_}.mat", squeeze_me=True, struct_as_record=False)["SES"]
    out = []
    for e in _objects(s):
        top = np.zeros(shape, bool)
        for o in _objects(e.obj):
            if int(o.IMP) != 1:
                continue
            bnds = o.BND if isinstance(o.BND, np.ndarray) and o.BND.dtype == object else [o.BND]
            for b in bnds:
                b = np.atleast_2d(np.asarray(b, int)) - 1
                ok = (b[:, 0] >= 0) & (b[:, 0] < shape[0]) & (b[:, 1] >= 0) & (b[:, 1] < shape[1])
                edge = np.zeros(shape, bool)
                edge[b[ok, 0], b[ok, 1]] = True
                top |= binary_fill_holes(edge)
        out.append({"sj": int(e.sj), "seg": int(e.index), "top": top})
    return out


def _msod_image(id_):
    g = data.load_gt("MSOD", id_)
    ses = _sessions(id_, g.shape)
    same = [(a, b) for a, b in combinations(ses, 2) if a["sj"] != b["sj"] and a["seg"] == b["seg"]]
    return {"id": id_, "n_pairs": len(same), "pw_pairs": int(sum(nested_part_whole(a["top"], b["top"]) for a, b in same))}


def msod_same_seg(workers):
    # all 300 MSOD images, as in annotator_support_sod.json (n = 300); data.list_ids now drops the MSOD images that
    # duplicate an evaluation image (data.EXCLUDED), which leaves 226
    ids = sorted(p.stem for p in data.mask_dir("MSOD").glob("*.png") if (SOD_RAW / f"SO{p.stem}.mat").exists())
    with Pool(workers) as pool:
        recs = list(pool.imap(_msod_image, ids, chunksize=4))
    n_pairs = sum(r["n_pairs"] for r in recs)
    res = {"n_images": len(recs), "n_images_with_pairs": sum(r["n_pairs"] > 0 for r in recs), "n_pairs": n_pairs,
           "n_part_whole_pairs": sum(r["pw_pairs"] for r in recs),
           "share_of_pairs": sum(r["pw_pairs"] for r in recs) / max(n_pairs, 1),
           "images_with_any_pair": float(np.mean([r["pw_pairs"] > 0 for r in recs])),
           "images_with_any_pair_among_images_with_pairs":
               float(np.mean([r["pw_pairs"] > 0 for r in recs if r["n_pairs"]])),
           "per_image": recs}
    print(f"MSOD same-segmentation pairs: {n_pairs} pairs on {res['n_images_with_pairs']}/{len(recs)} images; "
          f"part/whole {res['share_of_pairs']:.1%} of pairs; images with any: {res['images_with_any_pair']:.0%} "
          f"(of images with pairs {res['images_with_any_pair_among_images_with_pairs']:.0%})", flush=True)
    return res


# ---------------------------------------------------------------- table

def pct1(x):
    return f"{100 * x:.1f}"


def table(same, sod, omron, pas):
    cols = MODELS
    nc = len(cols)
    out = ["\\begin{table*}[t]",
           "\\caption{Multi-annotator statistics of Sect.~5.5 of the paper. "
           "MSOD (all 300 images), people: pairs of sessions of different subjects on the same BSDS segmentation of the image; a pair "
           "is part/whole if one subject's most salient object lies at least 90\\% inside the other's and is at least "
           "two times smaller; share (\\%) of such pairs and of the images with at least one (in parentheses: all "
           "pairs of sessions of different subjects, regardless of the segmentation). "
           "MSOD, ground truth: mean IoU of the ground truth with a session's selection (all objects it chose; in "
           "parentheses its most salient object only). "
           "Per model (``--'': maps not available on that dataset): "
           "PASCAL-S, among images where the model adds at least 0.5\\% of the image outside the ground truth, the "
           "mean share (\\%) of the added pixels that lie in segments selected by at least 3 of the 12 subjects, by "
           "error type (whole of GT: the part/whole type with the larger model mask), and the share of the model's "
           "loss ($1-$ per-image $F$) on images whose added region is at least half selected by at least 3 subjects; "
           "DUT-OMRON, part/whole type with the larger model mask, share of the images whose added region holds at "
           "least one eye fixation (of 5 viewers); MSOD, the share of the loss in the part/whole type with the model "
           "mask inside the ground truth, and, for this type and for the other-object type, the share of the images "
           "where some session's most salient object has IoU $\\ge$ 0.8 with the model mask. "
           "Numbers of images per type are in parentheses.}",
           "\\label{tab:suppl_multi_stats}", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.5pt}",
           "\\begin{tabular}{@{}l" + "c" * nc + "@{}}", "\\toprule",
           " & " + " & ".join(label for _, label in cols) + " \\\\", "\\midrule"]
    allp = sod["human_part_whole_top"]
    span = f"\\multicolumn{{{nc}}}{{c}}"
    out += [f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{MSOD, people (most salient object)}} \\\\",
            f"\\quad Part/whole pairs (\\% of pairs) & {span}{{{pct1(same['share_of_pairs'])} "
            f"({pct1(allp['share_of_cross_subject_pairs'])})}} \\\\",
            f"\\quad Images with such a pair (\\%) & {span}{{{pct(same['images_with_any_pair'])} "
            f"({pct(allp['images_with_any_pair'])})}} \\\\",
            f"\\quad Mean IoU, ground truth vs a session's selection & {span}{{{sod['gt_iou_union_mean']:.2f} "
            f"({sod['gt_iou_top_mean']:.2f})}} \\\\", "\\midrule"]

    def cell(src, m, f):
        if src is None or m not in src:
            return "--"
        try:
            return f(src[m])
        except (KeyError, TypeError):
            return "--"

    def row(name, src, f):
        out.append(f"{name} & " + " & ".join(cell(src, m, f) for m, _ in cols) + " \\\\")

    def bycat(c, key, n=True):
        return lambda r: pct(r["by_category"][c][key]) + (f" ({r['by_category'][c]['n']})" if n else "")

    out.append(f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{PASCAL-S, added pixels selected by $\\ge$ 3 of 12 subjects (\\%)}} \\\\")
    row("\\quad Whole of GT", pas, bycat("whole_of_gt", ">=3/12"))
    row("\\quad Other object", pas, bycat("other_object", ">=3/12"))
    row("\\quad Boundary", pas, bycat("boundary", ">=3/12"))
    row("Loss on images whose added region is mostly selected (\\%)", pas,
        lambda r: pct(r["minority_supported_loss_share"]))
    out.append("\\midrule")
    row("DUT-OMRON, whole of GT: added region fixated (\\% of images)", omron,
        lambda r: pct(r["by_category"]["whole_of_gt"]["extra_fixated_share"]))
    out.append("\\midrule")
    out.append(f"\\multicolumn{{{nc + 1}}}{{@{{}}l}}{{MSOD}} \\\\")
    row("\\quad Part of GT: share of the loss (\\%)", sod, bycat("part_of_gt", "loss_share"))
    row("\\quad Part of GT: a subject's choice matches the mask (\\%)", sod,
        bycat("part_of_gt", "some_session_iou80_top", n=False))
    row("\\quad Other object: a subject's choice matches the mask (\\%)", sod,
        bycat("other_object", "some_session_iou80_top"))
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if args.recompute or not CACHE.exists():
        CACHE.write_text(json.dumps(msod_same_seg(args.workers), indent=1))
    same = json.loads(CACHE.read_text())
    sod = json.loads((ERR / "annotator_support_sod.json").read_text())["SOD"]
    omron = json.loads((ERR / "annotator_support_omron.json").read_text())["DUT-OMRON"]
    pas = json.loads((ERR / "pascal_agreement.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "suppl_multi_stats.tex").write_text(table(same, sod, omron, pas))
    print("wrote", out / "suppl_multi_stats.tex")


if __name__ == "__main__":
    main()
