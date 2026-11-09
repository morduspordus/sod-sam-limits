"""Figures of the revised manuscript (PDF, raster), from the saved maps:

  samsal    image | GT | SOD mask | SAM masks 1-3 (zoom-in prompt, by decreasing confidence), one row per
            case: all SAM masks better, SOD better, both poor, SAM misses a part, SAM adds a region
  edges     SAM masks and their oriented boundaries (at least two masks agree), as in Sect. 4.3
  results   image | GT | SOD | graph cut | fusion head, for both SOD models
  failures  the same columns for images where both methods fail
  teaser    image | GT | SOD map | SAM masks 1-3 | graph cut | fusion head, one image
            (the paper: --sod-model selfreformer --pick teaser=sun_axksigdbtcbsdxqi)

Examples are chosen by the stated criteria among DUTS-TE images with a 4:3 landscape aspect ratio;
--pick name=id1,id2,... overrides the choice.

  python scripts/make_figures.py --out <paper>/fig --figs samsal edges
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sodsam import data, metrics

DS = "DUTS-TE"
PROMPT = "sam_mask_zoom"
H = 150                                  # tile height in pixels (tiles are 4:3, width 200)
W = 200
GAP = 6
FONT_PX = 26                             # column titles; shrunk where a title is wider than its tile
FONT = ImageFont.truetype("arial.ttf", FONT_PX)
SCRATCH = Path(__file__).resolve().parents[1] / "results" / "figures"


def tile(a, mask=False):
    a = a.astype(np.uint8) * 255 if a.dtype == bool else a
    a = np.repeat(a[..., None], 3, 2) if a.ndim == 2 else a
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_AREA)


def grid(rows, headers, path):
    """rows: list of lists of HxWx3 tiles; headers: column titles."""
    n_c = max(len(r) for r in rows)
    top = FONT_PX + 12 if headers else 0
    canvas = np.full((top + len(rows) * (H + GAP), n_c * (W + GAP) - GAP, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, t in enumerate(r):
            y, x = top + i * (H + GAP), j * (W + GAP)
            canvas[y:y + H, x:x + W] = t
    im = Image.fromarray(canvas)
    if headers:
        d = ImageDraw.Draw(im)
        for j, h in enumerate(headers):
            font = FONT
            while d.textlength(h, font=font) > W - 4:
                font = ImageFont.truetype("arial.ttf", font.size - 1)
            w = d.textlength(h, font=font)
            d.text((j * (W + GAP) + (W - w) / 2, 4), h, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "PDF", resolution=200)
    im.save(SCRATCH / (path.stem + ".png"))
    print("wrote", path)


def sam_masks(sod, id_):
    d = data.run_dir(sod, DS, PROMPT)
    sc = json.loads((d / "scores.json").read_text())[id_]
    order = np.argsort(-np.asarray(sc))
    return [data.read_png(d / f"{id_}_{k + 1}.png") > 0 for k in order]


def sod_mask(s):
    """Thresholded SOD map, max-normalized as everywhere else (error_types.py): s / max >= 127/255."""
    return s.astype(np.float64) / (s.max() + 1e-13) >= 127 / 255 if s.max() > 0 else np.zeros(s.shape, bool)


def fval(m, g):
    m = m.astype(np.uint8) * 255 if m.dtype == bool else m
    return float(metrics.f_curve(m, g).max())


def landscape_ids():
    out = []
    for i in data.list_ids(DS):
        g = data.load_gt(DS, i)
        if abs(g.shape[1] / g.shape[0] - 4 / 3) < 0.05 and 0.05 < g.mean() < 0.5:
            out.append(i)
    return out


def stats(sod, ids):
    """Per image: F of the SOD map and of the three SAM masks (by confidence), and mask geometry."""
    cache = SCRATCH / f"stats_{sod}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for i in ids:
        g = data.load_gt(DS, i)
        s = data.read_png(data.run_dir(sod, DS, "sod") / f"{i}.png")
        S = sod_mask(s)
        ms = sam_masks(sod, i)
        dil = cv2.dilate(S.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
        out[i] = {"f_sod": fval(s, g), "f_sam": [fval(m, g) for m in ms],
                  "inside": [float((m & S).sum() / max(m.sum(), 1)) for m in ms],
                  "area_ratio": [float(m.sum() / max(S.sum(), 1)) for m in ms],
                  "far": [float((m & ~dil).sum() / max(m.sum(), 1)) for m in ms],
                  "sod_area": float(S.mean())}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def pick_samsal(st):
    cases = {
        "sam_better": lambda r: min(r["f_sam"]) - r["f_sod"],
        "sod_better": lambda r: r["f_sod"] - max(r["f_sam"]) if r["f_sod"] > 0.9 else -1,
        "both_poor": lambda r: 0.7 - max([r["f_sod"]] + r["f_sam"]) if r["f_sod"] > 0.3 else -1,
        "sam_misses": lambda r: max((1 - a) for a, ins in zip(r["area_ratio"], r["inside"]) if ins > 0.95)
        if r["f_sod"] > 0.9 and any(ins > 0.95 and a < 0.5 for a, ins in zip(r["area_ratio"], r["inside"])) else -1,
        "sam_adds": lambda r: max(r["far"]) if r["f_sod"] > 0.85 else -1,
    }
    chosen = {}
    ok = [i for i in st if 0.03 < st[i]["sod_area"] < 0.6]          # a visible, not image-filling SOD mask
    for name, score in cases.items():
        ranked = sorted(ok, key=lambda i: -score(st[i]))
        chosen[name] = ranked[:5]
    return chosen


def fig_samsal(sod, out, pick):
    ids = landscape_ids()
    st = stats(sod, ids)
    cand = pick_samsal(st)
    print(json.dumps(cand, indent=1))
    if not pick:                                 # candidate sheets, one per case, to choose from
        for name, ids in cand.items():
            fig_samsal_rows(sod, ids, out / f"cand_samsal_{name}.pdf")
        return
    fig_samsal_rows(sod, pick, out / "sal_sam_v2.pdf")


def fig_samsal_rows(sod, use, path):
    rows = []
    for i in use:
        g = data.load_gt(DS, i)
        s = data.read_png(data.run_dir(sod, DS, "sod") / f"{i}.png")
        rows.append([tile(data.load_image(DS, i)), tile(g, True), tile(sod_mask(s), True)]
                    + [tile(m, True) for m in sam_masks(sod, i)])
    grid(rows, ["Image", "Ground truth", "SOD mask", "SAM mask 1", "SAM mask 2", "SAM mask 3"], path)


BG_TO_OBJ = (230, 159, 0)     # orange (Okabe-Ito, color-blind safe): p (above / left) background, q object
OBJ_TO_BG = (0, 114, 178)     # blue: p object, q background


def oriented(n):
    """RGB visualization of the oriented boundaries of the SAM count map n (0..3) on the 4-grid (pairs whose
    count changes by >= 2): orange where the upper / left pixel is on the background side, blue where it is
    on the object side; light gray: pixels in at least two masks. Call it on a map already resized to the
    tile size (nearest), so that the 3x3 dilation keeps the boundaries visible at figure scale."""
    vis = np.full(n.shape + (3,), 255, np.uint8)
    vis[n >= 2] = (225, 225, 225)                       # majority of the SAM masks
    dv = n[:-1, :].astype(int) - n[1:, :]
    dh = n[:, :-1].astype(int) - n[:, 1:]
    k = np.ones((3, 3), np.uint8)
    for m, col, sl in ((dv <= -2, BG_TO_OBJ, (slice(0, -1), slice(None))), (dv >= 2, OBJ_TO_BG, (slice(0, -1), slice(None))),
                       (dh <= -2, BG_TO_OBJ, (slice(None), slice(0, -1))), (dh >= 2, OBJ_TO_BG, (slice(None), slice(0, -1)))):
        full = np.zeros(n.shape, bool)
        full[sl] = m
        full = cv2.dilate(full.astype(np.uint8), k) > 0         # visible at figure scale
        vis[full] = col
    return vis


def fig_edges(sod, out, pick):
    ids = pick or ["ILSVRC2012_test_00071459", "ILSVRC2012_test_00003592"]
    rows = []
    for i in ids:
        ms = sam_masks(sod, i)
        n = sum(m.astype(np.uint8) for m in ms)
        n_small = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)
        rows.append([tile(data.load_image(DS, i))] + [tile(m, True) for m in ms] + [oriented(n_small)])
    grid(rows, ["Image", "SAM mask 1", "SAM mask 2", "SAM mask 3", "Oriented boundaries"], out / "edges_v2.pdf")


def outputs(sod, i, gc_name, head_tag):
    s = data.read_png(data.run_dir(sod, DS, "sod") / f"{i}.png")
    gc = data.read_png(data.run_dir(sod, DS, gc_name) / f"{i}.png")
    hd = data.read_png(data.run_dir(head_tag, DS, "sod") / f"{i}.png")
    return s, gc, hd


def fig_results(out, pick, gc_name, failures=False, exclude=()):
    """Rows alternate SOD models; chosen where the fusion helps most and both fused results are good
    (F >= .9), or, for failures, where the SOD map is good (F > .8) and both methods are worse than it.
    An image is shown for one SOD model only, and never one of `exclude` (images of other figures)."""
    rows = []
    shown = set(exclude)
    for sod in ("selfreformer", "samba"):
        ids = pick.get(sod) if pick else None
        if not ids:
            sc = []
            for i in landscape_ids():
                if i in shown:
                    continue
                g = data.load_gt(DS, i)
                s, gc, hd = outputs(sod, i, gc_name, f"{sod}_hz")
                f0, f1, f2 = fval(s, g), fval(gc, g), fval(hd, g)
                sc.append((i, (min(f1, f2) - f0) * (min(f1, f2) >= 0.9) if not failures
                           else (f0 - max(f1, f2)) * (f0 > 0.8)))
            ids = [i for i, _ in sorted(sc, key=lambda t: -t[1])[:3 if not failures else 2]]
            print(sod, ids)
        shown.update(ids)
        for i in ids:
            s, gc, hd = outputs(sod, i, gc_name, f"{sod}_hz")
            rows.append([tile(data.load_image(DS, i)), tile(data.load_gt(DS, i), True), tile(s), tile(gc), tile(hd)])
    grid(rows, ["Image", "Ground truth", "SOD", "Graph cut", "Fusion head"],
         out / ("failures_v2.pdf" if failures else "results_v2.pdf"))


def fig_teaser(sod, out, i, gc_name):
    s, gc, hd = outputs(sod, i, gc_name, f"{sod}_hz")
    rows = [[tile(data.load_image(DS, i)), tile(data.load_gt(DS, i), True), tile(s)]
            + [tile(m, True) for m in sam_masks(sod, i)] + [tile(gc), tile(hd)]]
    grid(rows, ["Image", "Ground truth", "SOD map", "SAM mask 1", "SAM mask 2", "SAM mask 3", "Graph cut",
                "Fusion head"], out / "teaser_v2.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figs", nargs="+", default=["samsal", "edges"])
    ap.add_argument("--sod-model", default="selfreformer")
    ap.add_argument("--gc-name", default="gc_zoom", help="subdir of the saved graph-cut maps (save_maps.py)")
    ap.add_argument("--pick", nargs="*", default=[], help="fig=id,id,... or fig:sod=id,id,...")
    ap.add_argument("--exclude", nargs="*", default=[], help="ids never shown in results/failures (used elsewhere)")
    args = ap.parse_args()
    out = Path(args.out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    picks = {}
    for p in args.pick:
        k, v = p.split("=", 1)
        picks[k] = v.split(",")
    if "samsal" in args.figs:
        fig_samsal(args.sod_model, out, picks.get("samsal"))
    if "edges" in args.figs:
        fig_edges(args.sod_model, out, picks.get("edges"))
    for fig, fail in (("results", False), ("failures", True)):
        if fig in args.figs:
            fig_results(out, {s: picks.get(f"{fig}:{s}") for s in ("selfreformer", "samba")}, args.gc_name, fail,
                        args.exclude)
    if "teaser" in args.figs:
        fig_teaser(args.sod_model, out, picks["teaser"][0], args.gc_name)


if __name__ == "__main__":
    main()
