"""Loading cached SOD/SAM outputs and producing the final maps of each method."""
import json

import numpy as np

from . import data
from .fusion import fuse, sod_binary

# methods evaluated in the main table and the ablation
METHODS = ["sod", "sod_crf", "sam_conf", "unary", "potts", "full", "full_legacy"]


def load_scores(sod_model, dataset, prompt):
    """SAM scores per id; for a combined prompt "p1+p2+..." the lists are concatenated."""
    out = {}
    for p in prompt.split("+"):
        with open(data.run_dir(sod_model, dataset, f"sam_{p}") / "scores.json") as f:
            for k, v in json.load(f).items():
                out.setdefault(k, []).extend(v)
    return out


def load_inputs(sod_model, dataset, prompt, id_, scores=None, neg_prompt=None, sod_model2=None,
                lo_model=None):
    """Image, SOD map, SAM masks/probabilities (3 per prompt; "p1+p2" concatenates prompts), GT.
    neg_prompt: masks of a complement prompt (e.g. mask_inv), returned as "sam_neg" (fusion2 +veto).
    sod_model2: a second SOD model; its map is returned as "sod2" and its masks of the same prompt(s)
    are appended to "sam" (fusion2 duo).
    lo_model: table | gbt loads the learned log-odds map (scripts/learn_unary.py) as "lo" (fusion2 logodds)."""
    img = data.load_image(dataset, id_)
    sod = data.read_png(data.run_dir(sod_model, dataset, "sod") / f"{id_}.png")
    sam, probs = [], []
    for model in [sod_model] + ([sod_model2] if sod_model2 else []):
        for p in prompt.split("+"):
            sam_dir = data.run_dir(model, dataset, f"sam_{p}")
            sam += [data.read_png(sam_dir / f"{id_}_{k}.png") > 0 for k in (1, 2, 3)]
            if probs is not None and (sam_dir / f"{id_}_1_p.png").exists():
                probs += [data.read_png(sam_dir / f"{id_}_{k}_p.png") for k in (1, 2, 3)]
            else:
                probs = None
    sam_neg = None
    if neg_prompt:
        sam_neg = [data.read_png(data.run_dir(sod_model, dataset, f"sam_{p}") / f"{id_}_{k}.png") > 0
                   for p in neg_prompt.split("+") for k in (1, 2, 3)]
    sod2 = data.read_png(data.run_dir(sod_model2, dataset, "sod") / f"{id_}.png") if sod_model2 else None
    return {"img": img, "sod": sod, "sam": sam, "probs": probs, "sam_neg": sam_neg, "sod2": sod2,
            "scores": None if scores is None else scores[id_],
            "lo": load_lo(sod_model, dataset, lo_model, id_) if lo_model else None, "lo_model": lo_model,
            "gt": data.load_gt(dataset, id_), "ds": dataset, "id": id_, "sod_model": sod_model}


def feat_dir(dataset):
    """SAM feature distances (run_sam_extra.py --prompt feat), independent of the SOD model."""
    return data.OUT_ROOT / "sam_feat" / dataset


def load_feat(dataset, id_):
    """(d_right, d_down) float32 in [0, 1], cropped to the (H, W-1) and (H-1, W) pair shapes."""
    out = []
    for suffix in ("r", "d"):
        p = feat_dir(dataset) / f"{id_}_{suffix}.png"
        if not p.exists():
            raise FileNotFoundError(f"SAM feature distances missing: {p} (run_sam_extra.py --prompt feat)")
        out.append(data.read_png(p).astype(np.float32) / 255.0)
    return out[0][:, :-1], out[1][:-1]


def lo_dir(sod_model, dataset, lo_model):
    """Learned log-odds maps of unary model lo_model (table | gbt; scripts/learn_unary.py)."""
    return data.run_dir(sod_model, dataset, f"lo_{lo_model}")


def load_lo(sod_model, dataset, lo_model, id_):
    """H x W float32 log-odds map (stored as float16)."""
    p = lo_dir(sod_model, dataset, lo_model) / f"{id_}.npy"
    if not p.exists():
        raise FileNotFoundError(f"log-odds map missing: {p} (learn_unary.py --model {lo_model})")
    return np.load(p).astype(np.float32)


V2_OUTPUTS = ("binary", "soft", "soft_gf", "soft_crf")
FEAT_PAIRWISE = ("feat", "both")
LOGODDS_UNARY = "logodds"


def method_output(method, inp, params, crf_params=None):
    """uint8 map (0..255) of a method for one image; crf_params are kwargs of dense_crf.

    method "v2:<path>[:<output>]" runs the fusion2 model of a tune2.py result json; the optional
    output overrides the model's: binary | soft | soft_gf (soft + guided filter) | soft_crf (soft +
    dense CRF with crf_params). A feat/both pairwise loads the feature distances of inp["ds"],
    inp["id"] unless inp["feat"] is set; a result tuned with --band passes its band_w through; a
    logodds unary loads the log-odds maps of the result's "lo_model" unless inp["lo"] holds them.
    """
    if method.startswith("v2:"):
        from .fusion2 import fuse2, prepare
        path, output = method[3:], None
        if ":" in path and path.rsplit(":", 1)[1] in V2_OUTPUTS:
            path, output = path.rsplit(":", 1)
        model = _v2_model(path)
        unary, pairwise, out0 = model["model"]
        output = output or out0
        if pairwise in FEAT_PAIRWISE and inp.get("feat") is None:
            inp = {**inp, "feat": load_feat(inp["ds"], inp["id"])}
        if unary.split("+")[0] == LOGODDS_UNARY:
            lo_model = model["lo_model"]
            if inp.get("lo") is None or inp.get("lo_model") != lo_model:
                inp = {**inp, "lo": load_lo(inp["sod_model"], inp["ds"], lo_model, inp["id"]), "lo_model": lo_model}
        band = bool(model.get("band", False))
        if output in ("binary", "soft"):
            return fuse2(prepare(inp), unary, pairwise, output, model["params"], band)
        soft = fuse2(prepare(inp), unary, pairwise, "soft", model["params"], band)
        if output == "soft_gf":
            from .guided_filter import guided_filter
            return (guided_filter(inp["img"], soft / 255.0) * 255).round().astype(np.uint8)
        from .dense_crf import dense_crf
        return dense_crf(inp["img"], soft, **(crf_params or {}))
    if method == "sod":
        return inp["sod"]
    if method == "sod_bin":
        return sod_binary(inp["sod"], params.sod_threshold).astype(np.uint8) * 255
    if method == "sam_conf":
        return inp["sam"][int(np.argmax(inp["scores"]))].astype(np.uint8) * 255
    if method == "sod_crf":
        from .dense_crf import dense_crf
        return dense_crf(inp["img"], inp["sod"], **(crf_params or {}))
    if method == "full_legacy":
        x = fuse(inp["img"], inp["sod"], inp["sam"], params, "full", "legacy")
    else:
        x = fuse(inp["img"], inp["sod"], inp["sam"], params, method, "paper")
    return x.astype(np.uint8) * 255


_V2_CACHE = {}


def _v2_model(path):
    if path not in _V2_CACHE:
        with open(path) as f:
            _V2_CACHE[path] = json.load(f)
    return _V2_CACHE[path]


def binary_f(pred_bool, gt):
    """F_beta of a binary map, i.e. the F curve (metrics.f_curve) of a 0/255 map at any threshold 1..255.
    Its dataset mean is F^max unless predicting every pixel as object (threshold 0)
    scores higher on average, which does not happen for reasonable maps (tests/test_metrics.py)."""
    if not gt.any():
        return None
    tp = np.logical_and(pred_bool, gt).sum()
    if tp == 0:
        return 0.0
    p = tp / pred_bool.sum()
    r = tp / gt.sum()
    return 1.3 * p * r / (0.3 * p + r)
