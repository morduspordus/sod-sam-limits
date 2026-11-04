"""Learned per-image candidate ranker: which of {SOD map, SAM masks, graph cut} to output for an image.

The oracle (scripts/oracle.py) shows that a perfect per-image chooser among the SOD map and SAM's masks
beats the paper's graph cut (DUTS-TE / Samba: SOD .931, graph cut .939, oracle .949). A gradient-boosted
selector on scalar features (scripts/selector.py) could not find the images where SAM wins. Here a small
CNN that sees the image, the SOD map and the candidate mask predicts the candidate's per-image F.

Candidates per image (CANDIDATES):
  c0  the soft SOD map (its per-image F = max over thresholds of metrics.f_curve)
  c1..c3  SAM masks of prompt PROMPT (<sod_model>/<dataset>/sam_mask_zoom/<id>_k.png, binary)
  c4  the paper's graph cut of the reference tune2 model (gc_params_path); read from gc_zoom/<id>.png
      when that directory exists, otherwise computed with pipeline.method_output("v2:<json>") (~0.3 s/image)
The candidates and their per-image F are cached once per dataset (build_cache, multiprocessing) in
<OUT_ROOT>/<sod_model>/<dataset>/ranker_cands/<id>.npz (packbits masks, F of the 5 candidates and the F
curve of the SOD map, which the validation decision needs).

Model (Ranker): ImageNet-pretrained ResNet-18 (torchvision; weights from --backbone-weights, the torch hub
cache or a download, else random init with a warning), first conv widened to 5 input channels (RGB,
SOD map, candidate mask; new channels zero-initialised), global pooling, MLP head -> predicted F.
Loss (ranker_loss): SmoothL1 regression + lam * softmax cross-entropy over the image's candidates
(logits = scale * predicted F) against the one-hot best candidate.
Decision (decide): argmax of the predicted F, unless it does not beat the SOD map's by more than tau
(abstain); tau chosen on the validation sets by dataset F^max of the chosen maps (choose_tau).
"""
import os
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import data, metrics, pipeline

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "mask_zoom"
CANDIDATES = ["sod", "sam1", "sam2", "sam3", "gc"]
N_CAND = len(CANDIDATES)
IN_CHANNELS = 5                     # RGB, SOD map, candidate mask
SIZE = 352
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
TAU_GRID = np.round(np.arange(0.0, 0.1 + 1e-9, 0.005), 3)
RESNET18_FILE = "resnet18-f37072fd.pth"     # torchvision's ResNet18_Weights.IMAGENET1K_V1


# ----------------------------------------------------------------------------------------------- candidates

def gc_params_path(sod_model):
    """The reference graph-cut model (tune2.py result json) of an SOD model."""
    return ROOT / "results" / "tune2" / f"{sod_model}_MSOD+MSRA-B1K+MSRA-B4K_{PROMPT}_linear_paper_binary_all.json"


def cache_dir(sod_model, dataset, cache_root=None):
    return Path(cache_root or data.OUT_ROOT) / sod_model / dataset / "ranker_cands"


def gc_mask(sod_model, dataset, id_, inp=None, gc_params=None):
    """Boolean graph-cut mask: gc_zoom/<id>.png if present, else the cut of the reference model."""
    cached = data.run_dir(sod_model, dataset, "gc_zoom") / f"{id_}.png"
    if cached.exists():
        return data.read_png(cached) > 0
    gc_params = Path(gc_params or gc_params_path(sod_model))
    if not gc_params.exists():
        raise FileNotFoundError(f"neither {cached} nor the graph-cut params {gc_params} exist")
    if inp is None:
        inp = pipeline.load_inputs(sod_model, dataset, PROMPT, id_)
    return pipeline.method_output(f"v2:{gc_params}", inp, None) > 0


def candidates(sod_model, dataset, id_, gc_params=None):
    """(sod uint8 HxW, masks bool 4xHxW [sam1, sam2, sam3, gc], gt bool HxW, img uint8 HxWx3)."""
    inp = pipeline.load_inputs(sod_model, dataset, PROMPT, id_)
    masks = np.stack(inp["sam"] + [gc_mask(sod_model, dataset, id_, inp, gc_params)])
    return inp["sod"], masks, inp["gt"], inp["img"]


def _build_one(id_, sod_model, dataset, gc_params, cache_root):
    path = cache_dir(sod_model, dataset, cache_root) / f"{id_}.npz"
    sod, masks, gt, _ = candidates(sod_model, dataset, id_, gc_params)
    sod_curve = metrics.f_curve(sod, gt)
    f = [sod_curve.max()] + [metrics.f_curve(m.astype(np.uint8) * 255, gt).max() for m in masks]
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, shape=np.array(masks.shape[1:]), masks=np.packbits(masks.reshape(N_CAND - 1, -1), axis=1),
                        f=np.asarray(f, np.float32), sod_curve=sod_curve.astype(np.float32))
    os.replace(tmp, path)
    return id_


def build_cache(sod_model, dataset, ids=None, workers=1, gc_params=None, cache_root=None, progress=True):
    """Write the missing cache files of a dataset; returns the number of images built."""
    ids = data.list_ids(dataset) if ids is None else ids
    d = cache_dir(sod_model, dataset, cache_root)
    d.mkdir(parents=True, exist_ok=True)
    missing = [i for i in ids if not (d / f"{i}.npz").exists()]
    if not missing:
        return 0
    fn = partial(_build_one, sod_model=sod_model, dataset=dataset, gc_params=gc_params, cache_root=cache_root)
    if progress:
        from tqdm import tqdm
        bar = partial(tqdm, total=len(missing), desc=f"{sod_model}/{dataset} candidates")
    else:
        bar = iter
    if workers <= 1:
        list(bar(map(fn, missing)))
    else:
        with Pool(workers) as pool:
            list(bar(pool.imap_unordered(fn, missing, chunksize=4)))
    return len(missing)


def load_cached(sod_model, dataset, id_, cache_root=None):
    """{"masks": bool 4xHxW, "f": (5,) per-candidate F, "sod_curve": (256,) F curve of the SOD map}."""
    with np.load(cache_dir(sod_model, dataset, cache_root) / f"{id_}.npz") as z:
        h, w = (int(v) for v in z["shape"])
        masks = np.unpackbits(z["masks"], axis=1, count=h * w).reshape(N_CAND - 1, h, w).astype(bool)
        return {"masks": masks, "f": z["f"].astype(np.float32), "sod_curve": z["sod_curve"].astype(np.float32)}


def candidate_map(k, sod, masks):
    """uint8 map output when candidate k is chosen: the soft SOD map for k = 0, else the binary mask x 255."""
    return sod if k == 0 else masks[k - 1].astype(np.uint8) * 255


# ----------------------------------------------------------------------------------------------- features

def features(img, sod, masks, size=SIZE):
    """N_CAND x IN_CHANNELS x size x size float32: per candidate [RGB (ImageNet-normalized), SOD map, candidate]."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)
    s = cv2.resize(sod, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    cands = [s] + [cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) for m in masks]
    base = np.concatenate([x, s[None]])
    return np.stack([np.concatenate([base, c[None]]) for c in cands]).astype(np.float32)


class RankerSet(Dataset):
    """Items: (x N_CAND x IN_CHANNELS x size x size, f (N_CAND,), sod_curve (256,), index)."""

    def __init__(self, sod_model, datasets, size=SIZE, train=False, cache_root=None, limit=0):
        self.sod_model, self.size, self.train, self.cache_root = sod_model, size, train, cache_root
        self.items = []
        for ds in datasets:
            ids = data.list_ids(ds)
            self.items += [(ds, i) for i in (ids[:limit] if limit else ids)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ds, id_ = self.items[i]
        img = data.load_image(ds, id_)
        sod = data.read_png(data.run_dir(self.sod_model, ds, "sod") / f"{id_}.png")
        c = load_cached(self.sod_model, ds, id_, self.cache_root)
        x = features(img, sod, c["masks"], self.size)
        if self.train and np.random.rand() < 0.5:
            x = x[:, :, :, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(c["f"]), torch.from_numpy(c["sod_curve"]), i


# ----------------------------------------------------------------------------------------------- model

def load_backbone_weights(backbone_weights=None):
    """ImageNet state dict of ResNet-18 and where it came from (None if unavailable): an explicit file,
    $RANKER_BACKBONE, the torch hub cache, or a download (compute nodes without internet get None)."""
    for src in (backbone_weights, os.environ.get("RANKER_BACKBONE"),
                Path(torch.hub.get_dir()) / "checkpoints" / RESNET18_FILE):
        if src and Path(src).exists():
            return torch.load(src, map_location="cpu"), str(src)
    try:
        from torchvision.models import ResNet18_Weights
        return ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=False), "download"
    except Exception as e:                                   # no internet: train from scratch
        warnings.warn(f"ResNet-18 ImageNet weights unavailable ({e}); the backbone is randomly initialised")
        return None, None


class Ranker(nn.Module):
    """ResNet-18 scorer: (N, IN_CHANNELS, S, S) -> (N,) predicted per-image F of each candidate;
    (B, K, IN_CHANNELS, S, S) -> (B, K)."""

    def __init__(self, in_channels=IN_CHANNELS, pretrained=True, backbone_weights=None, hidden=256, dropout=0.2):
        super().__init__()
        from torchvision.models import resnet18
        net = resnet18(weights=None)
        self.pretrained_from = None
        if pretrained:
            sd, self.pretrained_from = load_backbone_weights(backbone_weights)
            if sd is not None:
                net.load_state_dict(sd)
        conv = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, :3] = net.conv1.weight
        net.conv1 = conv
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Linear(512, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x):
        shape = x.shape[:-3]
        y = self.head(self.backbone(x.reshape(-1, *x.shape[-3:]))).squeeze(-1)
        return y.reshape(shape)


def ranker_loss(pred, target, lam=1.0, scale=20.0):
    """pred, target: (B, K) predicted / true per-image F. SmoothL1 regression plus lam times the softmax
    cross-entropy of logits scale * pred against the best candidate. Returns (loss, {"reg", "list"})."""
    reg = F.smooth_l1_loss(pred, target, beta=0.05)
    listwise = F.cross_entropy(pred * scale, target.argmax(1))
    return reg + lam * listwise, {"reg": reg.detach(), "list": listwise.detach()}


# ----------------------------------------------------------------------------------------------- decision

def decide(pred, tau):
    """Chosen candidate per image: argmax of the predicted F if it beats the SOD map's by more than tau, else 0."""
    pred = np.asarray(pred)
    k = pred.argmax(1)
    return np.where(pred[np.arange(len(pred)), k] - pred[:, 0] > tau, k, 0)


def chosen_curves(picks, f, sod_curves):
    """F curves (N, 256) of the chosen maps: the SOD map's curve, or the flat curve of a binary mask."""
    picks, f, sod_curves = np.asarray(picks), np.asarray(f), np.asarray(sod_curves)
    flat = np.repeat(f[np.arange(len(f)), picks][:, None], sod_curves.shape[1], axis=1)
    return np.where((picks == 0)[:, None], sod_curves, flat)


def fmax(picks, f, sod_curves):
    """Dataset F^max (max over thresholds of the mean F curve) of the chosen maps."""
    return float(chosen_curves(picks, f, sod_curves).mean(0).max())


def choose_tau(parts, grid=TAU_GRID):
    """parts: [(pred, f, sod_curves)] per validation dataset. The tau of the grid that maximises the mean
    over the datasets of their F^max under decide(pred, tau); returns (tau, {tau: mean F^max})."""
    scores = {float(t): float(np.mean([fmax(decide(p, t), f, c) for p, f, c in parts])) for t in grid}
    best = max(scores, key=lambda t: (scores[t], -t))
    return best, scores


@torch.no_grad()
def predict(net, loader, device):
    """Predicted F (N, K), true F (N, K) and SOD curves (N, 256) of a RankerSet, in dataset order."""
    net.eval()
    n = len(loader.dataset)
    pred, f, curves = np.zeros((n, N_CAND), np.float32), np.zeros((n, N_CAND), np.float32), np.zeros((n, 256), np.float32)
    for x, ft, ct, idx in loader:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            y = net(x.to(device, non_blocking=True)).float().cpu().numpy()
        idx = idx.numpy()
        pred[idx], f[idx], curves[idx] = y, ft.numpy(), ct.numpy()
    return pred, f, curves
