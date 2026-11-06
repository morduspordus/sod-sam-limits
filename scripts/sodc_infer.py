"""Saliency maps of MDSAM (512), VSCode-S and VST++-B for Part C of the second review: the three SOD models whose
released maps cover only the benchmarks, run here on the tuning sets too.

Each official network is built from its repository ($SODC_ROOT/<repo>, default ~/sodc), the released checkpoint is loaded as tensors
only (weights_only=True) and strictly, and the official test preprocessing and output resizing are reproduced:
  mdsam   MDSAM(512): RGB, cv2 bilinear resize to 512, ImageNet normalization; sigmoid, bilinear to image size
  vscode  VSCode-S (Swin-S, window 11, 352): PIL bilinear resize, the RGB-task normalization, task "RGB";
          sigmoid of the second saliency output (outputs_saliency_s, the released "yuanshi" maps)
  vstpp   VST++-B (select-token version, Swin-B, window 10, 320): PIL bilinear resize, ImageNet normalization;
          sigmoid of the first saliency output (the released "fenge" maps) or the second (--vst-out s)
Maps go to <out>/<model>/<dataset>/<stem>.png, uint8 at the image size.

  python -I scripts/sodc_infer.py --model vstpp --datasets ECSSD MSOD --out ~/sodc/preds
"""
import argparse
import os
import sys
import types
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.expanduser(os.environ.get("SODC_ROOT", "~/sodc"))          # repositories and ckpt/
DATA = os.path.expanduser(os.environ.get("SODC_DATA", "data/rgb_dataset"))  # <dataset>/images (Samba's layout)
CKPT = {"mdsam": "mdsam_512.pth", "vscode": "vscode_s.pth", "vstpp": "vstpp_b.pth"}


def load_sd(name):
    sd = torch.load(os.path.join(ROOT, "ckpt", CKPT[name]), map_location="cpu", weights_only=True)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())


def load_strict(net, sd, name):
    res = net.load_state_dict(sd, strict=False)
    # the Swin ImageNet classifier is built but unused by the saliency networks, and not in their checkpoints
    missing = [k for k in res.missing_keys if not k.startswith("rgb_backbone.head.")]
    # VST++-B's checkpoint also holds a 384 -> 1 token head (convcertor_pre) that no released code version defines
    # or uses; the released maps are reproduced without it (checked against them on ECSSD)
    unexpected = [k for k in res.unexpected_keys if not k.startswith("convcertor_pre.")]
    print(f"{name}: missing {len(res.missing_keys)} ({len(missing)} other than the unused Swin classifier), "
          f"unexpected {len(res.unexpected_keys)} ({len(unexpected)} other than convcertor_pre)", flush=True)
    if missing or unexpected:
        print("missing", missing[:20], "\nunexpected", unexpected[:20])
        sys.exit(3)


def fix_linears(net, sd):
    """Replace nn.Linear layers whose shape differs from the checkpoint (the repositories hard-code the Swin-T
    widths); the strict load that follows checks that every parameter then matches."""
    for k, v in sd.items():
        if not k.endswith(".weight") or v.dim() != 2:
            continue
        path = k[:-len(".weight")].split(".")
        parent = net
        for p in path[:-1]:
            parent = getattr(parent, p, None) if not p.isdigit() else parent[int(p)]
            if parent is None:
                break
        if parent is None:
            continue
        leaf = path[-1]
        mod = parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf, None)
        if isinstance(mod, nn.Linear) and tuple(mod.weight.shape) != tuple(v.shape):
            new = nn.Linear(v.shape[1], v.shape[0], bias=mod.bias is not None)
            if leaf.isdigit():
                parent[int(leaf)] = new
            else:
                setattr(parent, leaf, new)
            print("resized", k, tuple(mod.weight.shape), "->", tuple(v.shape))


class MDSAMRunner:
    size = 512

    def __init__(self):
        sys.path.insert(0, os.path.join(ROOT, "MDSAM"))
        from model.mdsam import MDSAM
        self.net = MDSAM(self.size)
        load_strict(self.net, load_sd("mdsam"), "mdsam")
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def __call__(self, img):
        import cv2
        x = cv2.resize(np.asarray(img), (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = (x / 255.0 - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda()
        out, _ = self.net(x)
        out = F.interpolate(torch.sigmoid(out), [img.size[1], img.size[0]], mode="bilinear", align_corners=False)
        return (out * 255).squeeze().cpu().numpy().astype(np.uint8)


class VSTRunner:
    """VSCode-S and VST++-B share the VST code base: Swin backbone, token transformer and decoder."""

    def __init__(self, name, vst_out="fenge"):
        import importlib
        self.name = name
        if name == "vscode":
            code, self.size, self.out_idx = os.path.join(ROOT, "VSCode", "VSCode"), 352, 2
            mean, std = [0.492, 0.463, 0.298], [0.222, 0.219, 0.225]
        else:
            code, self.size = os.path.join(ROOT, "VST-plusplus", "RGB_VST_select_token"), 320
            self.out_idx = 0 if vst_out == "fenge" else 2
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        sys.path.insert(0, code)
        pkg = "Models_Test" if name == "vscode" else "Models"
        swin = importlib.import_module(pkg + ".swin_transformer")
        idn = importlib.import_module(pkg + ".ImageDepthNet")
        if name == "vscode":
            backbone = lambda pretrained=True, **kw: swin.SwinTransformer(
                img_size=352, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], window_size=11,
                drop_path_rate=0.1)
            idn.swin_transformer = backbone
        else:
            backbone = lambda pretrained=True, **kw: swin.SwinTransformer(
                img_size=320, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=10,
                drop_path_rate=0.2)
            idn.swin_transformer_T = backbone
        args = types.SimpleNamespace(img_size=self.size, pretrained_model="", embed_dim=384, dim=64,
                                     encoder_dim=[96, 192, 384, 768], domain_num=[1, 1, 1, 1], task_num=[1, 1, 5, 10],
                                     task_deco_num=10)
        self.net = idn.ImageDepthNet(args)
        sd = load_sd(name)
        fix_linears(self.net, sd)
        load_strict(self.net, sd, name)
        from torchvision import transforms
        self.norm = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img):
        w, h = img.size
        x = self.norm(self.to_tensor(img.resize((self.size, self.size), Image.BILINEAR)))[None].cuda()
        if self.name == "vscode":
            depth = torch.zeros_like(x)
            outs = self.net(x, depth, "RGB")
        else:
            outs = self.net(x)
        sal = torch.sigmoid(outs[self.out_idx][-1]).cpu().squeeze(0)
        # official: ToPILImage (x255, truncated to uint8), then PIL bilinear resize to the image size
        pil = Image.fromarray(sal.mul(255).byte().squeeze(0).numpy())
        return np.asarray(pil.resize((w, h), Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPT))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "preds"))
    ap.add_argument("--vst-out", default="fenge", choices=["fenge", "s"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    run = MDSAMRunner() if args.model == "mdsam" else VSTRunner(args.model, args.vst_out)
    run.net.cuda().eval()
    tag = args.model if args.vst_out == "fenge" else args.model + "_s"
    for ds in args.datasets:
        src = os.path.join(DATA, ds, "images")
        # image files only (DUT-OMRON's folder also holds a Thumbs.db)
        names = sorted(n for n in os.listdir(src) if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
        names = names[: args.limit or None]
        dst = os.path.join(args.out, tag, ds)
        os.makedirs(dst, exist_ok=True)
        with torch.no_grad():
            for n in names:
                out = os.path.join(dst, os.path.splitext(n)[0] + ".png")
                if os.path.exists(out):
                    continue
                img = Image.open(os.path.join(src, n)).convert("RGB")
                Image.fromarray(run(img)).save(out)
        print(ds, len(names), "maps ->", dst, flush=True)
    print("SODC_INFER_DONE")


if __name__ == "__main__":
    main()
