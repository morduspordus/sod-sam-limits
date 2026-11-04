"""SOD models. Each model maps an RGB uint8 image to a uint8 saliency map of the same size.

selfreformer  SelfReformer (Yun & Lin), official code in $SELFREFORMER_ROOT (default
              third_party/SelfReformer) and the authors' DUTS-TR checkpoint best_DUTS-TE.pt; inference as in
              its tester.py.
precomputed   maps produced elsewhere (e.g. Samba, with its official code), read from a directory.
"""
import contextlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SELFREFORMER_ROOT = Path(os.environ.get("SELFREFORMER_ROOT", "third_party/SelfReformer"))
SELFREFORMER_CKPT = Path(os.environ.get("SELFREFORMER_CKPT",
                                        SELFREFORMER_ROOT / "ckpt" / "from_authors" / "best_DUTS-TE.pt"))


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class SelfReformer:
    name = "selfreformer"

    def __init__(self, ckpt=SELFREFORMER_CKPT, device="cuda", img_size=224):
        sys.path.insert(0, str(SELFREFORMER_ROOT))
        from model.network import Net
        opt = Namespace(transformer=[[2, 1, 512, 3, 49], [2, 1, 320, 3, 196],
                                     [2, 1, 128, 3, 784], [2, 1, 64, 3, 3136]])
        with _cwd(SELFREFORMER_ROOT):  # the encoder loads ./model/pretrain/pvt_v2_b2.pth
            net = Net(opt)
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.net = net.to(device).eval()
        self.device = device
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.458, 0.407], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        h, w = img.shape[:2]
        x = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float().to(self.device)
        x = (x[None] / 255.0 - self.mean) / self.std
        pred = self.net(x)
        sal = F.pixel_shuffle(pred[-1], 4)
        sal = F.interpolate(sal, (h, w), mode="bilinear", align_corners=False)
        sal = torch.sigmoid(sal).squeeze()
        return (sal * 255.0).cpu().numpy().astype(np.uint8)


class Precomputed:
    """Reads <dir>/<id>.png; used for models that run outside this environment."""
    name = "precomputed"

    def __init__(self, directory):
        self.directory = Path(directory)

    def load(self, id_, shape):
        a = cv2.imread(str(self.directory / (id_ + ".png")), cv2.IMREAD_GRAYSCALE)
        if a is None:
            raise FileNotFoundError(self.directory / (id_ + ".png"))
        if a.shape != shape:
            a = cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return a


MODELS = {"selfreformer": SelfReformer}
