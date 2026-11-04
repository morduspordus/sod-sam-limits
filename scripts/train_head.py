"""Item 7: a small refinement head trained on DUTS-TR to fuse the SOD map and the SAM outputs.

Input (8 channels at 352x352): RGB (ImageNet-normalized), SOD map, SAM mask count / 3, the three SAM
mask probabilities. Output: a saliency map. Loss: BCE + IoU. Light U-Net (~2M parameters).
Selection: best epoch by F^max on MSOD + MSRA-B1K (the selection set used for the tuned models).
Ablations: --inputs keeps only some channel groups (rgb 3, sod 1, sam 4), e.g. "rgb sod" (no SAM).
Training sets: DUTS-TR (the SOD models' own training set, where their maps are unrealistically good)
and MSRA-B4K (4000 images unseen by them); the cluster job trains both and validation decides.
Then the maps of the seven datasets are written to <OUT_ROOT>/<sod_model>_head/<dataset>/sod/<id>.png,
so they evaluate like any other SOD model (fuse_eval --methods sod, or as the input of the graph cut).

  python scripts/train_head.py --sod-model samba --epochs 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sodsam import data, metrics, pipeline

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
GROUPS = {"rgb": [0, 1, 2], "sod": [3], "sam": [4, 5, 6, 7]}     # channels of features()


def channels(inputs):
    return sorted(c for g in inputs for c in GROUPS[g])


def features(inp, size):
    """8 x size x size float32 input tensor of one image (probs required)."""
    img = cv2.resize(inp["img"], (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    s = cv2.resize(inp["sod"], (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    n = np.stack(inp["sam"]).astype(np.float32).mean(0)
    n = cv2.resize(n, (size, size), interpolation=cv2.INTER_LINEAR)
    pr = [cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
          for p in inp["probs"]]
    return np.concatenate([img.transpose(2, 0, 1), s[None], n[None], np.stack(pr)]).astype(np.float32)


class SodSamSet(Dataset):
    def __init__(self, sod_model, dataset, size, train, chans, prompt="mask_soft"):
        self.sod_model, self.dataset, self.size, self.train, self.chans = sod_model, dataset, size, train, chans
        self.prompt = prompt
        self.ids = data.list_ids(dataset)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        inp = pipeline.load_inputs(self.sod_model, self.dataset, self.prompt, id_)
        if inp["probs"] is None:
            raise FileNotFoundError(f"no SAM probabilities for {self.dataset}/{id_}")
        x = features(inp, self.size)[self.chans]
        gt = cv2.resize(inp["gt"].astype(np.float32), (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if self.train and np.random.rand() < 0.5:
            x, gt = x[:, :, ::-1].copy(), gt[:, ::-1].copy()
        return torch.from_numpy(x), torch.from_numpy(gt[None]), id_


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Head(nn.Module):
    """U-Net with 4 levels (32-64-128-256 channels)."""

    def __init__(self, cin=8, ch=(32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList([block(cin, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(len(ch) - 1)])
        self.dec = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in range(len(ch) - 1)])
        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for i in reversed(range(len(self.dec))):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.out(x)


def loss_fn(logit, gt):
    bce = F.binary_cross_entropy_with_logits(logit, gt)
    p = torch.sigmoid(logit)
    inter = (p * gt).sum((1, 2, 3))
    union = (p + gt - p * gt).sum((1, 2, 3))
    return bce + (1 - (inter + 1) / (union + 1)).mean()


@torch.no_grad()
def predict_maps(net, sod_model, dataset, size, device, out_dir=None, chans=None, prompt="mask_soft"):
    """Full-size uint8 maps (resized back from size x size); returns per-image metrics."""
    net.eval()
    per = []
    for id_ in data.list_ids(dataset):
        inp = pipeline.load_inputs(sod_model, dataset, prompt, id_)
        x = features(inp, size)
        x = torch.from_numpy(x if chans is None else x[chans])[None].to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            p = torch.sigmoid(net(x).float())[0, 0].cpu().numpy()
        h, w = inp["sod"].shape
        m = (cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) * 255).round().astype(np.uint8)
        if out_dir is not None:
            data.write_png(out_dir / f"{id_}.png", m)
        per.append(metrics.image_scores(m, inp["gt"]))
    return metrics.aggregate(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=352)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--train-dataset", default="DUTS-TR")
    ap.add_argument("--val-datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--test-datasets", nargs="+", default=data.EVAL_DATASETS)
    ap.add_argument("--max-train", type=int, default=0, help="debug: use only this many training images")
    ap.add_argument("--tag", default=None, help="output model name (default <sod_model>_head)")
    ap.add_argument("--inputs", nargs="+", default=["rgb", "sod", "sam"], choices=list(GROUPS))
    ap.add_argument("--prompt", default="mask_soft", help="SAM prompt whose masks and probabilities are used")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    chans = channels(args.inputs)
    train = SodSamSet(args.sod_model, args.train_dataset, args.size, True, chans, args.prompt)
    if args.max_train:
        train.ids = train.ids[:args.max_train]
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=True, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    net = Head(cin=len(chans)).to(device)
    print("parameters", sum(p.numel() for p in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader),
                                                pct_start=0.1)
    tag = args.tag or f"{args.sod_model}_head"
    ckpt_dir = Path(__file__).resolve().parents[1] / "results" / "head"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best, log, t0 = -1.0, [], time.time()
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for x, gt, _ in loader:
            x, gt = x.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                logit = net(x)
            loss = loss_fn(logit.float(), gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        val = {ds: predict_maps(net, args.sod_model, ds, args.size, device, chans=chans, prompt=args.prompt)
               for ds in args.val_datasets}
        f = float(np.mean([v["Fmax"] for v in val.values()]))
        log.append({"epoch": ep, "loss": tot / len(loader), "val": val, "seconds": time.time() - t0})
        print(f"epoch {ep} loss {tot / len(loader):.4f} val F " + " ".join(f"{ds}:{v['Fmax']:.4f}" for ds, v in val.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if f > best:
            best = f
            torch.save(net.state_dict(), ckpt_dir / f"{tag}.pt")
    net.load_state_dict(torch.load(ckpt_dir / f"{tag}.pt", map_location=device))
    test = {}
    for ds in args.val_datasets + args.test_datasets:
        test[ds] = predict_maps(net, args.sod_model, ds, args.size, device, data.run_dir(tag, ds, "sod"), chans,
                                args.prompt)
        print(metrics.format_row(f"{ds} {tag}", test[ds]), flush=True)
    (ckpt_dir / f"{tag}.json").write_text(json.dumps({"args": vars(args), "best_val_F": best, "log": log,
                                                      "test": test}, indent=1))


if __name__ == "__main__":
    main()
