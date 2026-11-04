"""Train the per-image candidate ranker (sodsam/ranker.py) and choose its abstention threshold.

Training: DUTS-TR, all 5 candidates of an image in one batch (--batch images x 5), loss = SmoothL1
regression of the per-image F + listwise cross-entropy against the best candidate, AdamW + cosine
schedule, horizontal flips. Validation: MSOD + MSRA-B1K (never seen by the SOD models); each epoch
tau is chosen on the grid 0..0.1 and the epoch is scored by the mean over the validation datasets of
the F^max of the chosen maps (the actual decision), which selects the checkpoint.
Outputs: weights/ranker_<sod_model><tag>.pt and results/ranker_<sod_model><tag>_train.json (args, tau,
validation F per epoch, backbone source).

The candidate cache is built first when missing (graph cut ~0.3 s/image on one core):
  python scripts/train_ranker.py --sod-model samba --build-cache --datasets DUTS-TR MSOD MSRA-B1K --workers 8
  python scripts/train_ranker.py --sod-model samba --workers 8
Smoke test: --limit 10 --epochs 1 --batch 5 --workers 0 (--cache-root / --out-root keep it out of the real dirs).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from sodsam import data, ranker

ROOT = Path(__file__).resolve().parents[1]


def add_common_args(ap):
    ap.add_argument("--sod-model", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N images of every dataset")
    ap.add_argument("--gc-params", default=None, help="tune2 json of the graph cut (default ranker.gc_params_path)")
    ap.add_argument("--cache-root", default=None, help="root of the candidate cache (default OUT_ROOT)")
    ap.add_argument("--out-root", default=str(ROOT), help="where weights/ and results/ are written")
    ap.add_argument("--tag", default="", help="suffix of the model name")
    ap.add_argument("--size", type=int, default=ranker.SIZE)
    ap.add_argument("--batch", type=int, default=16, help="images per batch (x 5 candidates)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")


def get_device(args):
    return args.device or ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")


def ensure_cache(args, datasets):
    for ds in datasets:
        ids = data.list_ids(ds)
        n = ranker.build_cache(args.sod_model, ds, ids[:args.limit] if args.limit else ids, args.workers,
                               args.gc_params, args.cache_root)
        print(f"{args.sod_model}/{ds}: candidate cache, {n} images built", flush=True)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--build-cache", action="store_true", help="only build the candidate cache of --datasets")
    ap.add_argument("--datasets", nargs="+", default=["DUTS-TR"] + data.TUNE_DATASETS + data.EVAL_DATASETS,
                    help="datasets of --build-cache")
    ap.add_argument("--train-dataset", default="DUTS-TR")
    ap.add_argument("--val-datasets", nargs="+", default=data.TUNE_DATASETS)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the listwise term")
    ap.add_argument("--list-scale", type=float, default=20.0, help="logit scale of the listwise softmax")
    ap.add_argument("--backbone-weights", default=None, help="ResNet-18 ImageNet state dict (.pth)")
    ap.add_argument("--scratch", action="store_true", help="no ImageNet initialisation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.build_cache:
        ensure_cache(args, args.datasets)
        return
    device = get_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_cache(args, [args.train_dataset] + args.val_datasets)

    train = ranker.RankerSet(args.sod_model, [args.train_dataset], args.size, True, args.cache_root, args.limit)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=len(train) >= args.batch, pin_memory=device == "cuda",
                        persistent_workers=args.workers > 0)
    val_loaders = {ds: DataLoader(ranker.RankerSet(args.sod_model, [ds], args.size, False, args.cache_root, args.limit),
                                  batch_size=args.batch, shuffle=False, num_workers=args.workers,
                                  pin_memory=device == "cuda")
                   for ds in args.val_datasets}
    net = ranker.Ranker(pretrained=not args.scratch, backbone_weights=args.backbone_weights).to(device)
    print(f"parameters {sum(p.numel() for p in net.parameters())}, backbone weights: {net.pretrained_from}, "
          f"{len(train)} training images, {len(loader)} steps/epoch, device {device}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs * len(loader)))

    name = f"ranker_{args.sod_model}{args.tag}"
    out_root = Path(args.out_root)
    (out_root / "weights").mkdir(parents=True, exist_ok=True)
    (out_root / "results").mkdir(parents=True, exist_ok=True)
    ckpt, log_path = out_root / "weights" / f"{name}.pt", out_root / "results" / f"{name}_train.json"
    best, best_tau, log, t0 = -1.0, 0.0, [], time.time()
    for ep in range(args.epochs):
        net.train()
        tot, parts, n = 0.0, {"reg": 0.0, "list": 0.0}, 0
        for x, f, _, _ in loader:
            x, f = x.to(device, non_blocking=True), f.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = net(x)
            loss, p = ranker.ranker_loss(pred.float(), f, args.lam, args.list_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            parts = {k: parts[k] + float(v) for k, v in p.items()}
            n += 1
        n = max(n, 1)
        val = {ds: ranker.predict(net, ld, device) for ds, ld in val_loaders.items()}
        tau, scores = ranker.choose_tau(list(val.values()))
        per = {ds: {"F_ranker": ranker.fmax(ranker.decide(p, tau), f, c),
                    "F_sod": ranker.fmax(np.zeros(len(f), int), f, c),
                    "F_oracle": ranker.fmax(f.argmax(1), f, c),
                    "mae_pred": float(np.abs(p - f).mean()),
                    "top1": float(np.mean(p.argmax(1) == f.argmax(1)))}
               for ds, (p, f, c) in val.items()}
        score = scores[tau]
        log.append({"epoch": ep, "loss": tot / n, "loss_reg": parts["reg"] / n, "loss_list": parts["list"] / n,
                    "tau": tau, "val_F": score, "val": per, "tau_curve": scores, "seconds": time.time() - t0})
        print(f"epoch {ep} loss {tot / n:.4f} (reg {parts['reg'] / n:.4f} list {parts['list'] / n:.4f}) tau {tau:.3f} "
              f"val F {score:.4f} " + " ".join(f"{ds}:{v['F_ranker']:.4f}(sod {v['F_sod']:.4f} oracle {v['F_oracle']:.4f}"
                                              f" top1 {v['top1']:.2f})" for ds, v in per.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if score > best:
            best, best_tau = score, tau
            torch.save(net.state_dict(), ckpt)
        log_path.write_text(json.dumps({"args": vars(args), "tau": best_tau, "best_val_F": best,
                                        "best_epoch": int(np.argmax([l["val_F"] for l in log])),
                                        "backbone_weights": net.pretrained_from, "candidates": ranker.CANDIDATES,
                                        "weights": str(ckpt), "log": log}, indent=1))
    print(f"best validation F {best:.4f} at tau {best_tau:.3f}; model {ckpt}, log {log_path}", flush=True)


if __name__ == "__main__":
    main()
