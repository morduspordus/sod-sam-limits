"""CPU checks of sodsam/ranker.py: forward shape, loss decreases on a toy batch, decision logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from sodsam import ranker


def test_forward_shape():
    torch.manual_seed(0)
    net = ranker.Ranker(pretrained=False).eval()
    with torch.no_grad():
        y = net(torch.randn(2, ranker.N_CAND, ranker.IN_CHANNELS, 64, 64))
        z = net(torch.randn(3, ranker.IN_CHANNELS, 64, 64))
    assert y.shape == (2, ranker.N_CAND) and z.shape == (3,), (y.shape, z.shape)
    assert torch.isfinite(y).all()
    # the new input channels start at zero, the RGB ones are the backbone's
    assert net.backbone.conv1.weight[:, 3:].abs().sum() == 0


def test_loss_decreases():
    torch.manual_seed(0)
    net = ranker.Ranker(pretrained=False, dropout=0.0)
    x = torch.randn(4, ranker.N_CAND, ranker.IN_CHANNELS, 32, 32)
    target = torch.rand(4, ranker.N_CAND)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    losses = []
    for _ in range(12):
        loss, parts = ranker.ranker_loss(net(x), target)
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in parts.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses


def test_decide():
    pred = np.array([[0.90, 0.95, 0.80, 0.70, 0.91],    # sam1 wins by .05
                     [0.90, 0.91, 0.80, 0.70, 0.85],    # sam1 wins by .01 only
                     [0.90, 0.50, 0.60, 0.70, 0.80],    # SOD is the best
                     [0.60, 0.50, 0.95, 0.70, 0.80]])   # sam2 wins by .35
    assert ranker.decide(pred, 0.0).tolist() == [1, 1, 0, 2]
    assert ranker.decide(pred, 0.02).tolist() == [1, 0, 0, 2]
    assert ranker.decide(pred, 0.1).tolist() == [0, 0, 0, 2]
    assert ranker.decide(pred, 0.5).tolist() == [0, 0, 0, 0]


def test_fmax_and_tau():
    # two images: the SOD map's curve peaks at .8 (threshold index 100); the masks' F are flat
    curves = np.zeros((2, 256), np.float32)
    curves[0, 100], curves[1, 100] = 0.8, 0.8
    curves[0, 50], curves[1, 50] = 0.7, 0.2
    f = np.array([[0.8, 0.9, 0.1, 0.1, 0.1], [0.8, 0.1, 0.1, 0.1, 0.1]], np.float32)
    assert abs(ranker.fmax(np.array([0, 0]), f, curves) - 0.8) < 1e-6
    assert abs(ranker.fmax(np.array([1, 0]), f, curves) - 0.85) < 1e-6
    assert abs(ranker.fmax(np.array([1, 1]), f, curves) - 0.5) < 1e-6
    # predictions: image 0 prefers sam1 by .03 (right), image 1 by .06 (wrong, and worse): abstaining
    # everywhere (tau >= .06) gives .8, tau in (.03, .06) keeps only the wrong replacement (.45), below .03 both (.5)
    pred = np.array([[0.80, 0.83, 0, 0, 0], [0.80, 0.86, 0, 0, 0]])
    tau, scores = ranker.choose_tau([(pred, f, curves)], grid=[0.0, 0.02, 0.04, 0.08])
    assert tau == 0.08 and abs(scores[0.0] - 0.5) < 1e-6 and abs(scores[0.04] - 0.45) < 1e-6 \
        and abs(scores[0.08] - 0.8) < 1e-6, (tau, scores)
    # equal scores prefer the smaller tau
    tau, _ = ranker.choose_tau([(pred, f, curves)], grid=[0.1, 0.08])
    assert tau == 0.08
    # the mean over datasets decides: a second dataset where every replacement is right favours tau = 0
    f2 = np.array([[0.5, 0.9, 0, 0, 0], [0.5, 0.9, 0, 0, 0]], np.float32)
    tau, _ = ranker.choose_tau([(pred, f, curves), (pred, f2, curves * 0.5)], grid=[0.0, 0.08])
    assert tau == 0.0


if __name__ == "__main__":
    test_forward_shape()
    test_loss_decreases()
    test_decide()
    test_fmax_and_tau()
    print("all tests passed")
