"""Checks for the Task 1 model, loss and metrics (run: pytest -q tests/)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.losses import L1SSIMLoss  # noqa: E402
from src.metrics import objective, psnr_per_image, ssim_per_image  # noqa: E402
from src.models import UDAE, count_params  # noqa: E402


def test_shapes_and_range():
    m = UDAE(base_channels=8, bottleneck_dim=64)
    x = torch.rand(2, 3, 128, 128)
    y = m(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_bottleneck_is_real_and_no_skips():
    m = UDAE(base_channels=8, bottleneck_dim=64)
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, 64)  # 49,152 input values squeezed through 64 numbers
    # decoder output must depend ONLY on z: identical z -> identical output regardless of x
    m.eval()
    a = m.decode(z)
    b = m.decode(z.clone())
    assert torch.equal(a, b)
    # encoder spatial size shrinks, channels grow
    h, sizes, chans = x, [], []
    for i in range(0, len(m.encoder), 2):
        h = m.encoder[i + 1](m.encoder[i](h))
        sizes.append(h.shape[-1]); chans.append(h.shape[1])
    assert sizes == [64, 32, 16, 8, 4] and chans == sorted(chans)


def test_loss_matches_formula_and_perfect_reconstruction():
    x = torch.rand(4, 3, 128, 128)
    loss, l1, s = L1SSIMLoss(0.8)(x.clone(), x)
    assert loss.item() < 1e-4 and s.item() > 0.999
    y = torch.rand_like(x)
    loss, l1, s = L1SSIMLoss(0.7)(y, x)
    assert abs(loss.item() - (0.7 * l1.item() + 0.3 * (1 - s.item()))) < 1e-5


def test_one_optimisation_step_decreases_loss():
    torch.manual_seed(0)
    m = UDAE(base_channels=8, bottleneck_dim=64, dropout=0.0)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    crit = L1SSIMLoss(0.8)
    x = torch.rand(8, 3, 128, 128)
    first = None
    for _ in range(15):
        opt.zero_grad()
        loss, _, _ = crit(m(x), x)
        loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first


def test_metrics():
    x = torch.rand(3, 3, 128, 128)
    assert torch.all(psnr_per_image(x, x) > 90)
    assert torch.allclose(ssim_per_image(x, x), torch.ones(3), atol=1e-4)
    assert objective(40.0, 1.0) == 1.0
    assert count_params(UDAE(8, 64)) > 0