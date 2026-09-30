"""Visualisation helpers: [clean target | corrupted input | reconstruction | absolute error] grids."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def _to_np(t):
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def make_grid_figure(clean, corrupted, output, titles=None, ncols=2, err_vmax=0.3):
    """clean / corrupted / output: float tensors [N,3,H,W] in [0,1].
    Each example takes 4 columns (clean, corrupted, output, |error|). The error map is the mean
    absolute error over RGB channels on a FIXED colour scale 0..err_vmax so maps are comparable."""
    n = clean.shape[0]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols * 4, figsize=(2.0 * ncols * 4, 2.15 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    heads = ["clean target", "corrupted input", "reconstruction", "abs. error"]
    for i in range(n):
        r, c0 = divmod(i, ncols)
        c0 *= 4
        err = (output[i] - clean[i]).abs().mean(0).detach().cpu().numpy()
        for j, im in enumerate([_to_np(clean[i]), _to_np(corrupted[i]), _to_np(output[i])]):
            axes[r, c0 + j].imshow(im)
        axes[r, c0 + 3].imshow(err, cmap="inferno", vmin=0, vmax=err_vmax)
        if r == 0:
            for j in range(4):
                axes[r, c0 + j].set_title(heads[j], fontsize=7)
        if titles is not None:
            axes[r, c0].text(2, 10, titles[i], color="w", fontsize=6,
                             bbox=dict(facecolor="black", alpha=0.6, pad=1, lw=0))
    fig.tight_layout()
    return fig