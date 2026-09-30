"""Evaluation metrics (PSNR, SSIM) and the Optuna validation objective for Task 1.

The Optuna objective must NOT depend on the loss weight alpha being tuned (otherwise trials with
different alpha would be compared on different scales), so it is built from the metrics only:

    objective = 0.5 * SSIM + 0.5 * (PSNR / 40)        (higher is better)

PSNR is divided by 40 dB, a typical upper range for this kind of restoration, so both terms live
on a similar 0..1 scale. It combines reconstruction quality (PSNR) and structural similarity (SSIM),
as the assignment asks.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from pytorch_msssim import ssim

from . import corruptions as C


def psnr_per_image(pred, target):
    mse = ((pred.float() - target.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_per_image(pred, target):
    return ssim(pred.float(), target.float(), data_range=1.0, size_average=False)


def objective(psnr: float, ssim_val: float) -> float:
    return 0.5 * ssim_val + 0.5 * (psnr / 40.0)


@torch.no_grad()
def evaluate_manifest(model, dataset, device, batch_size=128, num_workers=2, amp=True):
    """Runs `model` over a ManifestDataset (val or test). Returns
       per_entry : list of dicts (corruption, severity, psnr, ssim, l1, in_psnr, in_ssim)
       and the aggregated summaries produced by `summarize`."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    rows, k = [], 0
    for xc, x, _ in loader:
        xc, x = xc.to(device, non_blocking=True), x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(xc)
        out = out.float().clamp(0, 1)
        p, s = psnr_per_image(out, x).cpu(), ssim_per_image(out, x).cpu()
        ip, is_ = psnr_per_image(xc, x).cpu(), ssim_per_image(xc, x).cpu()
        l1 = (out - x).abs().flatten(1).mean(1).cpu()
        for j in range(x.shape[0]):
            e = dataset.entry(k)
            rows.append({"corruption": e["corruption"], "severity": e["severity"],
                         "psnr": p[j].item(), "ssim": s[j].item(), "l1": l1[j].item(),
                         # "input as reconstruction" reference is meaningless for clean inputs
                         # (identical to target -> infinite PSNR), so leave it undefined there
                         "in_psnr": float("nan") if e["corruption"] == "clean" else ip[j].item(),
                         "in_ssim": float("nan") if e["corruption"] == "clean" else is_[j].item()})
            k += 1
    return rows


def _nanmean(v):
    v = np.asarray(v, dtype=float)
    return None if np.isnan(v).all() else float(np.nanmean(v))


def summarize(rows):
    """Aggregates per-entry rows -> overall / by corruption / by (corruption, severity)."""
    def agg(rs):
        return {"n": len(rs),
                "psnr": float(np.mean([r["psnr"] for r in rs])),
                "ssim": float(np.mean([r["ssim"] for r in rs])),
                "l1": float(np.mean([r["l1"] for r in rs])),
                "in_psnr": _nanmean([r["in_psnr"] for r in rs]),
                "in_ssim": _nanmean([r["in_ssim"] for r in rs])}
    by_c, by_cs = defaultdict(list), defaultdict(list)
    for r in rows:
        by_c[r["corruption"]].append(r)
        by_cs[(r["corruption"], r["severity"])].append(r)
    out = {"overall": agg(rows),
           "by_corruption": {c: agg(by_c[c]) for c in C.CLASSES if c in by_c},
           "by_severity": {f"{c}/{s}": agg(v) for (c, s), v in sorted(by_cs.items())}}
    o = out["overall"]
    o["objective"] = objective(o["psnr"], o["ssim"])
    return out