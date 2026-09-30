"""Task 1 training: Universal Denoising Autoencoder with MLflow tracking.

Colab example (baseline run):
    python -m src.train_task1 --config configs/task1_baseline.yaml \
        --root /content/drive/MyDrive/genai_a1 --run_name baseline

Anything in the config can be overridden:  --set lr=5e-4 epochs=60 alpha=0.7

Layout under --root:
    processed/  manifests/            (Phase 1 / Phase 2 outputs)
    runs/<run_name>/                  best.pt, last.pt, sample grids, val_summary.json
    mlflow.db, mlartifacts/           MLflow tracking store (SQLite + artifacts)
The function `train(cfg, trial=None)` is reused by the Optuna study in Phase 4.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import corruptions as C
from .datasets import ManifestDataset, RuntimeCorruptionDataset, load_arrays
from .losses import L1SSIMLoss
from .metrics import evaluate_manifest, summarize
from .models import UDAE, count_params
from .viz import make_grid_figure

DEFAULTS = dict(
    # data
    root="data", num_workers=2,
    # model
    base_channels=64, bottleneck_dim=256, dropout=0.1,
    # optimisation
    lr=1e-3, weight_decay=1e-4, batch_size=32, epochs=40, alpha=0.8, amp=True,
    patience=10,            # early stopping on the val objective
    # bookkeeping
    seed=42, run_name="baseline", experiment="task1_udae", sample_every=5,
    use_mlflow=True, save_checkpoints=True, resume=True,
)


def load_cfg(path=None, overrides=None):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
    for kv in overrides or []:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg


def setup_mlflow(root, experiment):
    """MLflow with a SQLite backend (the plain-file backend is deprecated in current MLflow).
    Database + artifacts live under <root> so they survive Colab restarts (root is on Drive)."""
    import mlflow
    root = Path(root)
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log_fig(cfg, fig, run_dir, name):
    path = run_dir / name
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    if cfg["use_mlflow"]:
        import mlflow
        mlflow.log_artifact(str(path), artifact_path="samples")


@torch.no_grad()
def sample_grid(model, val_ds, idxs, device, amp):
    model.eval()
    items = [val_ds[i] for i in idxs]
    xc = torch.stack([it[0] for it in items]).to(device)
    x = torch.stack([it[1] for it in items])
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        out = model(xc)
    titles = [f"{val_ds.entry(i)['corruption']}/{val_ds.entry(i)['severity']}" for i in idxs]
    return make_grid_figure(x, xc.cpu(), out.float().cpu(), titles)


def train(cfg, trial=None):
    """Trains one model. Returns dict(best_objective, best_epoch, summary, run_dir)."""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    root = Path(cfg["root"])
    run_dir = root / "runs" / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    trainval, _test, split = load_arrays(root / "processed")  # test array is loaded but never used here
    train_ds = RuntimeCorruptionDataset(trainval, split["train_idx"], seed=cfg["seed"])
    val_ds = ManifestDataset(trainval, root / "manifests" / "val_manifest.json")
    nw = cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                              num_workers=nw, pin_memory=device.type == "cuda",
                              persistent_workers=nw > 0)

    # ---- model / optimiser ----
    model = UDAE(cfg["base_channels"], cfg["bottleneck_dim"], cfg["dropout"]).to(device)
    criterion = L1SSIMLoss(cfg["alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- resume (Colab may disconnect) ----
    start_epoch, best_obj, best_epoch, bad = 0, -math.inf, -1, 0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    run_id = None
    if cfg["resume"] and cfg["save_checkpoints"] and last_path.exists() and trial is None:
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best_obj, best_epoch, bad = ck["epoch"] + 1, ck["best_obj"], ck["best_epoch"], ck["bad"]
        run_id = ck.get("mlflow_run_id")
        print(f"resumed from epoch {start_epoch}")

    mlflow = None
    if cfg["use_mlflow"]:
        import mlflow
        setup_mlflow(root, cfg["experiment"])
        run = mlflow.start_run(run_id=run_id, run_name=cfg["run_name"]) if run_id else \
            mlflow.start_run(run_name=cfg["run_name"])
        run_id = run.info.run_id
        if start_epoch == 0:
            mlflow.log_params({k: v for k, v in cfg.items()})
            mlflow.log_param("n_params", count_params(model))
            mlflow.log_param("device", str(device))

    fixed_idx = np.linspace(0, len(val_ds) - 1, 8).astype(int).tolist()  # same val examples every time
    result = None
    try:
        for epoch in range(start_epoch, cfg["epochs"]):
            t0 = time.time()
            model.train()
            tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
            for xc, x, _ in train_loader:
                xc, x = xc.to(device, non_blocking=True), x.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(xc)
                loss, l1, s = criterion(out, x)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update(); sched.step()
                tot["loss"] += loss.item(); tot["l1"] += l1.item(); tot["ssim"] += s.item()
            n = len(train_loader)
            tr = {k: v / n for k, v in tot.items()}

            rows = evaluate_manifest(model, val_ds, device, num_workers=nw, amp=use_amp)
            summ = summarize(rows)
            o = summ["overall"]
            val_loss = cfg["alpha"] * o["l1"] + (1 - cfg["alpha"]) * (1 - o["ssim"])
            log = {"train_loss": tr["loss"], "train_l1": tr["l1"], "train_ssim": tr["ssim"],
                   "val_loss": val_loss, "val_l1": o["l1"], "val_psnr": o["psnr"],
                   "val_ssim": o["ssim"], "val_objective": o["objective"],
                   "lr": opt.param_groups[0]["lr"], "epoch_time_s": time.time() - t0}
            for c, v in summ["by_corruption"].items():
                log[f"val_psnr_{c}"] = v["psnr"]; log[f"val_ssim_{c}"] = v["ssim"]
            if mlflow:
                mlflow.log_metrics(log, step=epoch)
            print(f"ep {epoch:3d} | train {tr['loss']:.4f} | val loss {val_loss:.4f} "
                  f"PSNR {o['psnr']:.2f} SSIM {o['ssim']:.4f} obj {o['objective']:.4f} | {log['epoch_time_s']:.0f}s")

            improved = o["objective"] > best_obj
            if improved:
                best_obj, best_epoch, bad = o["objective"], epoch, 0
                if cfg["save_checkpoints"]:
                    torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)
            else:
                bad += 1
            if cfg["save_checkpoints"]:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": epoch, "best_obj": best_obj,
                            "best_epoch": best_epoch, "bad": bad, "mlflow_run_id": run_id, "cfg": cfg}, last_path)
            if cfg["sample_every"] and (epoch % cfg["sample_every"] == 0 or epoch == cfg["epochs"] - 1) \
                    and cfg["save_checkpoints"]:
                _log_fig(cfg, sample_grid(model, val_ds, fixed_idx, device, use_amp), run_dir, f"samples_ep{epoch:03d}.png")

            if trial is not None:  # Optuna pruning hook (Phase 4)
                import optuna
                trial.report(o["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if bad >= cfg["patience"]:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch})")
                break

        # ---- final validation summary with the BEST weights ----
        if cfg["save_checkpoints"] and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
        summ = summarize(evaluate_manifest(model, val_ds, device, num_workers=nw, amp=use_amp))
        summ["best_epoch"], summ["cfg"] = best_epoch, cfg
        with open(run_dir / "val_summary.json", "w") as f:
            json.dump(summ, f, indent=2)
        if mlflow:
            mlflow.log_metrics({k: v for k, v in {"best_val_objective": summ["overall"]["objective"],
                                "best_val_psnr": summ["overall"]["psnr"],
                                "best_val_ssim": summ["overall"]["ssim"],
                                "input_val_psnr": summ["overall"]["in_psnr"],
                                "input_val_ssim": summ["overall"]["in_ssim"],
                                "best_epoch": best_epoch}.items() if v is not None})
            mlflow.log_artifact(str(run_dir / "val_summary.json"))
            if cfg["save_checkpoints"] and best_path.exists():
                mlflow.log_artifact(str(best_path), artifact_path="checkpoint")
        result = {"best_objective": summ["overall"]["objective"], "best_epoch": best_epoch,
                  "summary": summ, "run_dir": str(run_dir)}
    finally:
        if mlflow:
            mlflow.end_run()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. lr=5e-4 epochs=60")
    a = ap.parse_args()
    cfg = load_cfg(a.config, a.set)
    if a.root:
        cfg["root"] = a.root
    if a.run_name:
        cfg["run_name"] = a.run_name
    res = train(cfg)
    o = res["summary"]["overall"]
    print(f"\nBEST epoch {res['best_epoch']}: PSNR {o['psnr']:.2f} dB  SSIM {o['ssim']:.4f}  "
          f"(corrupted images without any restoration: PSNR {o['in_psnr']:.2f}  SSIM {o['in_ssim']:.4f})")
    for c, v in res["summary"]["by_corruption"].items():
        print(f"  {c:12s} PSNR {v['psnr']:.2f}  SSIM {v['ssim']:.4f}  (n={v['n']})")


if __name__ == "__main__":
    main()