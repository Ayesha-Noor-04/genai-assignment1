"""Architecture diagnostic for Task 1: short, equal-budget runs of the candidate bottleneck designs.

Colab:
    python scripts/diagnose_task1.py --root /content/drive/MyDrive/genai_a1 \
        --base_config /content/drive/MyDrive/genai_a1/task1_best.yaml --epochs 20

All runs share the base config (learning rate, batch size, channels, dropout, alpha, weight decay
from the first Optuna study) and horizontal-flip augmentation; ONLY the bottleneck design changes:

    A  vector  1024     (first design, re-run with hflip)
    B  spatial 1024     8x8 grid x 16 ch
    C  spatial 4096     8x8 grid x 64 ch
    D  spatial 4096     16x16 grid x 16 ch

Results: <root>/results/task1/diagnostic.csv (+ printed table). Runs are logged to MLflow
(experiment task1_diagnostic) and each keeps its best checkpoint and sample grids under
<root>/runs/diag_<name>/ so the colours of the reconstructions can be inspected. Re-running skips
nothing: finished runs resume instantly, so an interrupted Colab session can simply be restarted.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import UDAE, count_params  # noqa: E402
from src.train_task1 import DEFAULTS, train  # noqa: E402

VARIANTS = {
    "A_vector1024": dict(latent="vector", latent_size=8, bottleneck_dim=1024),
    "B_spatial1024": dict(latent="spatial", latent_size=8, bottleneck_dim=1024),
    "C_spatial4096": dict(latent="spatial", latent_size=8, bottleneck_dim=4096),
    "D_spatial4096_16x16": dict(latent="spatial", latent_size=16, bottleneck_dim=4096),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base_config", default=None, help="yaml with lr, batch_size, base_channels, ... (e.g. task1_best.yaml)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--runs", nargs="*", default=list(VARIANTS), help="subset of variants to run")
    ap.add_argument("--no_hflip", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU dry run")
    a = ap.parse_args()

    base = dict(DEFAULTS)
    if a.base_config:
        with open(a.base_config) as f:
            base.update(yaml.safe_load(f))
    rows = []
    for name in a.runs:
        cfg = dict(base)
        cfg.update(VARIANTS[name])
        cfg.update(root=a.root, epochs=a.epochs, patience=10 ** 6, hflip=not a.no_hflip,
                   run_name=f"diag_{name}", experiment="task1_diagnostic",
                   save_checkpoints=True, sample_every=10, seed=42)
        if a.smoke:
            cfg.update(base_channels=8, batch_size=16, epochs=min(a.epochs, 2), sample_every=0)
        print(f"\n===== {name}: {VARIANTS[name]} =====")
        res = train(cfg)
        s = res["summary"]
        n_params = count_params(UDAE(cfg["base_channels"], cfg["bottleneck_dim"], cfg["dropout"],
                                     latent=cfg["latent"], latent_size=cfg["latent_size"]))
        row = {"variant": name, "latent_values": cfg["bottleneck_dim"],
               "compression_x": round(128 * 128 * 3 / cfg["bottleneck_dim"], 1), "params_M": round(n_params / 1e6, 2),
               "best_epoch": res["best_epoch"], "val_objective": s["overall"]["objective"],
               "val_psnr": s["overall"]["psnr"], "val_ssim": s["overall"]["ssim"]}
        for c, v in s["by_corruption"].items():
            row[f"psnr_{c}"] = v["psnr"]
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(a.root) / "results" / "task1"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "diagnostic.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nsaved {out / 'diagnostic.csv'}")


if __name__ == "__main__":
    main()