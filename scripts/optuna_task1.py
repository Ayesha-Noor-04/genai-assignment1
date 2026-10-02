"""Phase 4: Optuna hyper-parameter study for Task 1 (UDAE).

Colab:
    python scripts/optuna_task1.py --root /content/drive/MyDrive/genai_a1 --n_trials 30 --trial_epochs 12

* The study lives in a SQLite file under --root, so it survives Colab disconnects:
  just run the same command again and it continues until --n_trials trials exist.
* Each trial = one short training run (--trial_epochs) logged to MLflow (experiment task1_optuna).
* Objective (maximised) = 0.5*SSIM + 0.5*PSNR/40 on the fixed VALIDATION manifest (src/metrics.py).
  It does not use the test set and does not depend on alpha, so alpha can be tuned fairly.
* MedianPruner stops clearly-bad trials early.
* --space v1 : first study (vector bottleneck only).  --space v2 : second study after the diagnostic
  (searches the bottleneck type/grid, wider latent and weight-decay ranges, horizontal flip ON).
  The two studies live side by side (different study names, output folders and best-config files).

Outputs under <root>/optuna/ : trials.csv, study_summary.json, 3 PNG plots; and <root>/task1_best.yaml
(the selected configuration, used for the final training run).
"""
import argparse
import json
import sys
from pathlib import Path

import optuna
import yaml
from optuna.storages import RDBStorage, RetryFailedTrialCallback

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_task1 import DEFAULTS, train  # noqa: E402

STUDY_NAME = "task1_udae"

# ---- search spaces (reported in the paper) ------------------------------------------------
SPACE_V1 = {
    "lr": "log-uniform [1e-4, 3e-3]",
    "batch_size": "categorical {16, 32, 64}",
    "bottleneck_dim": "categorical {128, 256, 512, 1024}",
    "base_channels": "categorical {32, 48, 64}  (encoder channels = c, 2c, 4c, 8c, 8c)",
    "dropout": "uniform [0.0, 0.5] (on the latent)",
    "alpha": "uniform [0.5, 0.95]  (L1 weight; SSIM weight = 1 - alpha)",
    "weight_decay": "log-uniform [1e-6, 1e-3]",
}
SPACE_V2 = {
    "lr": "log-uniform [1e-4, 3e-3]",
    "batch_size": "categorical {16, 32, 64}",
    "latent": "categorical {vector, spatial}",
    "latent_size": "categorical {8, 16}  (only if latent = spatial: latent grid size)",
    "bottleneck_dim": "categorical {512, 1024, 2048, 4096}  (number of latent values)",
    "base_channels": "categorical {32, 48, 64}",
    "dropout": "uniform [0.0, 0.5] (on the latent)",
    "alpha": "uniform [0.5, 0.95]  (L1 weight; SSIM weight = 1 - alpha)",
    "weight_decay": "log-uniform [1e-6, 1e-2]",
    "(fixed)": "horizontal-flip augmentation of the clean training image = on",
}
SMOKE_SPACE_NOTE = "smoke mode: tiny model, 2 epochs - only for checking that the pipeline runs"


def suggest(trial, space="v1", smoke=False):
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "base_channels": trial.suggest_categorical("base_channels", [32, 48, 64]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "alpha": trial.suggest_float("alpha", 0.5, 0.95),
    }
    if space == "v1":
        p["bottleneck_dim"] = trial.suggest_categorical("bottleneck_dim", [128, 256, 512, 1024])
        p["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    else:
        p["latent"] = trial.suggest_categorical("latent", ["vector", "spatial"])
        p["latent_size"] = trial.suggest_categorical("latent_size", [8, 16]) if p["latent"] == "spatial" else 8
        p["bottleneck_dim"] = trial.suggest_categorical("bottleneck_dim", [512, 1024, 2048, 4096])
        p["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        p["hflip"] = True
    if smoke:  # keep CPU dry-runs fast; the sampled value is still recorded by Optuna
        p["base_channels"], p["batch_size"] = 8, 16
    return p


def make_objective(args):
    def objective(trial):
        cfg = dict(DEFAULTS)
        cfg.update(suggest(trial, args.space, args.smoke))
        cfg.update(root=args.root, epochs=args.trial_epochs, patience=10 ** 6,  # pruner decides, not patience
                   run_name=f"trial_{trial.number:03d}",
                   experiment="task1_optuna" if args.space == "v1" else "task1_optuna_v2",
                   save_checkpoints=False, resume=False, sample_every=0,
                   num_workers=args.num_workers, seed=42)
        res = train(cfg, trial=trial)
        trial.set_user_attr("best_epoch", res["best_epoch"])
        o = res["summary"]["overall"]
        trial.set_user_attr("val_psnr_last_epoch", o["psnr"])
        trial.set_user_attr("val_ssim_last_epoch", o["ssim"])
        return res["best_objective"]
    return objective


def report(study, args, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from optuna.importance import get_param_importances

    SPACE = SPACE_V1 if args.space == "v1" else SPACE_V2
    states = optuna.trial.TrialState
    complete = [t for t in study.trials if t.state == states.COMPLETE]
    pruned = [t for t in study.trials if t.state == states.PRUNED]
    failed = [t for t in study.trials if t.state == states.FAIL]
    best = study.best_trial
    summary = {
        "study_name": study.study_name, "sampler": "TPE (seed 42)",
        "pruner": "MedianPruner(n_startup_trials=5, n_warmup_steps=3)", "space_version": args.space,
        "objective": "maximise 0.5*SSIM + 0.5*PSNR/40 on the validation manifest",
        "search_space": SPACE, "epochs_per_trial": args.trial_epochs,
        "n_trials_total": len(study.trials), "n_complete": len(complete),
        "n_pruned": len(pruned), "n_failed": len(failed),
        "best_trial": {"number": best.number, "objective": best.value, "params": best.params,
                       "user_attrs": best.user_attrs},
    }
    if args.smoke:
        summary["note"] = SMOKE_SPACE_NOTE
    with open(out_dir / "study_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)

    # best configuration -> yaml for the final run
    best_cfg = {k: best.params[k] for k in SPACE if k in best.params}
    if args.space == "v2":
        best_cfg["hflip"] = True
        best_cfg.setdefault("latent_size", 8)
    if args.smoke:
        best_cfg["base_channels"], best_cfg["batch_size"] = 8, 16
    best_cfg.update(epochs=60, patience=12, num_workers=2, amp=True, seed=42, sample_every=5)
    best_yaml = Path(args.root) / ("task1_best.yaml" if args.space == "v1" else "task1_best_v2.yaml")
    with open(best_yaml, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    # plots
    vals = [(t.number, t.value) for t in complete]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if vals:
        xs, ys = zip(*vals)
        ax.scatter(xs, ys, label="complete", s=22)
        ax.plot(xs, np.maximum.accumulate(ys), color="C1", label="best so far")
    if pruned:
        ax.scatter([t.number for t in pruned], [t.intermediate_values[max(t.intermediate_values)]
                                                if t.intermediate_values else np.nan for t in pruned],
                   marker="x", color="gray", label="pruned (last value)")
    ax.set_xlabel("trial"); ax.set_ylabel("validation objective"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out_dir / "optuna_history.png", dpi=150); plt.close(fig)

    try:
        imp = get_param_importances(study)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        ax.set_xlabel("importance (fANOVA)"); fig.tight_layout()
        fig.savefig(out_dir / "optuna_importance.png", dpi=150); plt.close(fig)
        summary["param_importance"] = imp
        with open(out_dir / "study_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:  # too few completed trials
        print("importance plot skipped:", e)

    names = [k for k in SPACE if not k.startswith("(")]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
    for ax, n in zip(axes.ravel(), names):
        ts = [t for t in complete if n in t.params]
        ax.scatter([str(t.params[n]) if isinstance(t.params[n], str) else t.params[n] for t in ts],
                   [t.value for t in ts], s=18)
        if n in ("lr", "weight_decay"):
            ax.set_xscale("log")
        ax.set_xlabel(n); ax.grid(alpha=.3)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("objective"); axes[1, 0].set_ylabel("objective")
    fig.tight_layout(); fig.savefig(out_dir / "optuna_slices.png", dpi=150); plt.close(fig)

    print("\n=== Optuna summary ===")
    print(f"trials: {len(study.trials)} (complete {len(complete)}, pruned {len(pruned)}, failed {len(failed)})")
    print(f"best trial #{best.number}: objective {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:15s} {v}")
    print(f"\nwrote {out_dir}/ and {best_yaml}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n_trials", type=int, default=30, help="TOTAL trials wanted in the study")
    ap.add_argument("--trial_epochs", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=None, help="stop starting new trials after N seconds")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--space", choices=["v1", "v2"], default="v1")
    ap.add_argument("--smoke", action="store_true", help="tiny CPU-friendly dry run")
    args = ap.parse_args()
    if args.smoke:
        args.trial_epochs = min(args.trial_epochs, 2)

    root = Path(args.root)
    out_dir = root / ("optuna" if args.space == "v1" else "optuna_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = STUDY_NAME + ("" if args.space == "v1" else "_v2") + ("_smoke" if args.smoke else "")
    storage = RDBStorage(f"sqlite:///{root / 'optuna_task1.db'}", heartbeat_interval=60, grace_period=180,
                         failed_trial_callback=RetryFailedTrialCallback(max_retry=1))
    study = optuna.create_study(
        study_name=name, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))

    done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                         optuna.trial.TrialState.PRUNED)])
    remaining = max(0, args.n_trials - done)
    print(f"study '{name}': {done} finished trials, running {remaining} more")
    if remaining:
        study.optimize(make_objective(args), n_trials=remaining, timeout=args.timeout, gc_after_trial=True)
    report(study, args, out_dir)


if __name__ == "__main__":
    main()