import json
import sys

import numpy as np
import optuna
import torch
import yaml

sys.path.insert(0, "/content/genai-assignment1")

from torch.utils.data import DataLoader

from task3.src.train import load_task2_models
from task3.src.train_loop import (
    MoEDataset,
    freeze_experts,
    unfreeze_experts,
    train_epoch,
    evaluate,
)

from src import corruptions as C


def build_entries(images):
    entries = []

    labels = np.arange(len(images)) % 4

    rng = np.random.default_rng(42)
    rng.shuffle(labels)

    for i in range(len(images)):
        label = int(labels[i])
        corruption = C.CLASSES[label]

        if corruption == "clean":
            params = {}
            seed = 42 + i
        else:
            name, params, seed = C.sample_corruption(
                rng,
                corruption,
            )

        entries.append(
            {
                "image_idx": i,
                "corruption": corruption,
                "label": label,
                "severity": "train",
                "params": params,
                "seed": seed,
            }
        )

    return entries


def objective(trial):
    with open(
        "task3/configs/soft_moe.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    lr = trial.suggest_float(
        "lr",
        1e-5,
        3e-4,
        log=True,
    )

    temperature = trial.suggest_float(
        "temperature",
        0.5,
        3.0,
        log=True,
    )

    lam_c = trial.suggest_float(
        "lam_c",
        0.01,
        0.5,
        log=True,
    )

    lam_b = trial.suggest_float(
        "lam_b",
        1e-3,
        0.1,
        log=True,
    )

    alpha = trial.suggest_float(
        "alpha",
        0.5,
        0.95,
    )

    model, cfg, device = load_task2_models(
        "task3/configs/soft_moe.yaml"
    )

    model.temperature = temperature

    images = np.load(
        cfg["data"]["processed_dir"]
        + "/trainval_128.npy"
    )

    with open(
        cfg["data"]["manifests_dir"]
        + "/val_manifest.json",
        "r",
    ) as f:
        val_manifest = json.load(f)

    train_entries = build_entries(images)
    val_entries = val_manifest["entries"]

    train_dataset = MoEDataset(
        images,
        train_entries,
    )

    val_dataset = MoEDataset(
        images,
        val_entries,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    freeze_experts(model)

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=lr,
    )

    for epoch in range(2):
        train_epoch(
            model,
            train_loader,
            optimizer,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        value = evaluate(
            model,
            val_loader,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        score = (
            value["l1"]
            + value["ssim_loss"]
        )

        trial.report(score, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    unfreeze_experts(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    for epoch in range(3):
        train_epoch(
            model,
            train_loader,
            optimizer,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        value = evaluate(
            model,
            val_loader,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        score = (
            value["l1"]
            + value["ssim_loss"]
        )

        trial.report(score, epoch + 2)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


def main():
    study = optuna.create_study(
        direction="minimize",
        study_name="task3_soft_moe",
        storage=(
            "sqlite:////content/drive/MyDrive/"
            "genai_a1/task3/optuna.db"
        ),
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
        ),
    )

    study.optimize(
        objective,
        n_trials=15,
    )

    print()
    print("BEST SCORE:")
    print(study.best_value)

    print()
    print("BEST PARAMETERS:")

    for key, value in study.best_params.items():
        print(key, "=", value)


if __name__ == "__main__":
    main()