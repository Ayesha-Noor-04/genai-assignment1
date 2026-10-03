from __future__ import annotations

import argparse
import os
import json

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import RuntimeCorruptionDataset
from task2.src.losses import L1SSIMLoss
from task2.src.specialists import SpecialistAutoencoder


def objective(trial, config, corruption):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    processed_dir = config["data"]["processed_dir"]

    trainval = np.load(
        os.path.join(
            processed_dir,
            "trainval_128.npy",
        )
    )

    with open(
        os.path.join(
            processed_dir,
            "split.json",
        ),
        "r",
    ) as f:
        split = json.load(f)

    train_dataset = RuntimeCorruptionDataset(
        trainval,
        split["train_idx"],
        seed=config["seed"],
        corruption=corruption,
    )

    val_dataset = RuntimeCorruptionDataset(
        trainval,
        split["val_idx"],
        seed=config["seed"] + 1,
        corruption=corruption,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    channels = trial.suggest_categorical(
        "channels",
        [
            "small",
            "medium",
            "wide",
        ],
    )

    channel_values = {
        "small": [16, 32, 64, 128],
        "medium": [32, 64, 128, 256],
        "wide": [64, 128, 256, 512],
    }

    bottleneck = trial.suggest_categorical(
        "bottleneck",
        [128, 256, 512],
    )

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True,
    )

    alpha = trial.suggest_float(
        "alpha",
        0.5,
        0.95,
    )

    model = SpecialistAutoencoder(
        channels=tuple(channel_values[channels]),
        bottleneck=bottleneck,
    ).to(device)

    criterion = L1SSIMLoss(alpha=alpha)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config["training"]["weight_decay"],
    )

    for epoch in range(config["optuna"]["epochs"]):
        model.train()

        for x, target, _ in train_loader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(x)
            loss, _, _ = criterion(pred, target)

            loss.backward()
            optimizer.step()

        model.eval()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for x, target, _ in val_loader:
                x = x.to(device)
                target = target.to(device)

                pred = model(x)
                loss, _, _ = criterion(pred, target)

                total_loss += loss.item()
                count += 1

        val_loss = total_loss / count

        trial.report(-val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return -val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corruption", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    study_name = (
        config["optuna"]["study_name"]
        + "_"
        + args.corruption
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=config["optuna"]["storage"],
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            config,
            args.corruption,
        ),
        n_trials=config["optuna"]["trials"],
    )

    print("corruption:", args.corruption)
    print("best value:", study.best_value)
    print("best params:", study.best_params)


if __name__ == "__main__":
    main()
