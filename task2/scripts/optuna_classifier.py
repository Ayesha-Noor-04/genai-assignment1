from __future__ import annotations

import argparse
import os

import mlflow
import numpy as np
import optuna
import torch
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from task2.src.classifier import CorruptionClassifier
from task2.src.dataset import BalancedBatchSampler, ClassifierDataset


def objective(trial, config):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    trainval = np.load(
        os.path.join(
            config["data"]["processed_dir"],
            "trainval_128.npy",
        )
    )

    split = np.load(
        os.path.join(
            config["data"]["processed_dir"],
            "split.json",
        ),
        allow_pickle=True,
    ).item()

    train_dataset = ClassifierDataset(
        trainval,
        split["train_indices"],
        seed=config["seed"],
    )

    val_dataset = ClassifierDataset(
        trainval,
        split["val_indices"],
        seed=config["seed"] + 1,
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [16, 32, 64],
    )

    sampler = BalancedBatchSampler(
        train_dataset.labels,
        batch_size,
        seed=config["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
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
        ["small", "medium", "wide"],
    )

    dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.5,
    )

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    model = CorruptionClassifier(
        channels=channels,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["optuna"]["epochs"]):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        model.eval()

        labels = []
        predictions = []

        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                pred = logits.argmax(dim=1).cpu().numpy()

                predictions.extend(pred)
                labels.extend(y.numpy())

        macro_f1 = f1_score(
            labels,
            predictions,
            average="macro",
        )

        trial.report(macro_f1, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    storage = config["optuna"]["storage"]
    study_name = config["optuna"]["study_name"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config["optuna"]["trials"],
    )

    print("best value:", study.best_value)
    print("best params:", study.best_params)


if __name__ == "__main__":
    main()