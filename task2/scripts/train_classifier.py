from __future__ import annotations

import argparse
import os
import json

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from task2.src.classifier import CorruptionClassifier
from task2.src.dataset import BalancedBatchSampler, ClassifierDataset


def load_data(processed_dir):
    trainval = np.load(os.path.join(processed_dir, "trainval_128.npy"))
    with open(os.path.join(processed_dir, "split.json"), "r") as f:
        split = json.load(f)

    train_indices = split["train_idx"]
    val_indices = split["val_idx"]

    return trainval, train_indices, val_indices


def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
    )

    return accuracy, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    trainval, train_indices, val_indices = load_data(
        config["data"]["processed_dir"]
    )

    train_dataset = ClassifierDataset(
        trainval,
        train_indices,
        seed=config["seed"],
    )

    val_dataset = ClassifierDataset(
        trainval,
        val_indices,
        seed=config["seed"] + 1,
    )

    batch_size = config["training"]["batch_size"]

    train_sampler = BalancedBatchSampler(
        train_dataset.labels,
        batch_size,
        seed=config["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CorruptionClassifier(
        channels=config["model"]["channels"],
        dropout=config["model"]["dropout"],
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    mlflow.set_tracking_uri(
        config["mlflow"]["tracking_uri"]
    )

    mlflow.set_experiment(
        config["mlflow"]["experiment"]
    )

    with mlflow.start_run():
        mlflow.log_params({
            "channels": config["model"]["channels"],
            "dropout": config["model"]["dropout"],
            "batch_size": batch_size,
            "lr": config["training"]["lr"],
            "weight_decay": config["training"]["weight_decay"],
        })

        for epoch in range(config["training"]["epochs"]):
            model.train()

            total_loss = 0.0
            count = 0

            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)
                loss = criterion(logits, y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

            train_loss = total_loss / count

            val_accuracy, val_f1 = evaluate(
                model,
                val_loader,
                device,
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
            }, step=epoch)

            print(
                f"epoch {epoch + 1}/{config['training']['epochs']} "
                f"loss={train_loss:.4f} "
                f"accuracy={val_accuracy:.4f} "
                f"macro_f1={val_f1:.4f}"
            )

        os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)

        torch.save(
            model.state_dict(),
            os.path.join(
                config["output"]["checkpoint_dir"],
                "classifier.pt",
            ),
        )


if __name__ == "__main__":
    main()