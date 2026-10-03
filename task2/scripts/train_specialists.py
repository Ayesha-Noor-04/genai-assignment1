from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import RuntimeCorruptionDataset
from task2.src.losses import L1SSIMLoss
from task2.src.specialists import SpecialistAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corruption", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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

    train_indices = split["train_indices"]
    val_indices = split["val_indices"]

    train_dataset = RuntimeCorruptionDataset(
        trainval,
        train_indices,
        seed=config["seed"],
        corruption=args.corruption,
    )

    val_dataset = RuntimeCorruptionDataset(
        trainval,
        val_indices,
        seed=config["seed"] + 1,
        corruption=args.corruption,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = SpecialistAutoencoder(
        channels=tuple(config["model"]["channels"]),
        bottleneck=config["model"]["bottleneck"],
    ).to(device)

    criterion = L1SSIMLoss(
        alpha=config["training"]["alpha"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    for epoch in range(config["training"]["epochs"]):
        model.train()

        total_loss = 0.0
        count = 0

        for x, target, _ in train_loader:
            x = x.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(x)
            loss, _, _ = criterion(pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        train_loss = total_loss / count

        model.eval()

        total_val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for x, target, _ in val_loader:
                x = x.to(device)
                target = target.to(device)

                pred = model(x)
                loss, _, _ = criterion(pred, target)

                total_val_loss += loss.item()
                val_count += 1

        val_loss = total_val_loss / val_count

        print(
            f"{args.corruption} "
            f"epoch {epoch + 1}/{config['training']['epochs']} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

    os.makedirs(
        config["output"]["checkpoint_dir"],
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        os.path.join(
            config["output"]["checkpoint_dir"],
            f"{args.corruption}.pt",
        ),
    )


if __name__ == "__main__":
    main()