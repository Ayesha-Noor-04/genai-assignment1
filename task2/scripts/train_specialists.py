
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.datasets import RuntimeCorruptionDataset
from task2.src.losses import L1SSIMLoss
from task2.src.specialists import SpecialistAutoencoder


BEST_PARAMS = {
    "salt_pepper": {
        "batch_size": 64,
        "channels": (64, 128, 256, 512),
        "bottleneck": 256,
        "lr": 0.00021899358780305865,
        "alpha": 0.9480757313636454,
    },
    "blur": {
        "batch_size": 32,
        "channels": (64, 128, 256, 512),
        "bottleneck": 512,
        "lr": 0.00018948657703433632,
        "alpha": 0.9365441078282344,
    },
    "occlusion": {
        "batch_size": 16,
        "channels": (64, 128, 256, 512),
        "bottleneck": 128,
        "lr": 0.000032187933903320205,
        "alpha": 0.9483476589754145,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corruption", required=True)
    args = parser.parse_args()

    if args.corruption not in BEST_PARAMS:
        raise ValueError(
            f"Unknown corruption: {args.corruption}. "
            f"Expected one of: {list(BEST_PARAMS.keys())}"
        )

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    params = BEST_PARAMS[args.corruption]

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
        os.path.join(processed_dir, "split.json"),
        "r",
    ) as f:
        split = json.load(f)

    train_indices = split["train_idx"]
    val_indices = split["val_idx"]

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
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = SpecialistAutoencoder(
        channels=params["channels"],
        bottleneck=params["bottleneck"],
    ).to(device)

    criterion = L1SSIMLoss(
        alpha=params["alpha"]
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    print(f"Training {args.corruption} specialist")
    print(f"device: {device}")
    print(f"batch_size: {params['batch_size']}")
    print(f"channels: {params['channels']}")
    print(f"bottleneck: {params['bottleneck']}")
    print(f"lr: {params['lr']}")
    print(f"alpha: {params['alpha']}")
    print(f"epochs: {config['training']['epochs']}")

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

    checkpoint_path = os.path.join(
        config["output"]["checkpoint_dir"],
        f"{args.corruption}.pt",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
