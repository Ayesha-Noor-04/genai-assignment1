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


def run_experiment(
    name,
    trainval,
    train_indices,
    val_indices,
    config,
    device,
    corruption,
    bottleneck,
    bottleneck_type,
    spatial_size,
):
    train_dataset = RuntimeCorruptionDataset(
        trainval,
        train_indices,
        seed=config["seed"],
        corruption=corruption,
    )

    val_dataset = RuntimeCorruptionDataset(
        trainval,
        val_indices,
        seed=config["seed"] + 1,
        corruption=corruption,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = SpecialistAutoencoder(
        channels=(16, 32, 64, 128),
        bottleneck=bottleneck,
        bottleneck_type=bottleneck_type,
        spatial_size=spatial_size,
    ).to(device)

    criterion = L1SSIMLoss(alpha=0.8)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0002,
        weight_decay=0.00001,
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for epoch in range(15):
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
            f"epoch {epoch + 1}/15 "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corruption", required=True)
    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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

    experiments = [
        (
            "A: vector 1024",
            1024,
            "vector",
            8,
        ),
        (
            "B: spatial 8x8 1024",
            1024,
            "spatial",
            8,
        ),
        (
            "C: spatial 8x8 4096",
            4096,
            "spatial",
            8,
        ),
        (
            "D: spatial 16x16 4096",
            4096,
            "spatial",
            16,
        ),
    ]

    for (
        name,
        bottleneck,
        bottleneck_type,
        spatial_size,
    ) in experiments:
        run_experiment(
            name,
            trainval,
            train_indices,
            val_indices,
            config,
            device,
            args.corruption,
            bottleneck,
            bottleneck_type,
            spatial_size,
        )


if __name__ == "__main__":
    main()