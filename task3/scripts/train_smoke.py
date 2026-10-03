import json
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "/content/genai-assignment1")

from task3.src.train import load_task2_models
from task3.src.train_loop import (
    MoEDataset,
    freeze_experts,
    unfreeze_experts,
    train_epoch,
    evaluate,
    save_checkpoint,
)

from src import corruptions as C
from torch.utils.data import DataLoader


def main():
    with open("task3/configs/soft_moe.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    model, cfg, device = load_task2_models(
        "task3/configs/soft_moe.yaml"
    )

    processed_dir = cfg["data"]["processed_dir"]
    manifests_dir = cfg["data"]["manifests_dir"]

    images = np.load(
        processed_dir + "/trainval_128.npy"
    )

    labels = np.load(
        processed_dir + "/labels_trainval.npy"
    )

    with open(
        manifests_dir + "/val_manifest.json",
        "r",
    ) as f:
        val_manifest = json.load(f)

    train_entries = []

    rng = np.random.default_rng(42)

    for i in range(len(images)):
        label = int(labels[i])

        if label == 0:
            corruption = "clean"
        elif label == 1:
            corruption = "salt_pepper"
        elif label == 2:
            corruption = "blur"
        else:
            corruption = "occlusion"

        if corruption == "clean":
            params = {}
            seed = 42 + i
        else:
            name, params, seed = C.sample_corruption(
                rng,
                corruption,
            )

        train_entries.append(
            {
                "image_idx": i,
                "corruption": corruption,
                "label": label,
                "severity": "train",
                "params": params,
                "seed": seed,
            }
        )

    val_entries = val_manifest["entries"]

    train_dataset = MoEDataset(
        images,
        train_entries,
    )

    val_dataset = MoEDataset(
        images,
        val_entries,
    )

    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    alpha = cfg["training"]["alpha"]
    lam_c = cfg["training"]["lam_c"]
    lam_b = cfg["training"]["lam_b"]
    temperature = cfg["model"]["temperature"]
    lr = cfg["training"]["lr"]

    print("device:", device)
    print("training samples:", len(train_dataset))
    print("validation samples:", len(val_dataset))
    print()

    freeze_experts(model)

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=lr,
    )

    print("WARM-UP")

    for epoch in range(
        cfg["training"]["warmup_epochs"]
    ):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        print(
            "epoch",
            epoch + 1,
            "| train loss:",
            round(train_metrics["loss"], 4),
            "| val L1:",
            round(val_metrics["l1"], 4),
            "| val SSIM loss:",
            round(val_metrics["ssim_loss"], 4),
        )

    unfreeze_experts(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print()
    print("JOINT TRAINING")

    best_score = float("inf")

    checkpoint_path = (
        cfg["output"]["checkpoint_dir"]
        + "/smoke_best.pt"
    )

    for epoch in range(
        cfg["training"]["joint_epochs"]
    ):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            alpha,
            lam_c,
            lam_b,
            temperature,
            device,
        )

        score = (
            val_metrics["l1"]
            + val_metrics["ssim_loss"]
        )

        print(
            "epoch",
            epoch + 1,
            "| train loss:",
            round(train_metrics["loss"], 4),
            "| val L1:",
            round(val_metrics["l1"], 4),
            "| val SSIM loss:",
            round(val_metrics["ssim_loss"], 4),
            "| score:",
            round(score, 4),
        )

        if score < best_score:
            best_score = score

            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                score,
                checkpoint_path,
            )

    print()
    print("best validation score:", best_score)
    print("checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()