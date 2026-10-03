import json
import sys

import numpy as np
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
    save_checkpoint,
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


def main():
    with open(
        "task3/configs/soft_moe.yaml",
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    model, cfg, device = load_task2_models(
        "task3/configs/soft_moe.yaml"
    )

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

    warmup_epochs = cfg["training"]["warmup_epochs"]
    joint_epochs = cfg["training"]["joint_epochs"]

    model.temperature = temperature

    print("device:", device)
    print("training samples:", len(train_dataset))
    print("validation samples:", len(val_dataset))
    print("batch size:", batch_size)
    print("learning rate:", lr)
    print("temperature:", temperature)
    print("alpha:", alpha)
    print("lam_c:", lam_c)
    print("lam_b:", lam_b)

    print(
        "training corruption counts:",
        np.bincount(
            [entry["label"] for entry in train_entries]
        ),
    )

    freeze_experts(model)

    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=lr,
    )

    best_score = float("inf")

    checkpoint_path = (
        cfg["output"]["checkpoint_dir"]
        + "/final_best.pt"
    )

    history = []

    print()
    print("WARM-UP")

    for epoch in range(warmup_epochs):
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

        history.append(
            {
                "stage": "warmup",
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_l1": val_metrics["l1"],
                "val_ssim_loss": val_metrics["ssim_loss"],
                "score": score,
            }
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

    unfreeze_experts(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print()
    print("JOINT TRAINING")

    for epoch in range(joint_epochs):
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

        history.append(
            {
                "stage": "joint",
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_l1": val_metrics["l1"],
                "val_ssim_loss": val_metrics["ssim_loss"],
                "score": score,
            }
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
                warmup_epochs + epoch + 1,
                score,
                checkpoint_path,
            )

            print("saved best checkpoint")

    history_path = (
        cfg["output"]["results_dir"]
        + "/final_training_history.json"
    )

    import os

    os.makedirs(
        cfg["output"]["results_dir"],
        exist_ok=True,
    )

    with open(history_path, "w") as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print("BEST VALIDATION SCORE:", best_score)
    print("CHECKPOINT:", checkpoint_path)
    print("HISTORY:", history_path)


if __name__ == "__main__":
    main()