import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/content/genai-assignment1")

from task3.src.train import load_task2_models
from src import corruptions as C
from src.datasets import to_tensor


def main():
    model, cfg, device = load_task2_models(
        "task3/configs/soft_moe.yaml"
    )

    checkpoint_path = (
        cfg["output"]["checkpoint_dir"]
        + "/final_best.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

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

    entries = val_manifest["entries"]

    class_names = [
        "clean",
        "salt_pepper",
        "blur",
        "occlusion",
    ]

    severities = [
        "clean",
        "low",
        "medium",
        "high",
    ]

    routing = np.zeros(
        (len(class_names), 4),
        dtype=np.float64,
    )

    counts = np.zeros(
        len(class_names),
        dtype=np.int64,
    )

    all_weights = []
    all_labels = []

    print("device:", device)
    print("validation samples:", len(entries))

    for entry in entries:
        clean = images[entry["image_idx"]]

        corrupted = C.apply_corruption(
            clean,
            entry["corruption"],
            entry["params"],
            entry["seed"],
        )

        x = to_tensor(
            corrupted
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            _, weights, _ = model(x)

        weights = weights[0].cpu().numpy()

        label = int(entry["label"])

        routing[label] += weights
        counts[label] += 1

        all_weights.append(weights)
        all_labels.append(label)

    routing = routing / counts[:, None]

    all_weights = np.asarray(all_weights)
    all_labels = np.asarray(all_labels)

    dominant = np.argmax(
        all_weights,
        axis=1,
    )

    max_weights = np.max(
        all_weights,
        axis=1,
    )

    dominant_fraction = np.mean(
        max_weights > 0.9
    )

    column_means = all_weights.mean(
        axis=0
    )

    off_diagonal = []

    for i in range(4):
        mask = all_labels == i

        if np.any(mask):
            value = routing[i].copy()
            value[i] = 0
            off_diagonal.append(
                value.sum()
            )

    off_diagonal_mean = float(
        np.mean(off_diagonal)
    )

    entropy = -np.sum(
        all_weights
        * np.log(
            np.clip(
                all_weights,
                1e-8,
                1.0,
            )
        ),
        axis=1,
    )

    mean_entropy = float(
        entropy.mean()
    )

    print()
    print("ROUTING MATRIX")
    print("rows: true class")
    print("columns: clean, salt_pepper, blur, occlusion")
    print()

    print(
        np.round(
            routing,
            4,
        )
    )

    print()
    print("COLUMN MEANS")
    print(
        np.round(
            column_means,
            4,
        )
    )

    print()
    print(
        "dominant routing fraction (>0.9):",
        round(
            float(dominant_fraction),
            4,
        ),
    )

    print(
        "mean off-diagonal mass:",
        round(
            off_diagonal_mean,
            4,
        ),
    )

    print(
        "mean routing entropy:",
        round(
            mean_entropy,
            4,
        ),
    )

    collapse = (
        routing.diagonal().min() < 0.15
        or column_means.max() > 0.6
    )

    print(
        "routing collapse:",
        collapse,
    )

    os.makedirs(
        cfg["output"]["figures_dir"],
        exist_ok=True,
    )

    os.makedirs(
        cfg["output"]["results_dir"],
        exist_ok=True,
    )

    result = {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "checkpoint_score": float(
            checkpoint["score"]
        ),
        "routing_matrix": routing.tolist(),
        "column_means": column_means.tolist(),
        "dominant_fraction_gt_0_9": float(
            dominant_fraction
        ),
        "mean_off_diagonal_mass": off_diagonal_mean,
        "mean_entropy": mean_entropy,
        "routing_collapse": bool(collapse),
    }

    result_path = (
        cfg["output"]["results_dir"]
        + "/routing_stats.json"
    )

    with open(
        result_path,
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    figure_path = (
        cfg["output"]["figures_dir"]
        + "/routing_heatmap.png"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        routing,
        aspect="auto",
    )

    plt.colorbar(
        label="Mean routing weight"
    )

    plt.xticks(
        range(4),
        [
            "Clean",
            "Salt-pepper",
            "Blur",
            "Occlusion",
        ],
    )

    plt.yticks(
        range(4),
        [
            "Clean",
            "Salt-pepper",
            "Blur",
            "Occlusion",
        ],
    )

    for i in range(4):
        for j in range(4):
            plt.text(
                j,
                i,
                f"{routing[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.xlabel(
        "Selected branch"
    )

    plt.ylabel(
        "True corruption"
    )

    plt.title(
        "Soft MoE routing matrix"
    )

    plt.tight_layout()
    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print("routing statistics:", result_path)
    print("routing heatmap:", figure_path)


if __name__ == "__main__":
    main()