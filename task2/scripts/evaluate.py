from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import yaml

from src import metrics
from src.datasets import ManifestDataset
from src.manifest import load_manifest
from task2.src.classifier import CorruptionClassifier
from task2.src.specialists import SpecialistAutoencoder


def load_model(model, path, device):
    state = torch.load(
        path,
        map_location=device,
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    manifest_path = config["data"]["test_manifest"]

    entries = load_manifest(manifest_path)

    dataset = ManifestDataset(
        config["data"]["test_arrays"],
        entries,
    )

    classifier = CorruptionClassifier(
        channels=config["classifier"]["channels"],
        dropout=config["classifier"]["dropout"],
    )

    load_model(
        classifier,
        config["classifier"]["checkpoint"],
        device,
    )

    specialists = {}

    for corruption in [
        "salt_pepper",
        "blur",
        "occlusion",
    ]:
        model = SpecialistAutoencoder(
            channels=tuple(
                config["specialist"]["channels"]
            ),
            bottleneck=config["specialist"]["bottleneck"],
        )

        load_model(
            model,
            config["specialist"]["checkpoints"][corruption],
            device,
        )

        specialists[corruption] = model

    results = []

    for i in range(len(dataset)):
        x, target, true_label = dataset[i]

        x = x.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(x)
            predicted_label = int(logits.argmax(dim=1).item())

        true_corruption = entries[i]["corruption"]
        severity = entries[i]["severity"]

        if true_label == 0:
            true_model = None
        else:
            true_model = specialists[true_corruption]

        if true_model is None:
            oracle_pred = x
        else:
            with torch.no_grad():
                oracle_pred = true_model(x)

        if predicted_label == 0:
            routed_pred = x
        else:
            predicted_corruption = [
                "clean",
                "salt_pepper",
                "blur",
                "occlusion",
            ][predicted_label]

            with torch.no_grad():
                routed_pred = specialists[
                    predicted_corruption
                ](x)

        oracle_l1 = torch.abs(
            oracle_pred - target
        ).mean().item()

        routed_l1 = torch.abs(
            routed_pred - target
        ).mean().item()

        oracle_ssim = metrics.ssim_per_image(
            oracle_pred,
            target,
        )[0]

        routed_ssim = metrics.ssim_per_image(
            routed_pred,
            target,
        )[0]

        oracle_psnr = metrics.psnr_per_image(
            oracle_pred,
            target,
        )[0]

        routed_psnr = metrics.psnr_per_image(
            routed_pred,
            target,
        )[0]

        results.append({
            "record_id": i,
            "kind": entries[i]["corruption"],
            "severity": severity,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "oracle_l1": oracle_l1,
            "oracle_ssim": oracle_ssim,
            "oracle_psnr": oracle_psnr,
            "predicted_l1": routed_l1,
            "predicted_ssim": routed_ssim,
            "predicted_psnr": routed_psnr,
        })

    os.makedirs(
        config["output"]["results_dir"],
        exist_ok=True,
    )

    output_path = os.path.join(
        config["output"]["results_dir"],
        "routing_results.csv",
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()