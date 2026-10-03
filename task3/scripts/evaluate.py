import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/content/genai-assignment1")

from task3.src.train import load_task2_models
from src import corruptions as C
from src.datasets import to_tensor


def ssim_value(x, y):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(
        x * x,
        3,
        1,
        1,
    ) - mu_x * mu_x

    sigma_y = F.avg_pool2d(
        y * y,
        3,
        1,
        1,
    ) - mu_y * mu_y

    sigma_xy = F.avg_pool2d(
        x * y,
        3,
        1,
        1,
    ) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    value = (
        (2 * mu_x * mu_y + c1)
        * (2 * sigma_xy + c2)
    ) / (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (sigma_x + sigma_y + c2)
    )

    return value.mean().item()


def psnr_value(x, y):
    mse = F.mse_loss(x, y).item()

    if mse == 0:
        return float("inf")

    return 10 * np.log10(1.0 / mse)


def update_stats(stats, l1, ssim, psnr):
    stats["count"] += 1
    stats["l1"] += l1
    stats["ssim"] += ssim
    stats["psnr"] += psnr


def average_stats(stats):
    if stats["count"] == 0:
        return {
            "count": 0,
            "l1": 0,
            "ssim": 0,
            "psnr": 0,
        }

    return {
        "count": stats["count"],
        "l1": stats["l1"] / stats["count"],
        "ssim": stats["ssim"] / stats["count"],
        "psnr": stats["psnr"] / stats["count"],
    }


def make_stats():
    return {
        "count": 0,
        "l1": 0,
        "ssim": 0,
        "psnr": 0,
    }


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

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    images = np.load(
        cfg["data"]["processed_dir"]
        + "/test_128.npy"
    )

    with open(
        cfg["data"]["manifests_dir"]
        + "/test_manifest.json",
        "r",
    ) as f:
        manifest = json.load(f)

    entries = manifest["entries"]

    results = {
        "no_restoration": make_stats(),
        "soft_moe": make_stats(),
    }

    by_corruption = {}
    by_severity = {}
    rows = []

    print("device:", device)
    print("test samples:", len(entries))
    print()

    for index, entry in enumerate(entries):
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

        target = to_tensor(
            clean
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            restored, weights, logits = model(x)

        restored = restored.clamp(
            0.0,
            1.0,
        )

        l1_no = F.l1_loss(
            x,
            target,
        ).item()

        ssim_no = ssim_value(
            x,
            target,
        )

        psnr_no = psnr_value(
            x,
            target,
        )

        l1_moe = F.l1_loss(
            restored,
            target,
        ).item()

        ssim_moe = ssim_value(
            restored,
            target,
        )

        psnr_moe = psnr_value(
            restored,
            target,
        )

        update_stats(
            results["no_restoration"],
            l1_no,
            ssim_no,
            psnr_no,
        )

        update_stats(
            results["soft_moe"],
            l1_moe,
            ssim_moe,
            psnr_moe,
        )

        corruption = entry["corruption"]
        severity = entry["severity"]

        if corruption not in by_corruption:
            by_corruption[corruption] = {
                "no_restoration": make_stats(),
                "soft_moe": make_stats(),
            }

        if severity not in by_severity:
            by_severity[severity] = {
                "no_restoration": make_stats(),
                "soft_moe": make_stats(),
            }

        update_stats(
            by_corruption[corruption]["no_restoration"],
            l1_no,
            ssim_no,
            psnr_no,
        )

        update_stats(
            by_corruption[corruption]["soft_moe"],
            l1_moe,
            ssim_moe,
            psnr_moe,
        )

        update_stats(
            by_severity[severity]["no_restoration"],
            l1_no,
            ssim_no,
            psnr_no,
        )

        update_stats(
            by_severity[severity]["soft_moe"],
            l1_moe,
            ssim_moe,
            psnr_moe,
        )

        rows.append(
            {
                "image_idx": entry["image_idx"],
                "corruption": corruption,
                "severity": severity,
                "no_l1": l1_no,
                "no_ssim": ssim_no,
                "no_psnr": psnr_no,
                "moe_l1": l1_moe,
                "moe_ssim": ssim_moe,
                "moe_psnr": psnr_moe,
                "clean_weight": float(
                    weights[0, 0].item()
                ),
                "salt_pepper_weight": float(
                    weights[0, 1].item()
                ),
                "blur_weight": float(
                    weights[0, 2].item()
                ),
                "occlusion_weight": float(
                    weights[0, 3].item()
                ),
            }
        )

        if (index + 1) % 1000 == 0:
            print(
                "processed:",
                index + 1,
                "/",
                len(entries),
            )

    results["no_restoration"] = average_stats(
        results["no_restoration"]
    )

    results["soft_moe"] = average_stats(
        results["soft_moe"]
    )

    for corruption in by_corruption:
        by_corruption[corruption][
            "no_restoration"
        ] = average_stats(
            by_corruption[corruption][
                "no_restoration"
            ]
        )

        by_corruption[corruption][
            "soft_moe"
        ] = average_stats(
            by_corruption[corruption][
                "soft_moe"
            ]
        )

    for severity in by_severity:
        by_severity[severity][
            "no_restoration"
        ] = average_stats(
            by_severity[severity][
                "no_restoration"
            ]
        )

        by_severity[severity][
            "soft_moe"
        ] = average_stats(
            by_severity[severity][
                "soft_moe"
            ]
        )

    results["by_corruption"] = by_corruption
    results["by_severity"] = by_severity

    results["checkpoint_epoch"] = int(
        checkpoint["epoch"]
    )

    results["checkpoint_score"] = float(
        checkpoint["score"]
    )

    results_dir = cfg["output"]["results_dir"]

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    json_path = (
        results_dir
        + "/evaluation_results.json"
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    csv_path = (
        results_dir
        + "/test_predictions.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("OVERALL RESULTS")

    for name in [
        "no_restoration",
        "soft_moe",
    ]:
        value = results[name]

        print(
            name,
            "| L1:",
            round(value["l1"], 6),
            "| SSIM:",
            round(value["ssim"], 6),
            "| PSNR:",
            round(value["psnr"], 4),
        )

    print()
    print("BY CORRUPTION")

    for name in [
        "clean",
        "salt_pepper",
        "blur",
        "occlusion",
    ]:
        value = results["by_corruption"][name]

        print()
        print(name)

        for method in [
            "no_restoration",
            "soft_moe",
        ]:
            metrics = value[method]

            print(
                method,
                "| L1:",
                round(metrics["l1"], 6),
                "| SSIM:",
                round(metrics["ssim"], 6),
                "| PSNR:",
                round(metrics["psnr"], 4),
            )

    print()
    print("BY SEVERITY")

    for name in [
        "clean",
        "low",
        "medium",
        "high",
    ]:
        value = results["by_severity"][name]

        print()
        print(name)

        for method in [
            "no_restoration",
            "soft_moe",
        ]:
            metrics = value[method]

            print(
                method,
                "| L1:",
                round(metrics["l1"], 6),
                "| SSIM:",
                round(metrics["ssim"], 6),
                "| PSNR:",
                round(metrics["psnr"], 4),
            )

    print()
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()