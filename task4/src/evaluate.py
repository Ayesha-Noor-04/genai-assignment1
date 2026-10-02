import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

from .dataset import FS2KDataset
from .models import ConditionalGenerator


def denormalize(x):
    """Convert [-1, 1] tensor values to [0, 1]."""
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def calculate_metrics(fake, real):
    """
    Calculate image reconstruction metrics.

    Inputs:
        fake: [C, H, W] tensor in [0, 1]
        real: [C, H, W] tensor in [0, 1]
    """

    fake_np = (
        fake.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    real_np = (
        real.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    fake_np = np.clip(fake_np, 0.0, 1.0)
    real_np = np.clip(real_np, 0.0, 1.0)

    # Mean Absolute Error
    l1 = np.mean(
        np.abs(fake_np - real_np)
    )

    # Mean Squared Error
    mse = np.mean(
        (fake_np - real_np) ** 2
    )

    # PSNR
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(1.0 / mse)

    # SSIM
    ssim = structural_similarity(
        real_np,
        fake_np,
        channel_axis=2,
        data_range=1.0,
    )

    return {
        "l1": float(l1),
        "mse": float(mse),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }


def save_image(tensor, path):
    """Save a [C,H,W] tensor in [0,1] as an image."""

    array = (
        tensor.detach()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    array = np.clip(
        array * 255.0,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(array).save(path)


def load_generator(checkpoint_path, device):
    """Load the trained generator from a checkpoint."""

    generator = ConditionalGenerator().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    generator.load_state_dict(
        checkpoint["generator"]
    )

    generator.eval()

    epoch = checkpoint.get(
        "epoch",
        None,
    )

    return generator, epoch


def main(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = FS2KDataset(
        root=args.data_root,
        split="test",
        image_size=args.image_size,
    )

    print(
        f"Test samples: {len(dataset)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    generator, epoch = load_generator(
        args.checkpoint,
        device,
    )

    if epoch is not None:
        print(
            f"Loaded checkpoint from epoch {epoch}"
        )

    # --------------------------------------------------
    # Output directories
    # --------------------------------------------------

    output_dir = Path(args.output_dir)

    generated_dir = (
        output_dir / "generated"
    )

    generated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    metrics = []

    with torch.no_grad():

        for index in range(len(dataset)):

            sample = dataset[index]

            photo = sample["photo"].unsqueeze(0)
            photo = photo.to(device)

            style = torch.tensor(
                [sample["style"]],
                dtype=torch.long,
                device=device,
            )

            real_sketch = sample["sketch"]

            fake_sketch = generator(
                photo,
                style,
            )[0]

            fake_sketch = denormalize(
                fake_sketch
            )

            real_sketch = denormalize(
                real_sketch
            )

            sample_metrics = calculate_metrics(
                fake_sketch,
                real_sketch,
            )

            sample_metrics[
                "image_name"
            ] = sample["image_name"]

            sample_metrics[
                "style"
            ] = sample["style"]

            metrics.append(
                sample_metrics
            )

            # Save generated sketch
            safe_name = (
                sample["image_name"]
                .replace("/", "_")
            )

            save_image(
                fake_sketch,
                generated_dir
                / f"{safe_name}.png",
            )

            if (
                (index + 1) % 100 == 0
                or index == 0
                or index == len(dataset) - 1
            ):
                print(
                    f"Evaluated "
                    f"{index + 1}/{len(dataset)}"
                )

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    numeric_keys = [
        "l1",
        "mse",
        "psnr",
        "ssim",
    ]

    averages = {}

    for key in numeric_keys:

        values = [
            item[key]
            for item in metrics
            if np.isfinite(item[key])
        ]

        averages[key] = float(
            np.mean(values)
        )

    results = {
        "checkpoint": str(
            args.checkpoint
        ),
        "epoch": epoch,
        "num_test_samples": len(dataset),
        "metrics": averages,
        "per_sample": metrics,
    }

    results_path = (
        output_dir
        / "evaluation_results.json"
    )

    with open(
        results_path,
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    for key, value in averages.items():
        print(
            f"{key.upper():6s}: {value:.6f}"
        )

    print()
    print(
        "Results saved to:",
        results_path,
    )

    print(
        "Generated images saved to:",
        generated_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="task4/data/FS2K",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "task4/outputs/"
            "checkpoints/epoch_050.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="task4/outputs/evaluation",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    main(args)
