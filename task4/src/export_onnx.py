import argparse
from pathlib import Path

import torch

from .models import ConditionalGenerator


def load_generator(checkpoint_path, device):
    """Load the trained generator from a training checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    generator = ConditionalGenerator()
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    generator.eval()

    epoch = checkpoint.get("epoch", "unknown")

    print(f"Loaded generator from epoch {epoch}")

    return generator


def export_onnx(
    checkpoint_path,
    output_path,
    image_size=256,
    opset_version=17,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    generator = load_generator(
        checkpoint_path,
        device,
    )

    # Dummy inputs matching the generator interface.
    dummy_photo = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    dummy_style = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Exporting generator to ONNX...")

    torch.onnx.export(
        generator,
        (dummy_photo, dummy_style),
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[
            "photo",
            "style",
        ],
        output_names=[
            "sketch",
        ],
        dynamic_axes={
            "photo": {
                0: "batch_size",
            },
            "style": {
                0: "batch_size",
            },
            "sketch": {
                0: "batch_size",
            },
        },
    )

    print("ONNX export successful")
    print(f"Saved to: {output_path}")
    print(
        f"Size: "
        f"{output_path.stat().st_size / 1024 / 1024:.2f} MB"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export FS2K generator to ONNX"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
    )

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        image_size=args.image_size,
        opset_version=args.opset_version,
    )


if __name__ == "__main__":
    main()