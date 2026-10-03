
import argparse
import os

import torch
import yaml

from task2.src.classifier import CorruptionClassifier
from task2.src.specialists import SpecialistAutoencoder


SPECIALIST_PARAMS = {
    "salt_pepper": {
        "channels": (64, 128, 256, 512),
        "bottleneck": 256,
    },
    "blur": {
        "channels": (64, 128, 256, 512),
        "bottleneck": 512,
    },
    "occlusion": {
        "channels": (64, 128, 256, 512),
        "bottleneck": 128,
    },
}


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Exporting classifier...")

    classifier = CorruptionClassifier(
        channels=config["classifier"]["channels"],
        dropout=config["classifier"]["dropout"],
    )

    load_checkpoint(
        classifier,
        config["classifier"]["checkpoint"],
    )

    dummy_input = torch.randn(1, 3, 128, 128)

    classifier_path = os.path.join(
        results_dir,
        "classifier.onnx",
    )

    torch.onnx.export(
        classifier,
        dummy_input,
        classifier_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )

    print("saved:", classifier_path)

    for name, params in SPECIALIST_PARAMS.items():
        print("Exporting", name, "specialist...")

        model = SpecialistAutoencoder(
            channels=params["channels"],
            bottleneck=params["bottleneck"],
        )

        load_checkpoint(
            model,
            config["specialist"]["checkpoints"][name],
        )

        output_path = os.path.join(
            results_dir,
            name + ".onnx",
        )

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
        )

        print("saved:", output_path)

    print("ONNX export complete.")


if __name__ == "__main__":
    main()
