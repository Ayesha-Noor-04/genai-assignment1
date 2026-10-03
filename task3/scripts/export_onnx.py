import os
import sys

import torch
import onnx

sys.path.insert(0, "/content/genai-assignment1")

from task3.src.train import load_task2_models


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

    model_path = (
        cfg["output"]["results_dir"]
        + "/task3_soft_moe.onnx"
    )

    os.makedirs(
        cfg["output"]["results_dir"],
        exist_ok=True,
    )

    x = torch.randn(
        1,
        3,
        128,
        128,
        device=device,
    )

    torch.onnx.export(
        model,
        x,
        model_path,
        input_names=["input"],
        output_names=[
            "restored",
            "weights",
            "logits",
        ],
        dynamic_axes={
            "input": {
                0: "batch"
            },
            "restored": {
                0: "batch"
            },
            "weights": {
                0: "batch"
            },
            "logits": {
                0: "batch"
            },
        },
        opset_version=17,
    )

    print("ONNX exported:", model_path)

    model_onnx = onnx.load(model_path)

    onnx.checker.check_model(
        model_onnx
    )

    print("ONNX check: passed")

    print(
        "checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "checkpoint score:",
        checkpoint["score"],
    )

    print(
        "file size:",
        round(
            os.path.getsize(model_path)
            / (1024 ** 2),
            2,
        ),
        "MB",
    )


if __name__ == "__main__":
    main()