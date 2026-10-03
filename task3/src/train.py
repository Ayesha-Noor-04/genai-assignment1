import sys

import torch
import yaml

sys.path.insert(0, "/content/genai-assignment1")

from task2.src.classifier import CorruptionClassifier
from task2.src.specialists import SpecialistAutoencoder

from task3.src.soft_moe import SoftMoE


def load_task2_models(config_path):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    classifier_cfg = cfg["task2"]["classifier"]
    specialist_cfg = cfg["task2"]["specialists"]

    gate = CorruptionClassifier(
        channels=classifier_cfg["channels"],
        dropout=classifier_cfg["dropout"],
        num_classes=4,
    )

    gate.load_state_dict(
        torch.load(
            classifier_cfg["checkpoint"],
            map_location=device,
        )
    )

    experts = []

    names = [
        "salt_pepper",
        "blur",
        "occlusion",
    ]

    for name in names:
        expert = SpecialistAutoencoder(
            channels=tuple(specialist_cfg["channels"]),
            bottleneck=specialist_cfg["bottleneck"],
        )

        expert.load_state_dict(
            torch.load(
                specialist_cfg["checkpoints"][name],
                map_location=device,
            )
        )

        experts.append(expert)

    gate.to(device)

    for expert in experts:
        expert.to(device)

    model = SoftMoE(
        gate=gate,
        experts=experts,
        temperature=cfg["model"]["temperature"],
    )

    model.to(device)

    return model, cfg, device