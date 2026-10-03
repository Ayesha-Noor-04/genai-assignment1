from __future__ import annotations

import torch
import torch.nn as nn


CHANNEL_CONFIGS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "wide": (64, 128, 256, 512),
}


def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CorruptionClassifier(nn.Module):
    def __init__(self, channels="medium", dropout=0.2, num_classes=4):
        super().__init__()

        if channels not in CHANNEL_CONFIGS:
            raise ValueError(f"unknown channel configuration: {channels}")

        chans = CHANNEL_CONFIGS[channels]

        layers = []
        cin = 3

        for c in chans:
            layers.append(conv_bn_relu(cin, c, stride=2))
            layers.append(conv_bn_relu(c, c, stride=1))
            cin = c

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(chans[-1], num_classes)

    def forward(self, x):
        h = self.encoder(x)
        h = self.pool(h)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.classifier(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)