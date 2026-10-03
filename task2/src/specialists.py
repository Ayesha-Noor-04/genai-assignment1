from __future__ import annotations

import torch.nn as nn

from src.models import conv_bn_relu


class SpecialistAutoencoder(nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128, 256),
        bottleneck=256,
    ):
        super().__init__()

        layers = []
        cin = 3

        for c in channels:
            layers.append(conv_bn_relu(cin, c, stride=2))
            cin = c

        self.encoder = nn.Sequential(*layers)

        self.to_bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], bottleneck),
            nn.ReLU(inplace=True),
        )

        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck, channels[-1] * 8 * 8),
            nn.ReLU(inplace=True),
        )

        decoder = []
        reversed_channels = list(channels[::-1])

        for i in range(len(reversed_channels) - 1):
            decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(
                        reversed_channels[i],
                        reversed_channels[i + 1],
                        3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(reversed_channels[i + 1]),
                    nn.ReLU(inplace=True),
                )
            )

        decoder.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(reversed_channels[-1], 3, 3, padding=1),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        h = self.encoder(x)
        z = self.to_bottleneck(h)
        h = self.from_bottleneck(z)
        h = h.view(x.size(0), -1, 8, 8)
        return self.decoder(h)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)