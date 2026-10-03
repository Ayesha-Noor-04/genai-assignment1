import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, stride=2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpecialistAutoencoder(nn.Module):
    def __init__(
        self,
        channels=(32, 64, 128, 256),
        bottleneck=256,
        bottleneck_type="vector",
        spatial_size=8,
    ):
        super().__init__()

        layers = []
        cin = 3

        for c in channels:
            layers.append(conv_bn_relu(cin, c, stride=2))
            cin = c

        self.encoder = nn.Sequential(*layers)

        self.bottleneck_type = bottleneck_type
        self.spatial_size = spatial_size

        if bottleneck_type == "vector":
            self.to_bottleneck = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1], bottleneck),
                nn.ReLU(inplace=True),
            )

            self.from_bottleneck = nn.Sequential(
                nn.Linear(
                    bottleneck,
                    channels[-1] * 8 * 8,
                ),
                nn.ReLU(inplace=True),
            )

        elif bottleneck_type == "spatial":
            self.to_bottleneck = nn.Sequential(
                nn.Conv2d(
                    channels[-1],
                    channels[-1],
                    3,
                    padding=1,
                ),
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(inplace=True),
            )

            if spatial_size == 8:
                self.spatial_projection = nn.Identity()
            else:
                self.spatial_projection = nn.Upsample(
                    size=(spatial_size, spatial_size),
                    mode="nearest",
                )

            self.from_bottleneck = nn.Sequential(
                nn.Conv2d(
                    channels[-1],
                    channels[-1],
                    3,
                    padding=1,
                ),
                nn.BatchNorm2d(channels[-1]),
                nn.ReLU(inplace=True),
            )

        else:
            raise ValueError(
                "bottleneck_type must be 'vector' or 'spatial'"
            )

        decoder = []
        reversed_channels = list(channels[::-1])

        for i in range(len(reversed_channels) - 1):
            decoder.append(
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=2,
                        mode="nearest",
                    ),
                    nn.Conv2d(
                        reversed_channels[i],
                        reversed_channels[i + 1],
                        3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(
                        reversed_channels[i + 1]
                    ),
                    nn.ReLU(inplace=True),
                )
            )

        decoder.append(
            nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="nearest",
                ),
                nn.Conv2d(
                    reversed_channels[-1],
                    3,
                    3,
                    padding=1,
                ),
                nn.Sigmoid(),
            )
        )

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        h = self.encoder(x)

        if self.bottleneck_type == "vector":
            z = self.to_bottleneck(h)
            h = self.from_bottleneck(z)
            h = h.view(
                x.size(0),
                -1,
                8,
                8,
            )

        else:
            z = self.to_bottleneck(h)
            z = self.spatial_projection(z)
            h = self.from_bottleneck(z)

        return self.decoder(h)