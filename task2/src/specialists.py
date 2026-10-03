import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpecialistAutoencoder(nn.Module):
    """U-Net denoiser/deblurrer/inpainter with a residual output.

    Same constructor signature as the old vector/spatial autoencoder so existing
    scripts keep working. `bottleneck`, `bottleneck_type` and `spatial_size` are
    accepted for compatibility but ignored: skip connections make the bottleneck
    size irrelevant. Input/output: (B,3,128,128) in [0,1].
    """

    def __init__(
        self,
        channels=(32, 64, 128, 256),
        bottleneck=256,
        bottleneck_type="vector",
        spatial_size=8,
        residual=True,
    ):
        super().__init__()
        channels = tuple(channels)
        self.residual = residual

        self.encoders = nn.ModuleList()
        cin = 3
        for c in channels:
            self.encoders.append(double_conv(cin, c))
            cin = c

        self.mid = double_conv(channels[-1], channels[-1] * 2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = channels[-1] * 2
        for c in reversed(channels):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.decoders.append(double_conv(c * 2, c))
            prev = c

        self.head = nn.Conv2d(channels[0], 3, 1)
        if residual:
            # start as identity: output == input, so the model can never begin worse than "do nothing"
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        h = x
        for enc in self.encoders:
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)

        h = self.mid(h)

        for up, dec in zip(self.ups, self.decoders):
            h = up(h)
            h = dec(torch.cat([h, skips.pop()], dim=1))

        y = self.head(h)
        if self.residual:
            y = x + y
            # Clamp only at inference. Clamping in training zeroes the gradient on
            # saturated (0/1) salt-and-pepper pixels and the model never learns.
            return y if self.training else y.clamp(0.0, 1.0)
        return torch.sigmoid(y)