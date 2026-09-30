"""Task 1 model: Universal Denoising Autoencoder (UDAE).

    x~ [3,128,128] -> conv encoder -> latent vector z (bottleneck_dim) -> conv decoder -> x^ [3,128,128]

* Encoder : 5 stages, each halves the spatial size (128 -> 64 -> 32 -> 16 -> 8 -> 4) while the
            channel count grows (c, 2c, 4c, 8c, 8c). Each stage = strided 3x3 conv + 3x3 conv
            (BatchNorm + ReLU after each).
* Bottleneck: flatten (8c x 4 x 4) -> Linear -> z (bottleneck_dim) -> Dropout -> Linear -> reshape.
            With bottleneck_dim = 256 the image (49,152 numbers) is squeezed through 256 numbers,
            so the network cannot simply copy its input.
* Decoder : 5 stages of (nearest-neighbour upsample x2 + 3x3 conv + 3x3 conv, BN + ReLU), mirroring
            the encoder, then a 3x3 conv to 3 channels and a sigmoid (output in [0,1]).
* NO skip connections of any kind between encoder and decoder.

Upsample+conv (instead of transposed conv) is used in the decoder to avoid checkerboard artefacts.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UDAE(nn.Module):
    def __init__(self, base_channels: int = 64, bottleneck_dim: int = 256, dropout: float = 0.1,
                 in_size: int = 128, n_stages: int = 5):
        super().__init__()
        assert in_size % (2 ** n_stages) == 0
        mult = [1, 2, 4, 8, 8][:n_stages]
        chans = [base_channels * m for m in mult]
        self.chans = chans
        self.bottleneck_dim = bottleneck_dim
        self.final_size = in_size // (2 ** n_stages)  # 4
        self.final_ch = chans[-1]
        flat = self.final_ch * self.final_size ** 2

        # ---- encoder ----
        enc, cin = [], 3
        for c in chans:
            enc += [conv_bn_relu(cin, c, stride=2), conv_bn_relu(c, c, stride=1)]
            cin = c
        self.encoder = nn.Sequential(*enc)

        # ---- bottleneck ----
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(flat, bottleneck_dim))
        self.latent_dropout = nn.Dropout(dropout)
        self.from_latent = nn.Sequential(nn.Linear(bottleneck_dim, flat), nn.ReLU(inplace=True))

        # ---- decoder ----
        dec, rev = [], list(reversed(chans))
        cin = rev[0]
        for i, c in enumerate(rev):
            cout = rev[i + 1] if i + 1 < len(rev) else chans[0]
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    conv_bn_relu(cin, cout), conv_bn_relu(cout, cout)]
            cin = cout
        self.decoder = nn.Sequential(*dec)
        self.out_conv = nn.Conv2d(chans[0], 3, 3, padding=1)

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        h = self.from_latent(self.latent_dropout(z))
        h = h.view(-1, self.final_ch, self.final_size, self.final_size)
        return torch.sigmoid(self.out_conv(self.decoder(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)