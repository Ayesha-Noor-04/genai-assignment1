"""Task 1 model: Universal Denoising Autoencoder (UDAE).

    x~ [3,128,128] -> conv encoder -> compressed latent z -> conv decoder -> x^ [3,128,128]

Two bottleneck variants (selected with `latent`), both with the SAME number of latent values
(`bottleneck_dim`) so they are directly comparable:

  latent="vector"  (first design)
      encoder: 5 stages, 128 -> 64 -> 32 -> 16 -> 8 -> 4, channels (c, 2c, 4c, 8c, 8c)
      bottleneck: flatten(8c x 4 x 4) -> Linear -> z in R^bottleneck_dim -> Dropout -> Linear -> reshape
      Spatial layout is destroyed by the Linear layers; the decoder must regenerate everything
      from one vector.

  latent="spatial" (convolutional bottleneck), latent_size = 8 (default) or 16
      encoder: 4 stages (8x8 grid): 128 -> 64 -> 32 -> 16 -> 8, channels (c, 2c, 4c, 8c)
               3 stages (16x16 grid): 128 -> 64 -> 32 -> 16,    channels (c, 2c, 4c)
      bottleneck: 1x1 conv -> z in R^{(bottleneck_dim/size^2) x size x size} -> Dropout -> 1x1 conv
      Same compression (e.g. 1024 numbers for 49,152 pixel values) but the latent keeps a coarse
      8x8 spatial grid, so positions of structures survive the bottleneck.

Decoder (both): stages of (nearest-neighbour upsample x2 + 3x3 conv + 3x3 conv, BN + ReLU) mirroring
the encoder, then a 3x3 conv to 3 channels and a sigmoid (output in [0,1]).

There are NO skip connections between encoder and decoder: the decoder only ever sees z.
Upsample+conv (not transposed conv) is used in the decoder to avoid checkerboard artefacts.
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
                 in_size: int = 128, latent: str = "vector", latent_size: int = 8):
        super().__init__()
        assert latent in ("vector", "spatial")
        self.latent_type = latent
        if latent == "vector":
            n_stages, mult = 5, [1, 2, 4, 8, 8]
        else:  # spatial latent grid of latent_size x latent_size: 8 -> 4 stages, 16 -> 3 stages
            assert latent_size in (8, 16)
            n_stages = {8: 4, 16: 3}[latent_size]
            mult = [1, 2, 4, 8][:n_stages]
        assert in_size % (2 ** n_stages) == 0
        chans = [base_channels * m for m in mult]
        self.chans = chans
        self.bottleneck_dim = bottleneck_dim
        self.final_size = in_size // (2 ** n_stages)  # 4 (vector) or 8 (spatial)
        self.final_ch = chans[-1]

        # ---- encoder ----
        enc, cin = [], 3
        for c in chans:
            enc += [conv_bn_relu(cin, c, stride=2), conv_bn_relu(c, c, stride=1)]
            cin = c
        self.encoder = nn.Sequential(*enc)

        # ---- bottleneck ----
        self.latent_dropout = nn.Dropout(dropout)
        if latent == "vector":
            flat = self.final_ch * self.final_size ** 2
            self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(flat, bottleneck_dim))
            self.from_latent = nn.Sequential(nn.Linear(bottleneck_dim, flat), nn.ReLU(inplace=True))
        else:
            cells = self.final_size ** 2
            assert bottleneck_dim % cells == 0, f"bottleneck_dim must be a multiple of {cells}"
            self.latent_ch = bottleneck_dim // cells
            self.to_latent = nn.Conv2d(self.final_ch, self.latent_ch, 1)
            self.from_latent = nn.Sequential(nn.Conv2d(self.latent_ch, self.final_ch, 1), nn.ReLU(inplace=True))

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
        if self.latent_type == "vector":
            h = h.view(-1, self.final_ch, self.final_size, self.final_size)
        return torch.sigmoid(self.out_conv(self.decoder(h)))

    def forward(self, x):
        return self.decode(self.encode(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)