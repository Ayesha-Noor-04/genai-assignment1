import torch
import torch.nn as nn


class StyleEmbedding(nn.Module):
    """Learnable embedding for the three FS2K sketch styles."""

    def __init__(self, num_styles=3, embedding_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_styles,
            embedding_dim=embedding_dim,
        )

    def forward(self, style):
        return self.embedding(style)


class ConditionalGenerator(nn.Module):
    """
    Conditional U-Net-style generator.

    Input:
        photo: [B, 3, 256, 256]
        style: [B]

    Output:
        generated sketch: [B, 3, 256, 256]
    """

    def __init__(
        self,
        num_styles=3,
        style_dim=64,
        base_channels=64,
    ):
        super().__init__()

        self.style_embedding = StyleEmbedding(
            num_styles=num_styles,
            embedding_dim=style_dim,
        )

        # Encode the style into a spatial feature map.
        self.style_projection = nn.Sequential(
            nn.Linear(style_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
        )

        # Photo encoder.
        self.enc1 = self._down_block(3, base_channels, normalize=False)
        self.enc2 = self._down_block(
            base_channels,
            base_channels * 2,
        )
        self.enc3 = self._down_block(
            base_channels * 2,
            base_channels * 4,
        )
        self.enc4 = self._down_block(
            base_channels * 4,
            base_channels * 8,
        )
        self.enc5 = self._down_block(
            base_channels * 8,
            base_channels * 8,
        )

        # Bottleneck receives image features + style features.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                base_channels * 8 + 256,
                base_channels * 8,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        # Decoder.
        self.dec5 = self._up_block(
            base_channels * 8,
            base_channels * 8,
        )

        self.dec4 = self._up_block(
            base_channels * 16,
            base_channels * 4,
        )

        self.dec3 = self._up_block(
            base_channels * 8,
            base_channels * 2,
        )

        self.dec2 = self._up_block(
            base_channels * 4,
            base_channels,
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    @staticmethod
    def _down_block(
        in_channels,
        out_channels,
        normalize=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    @staticmethod
    def _up_block(
        in_channels,
        out_channels,
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, photo, style):
        e1 = self.enc1(photo)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        style_vector = self.style_embedding(style)

        style_features = self.style_projection(style_vector)
        style_features = style_features.view(
            style_features.size(0),
            256,
            8,
            8,
        )

        bottleneck_input = torch.cat(
            [e5, style_features],
            dim=1,
        )

        b = self.bottleneck(bottleneck_input)

        d5 = self.dec5(b)
        d5 = torch.cat([d5, e4], dim=1)

        d4 = self.dec4(d5)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        return self.dec1(d2)


class ConditionalPatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator.

    Receives:
        photo
        real/fake sketch
        style ID

    and predicts a patch-wise realism score.
    """

    def __init__(
        self,
        num_styles=3,
        style_dim=64,
        base_channels=64,
    ):
        super().__init__()

        self.style_embedding = StyleEmbedding(
            num_styles=num_styles,
            embedding_dim=style_dim,
        )

        self.style_projection = nn.Sequential(
            nn.Linear(style_dim, 256 * 256),
            nn.ReLU(inplace=True),
        )

        self.model = nn.Sequential(
            self._block(
                3 + 3 + 1,
                base_channels,
                normalize=False,
            ),
            self._block(
                base_channels,
                base_channels * 2,
            ),
            self._block(
                base_channels * 2,
                base_channels * 4,
            ),
            self._block(
                base_channels * 4,
                base_channels * 8,
            ),
            nn.Conv2d(
                base_channels * 8,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        )

    @staticmethod
    def _block(
        in_channels,
        out_channels,
        normalize=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize,
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, photo, sketch, style):
        style_vector = self.style_embedding(style)

        style_map = self.style_projection(style_vector)
        style_map = style_map.view(
            style_map.size(0),
            1,
            256,
            256,
        )

        x = torch.cat(
            [photo, sketch, style_map],
            dim=1,
        )

        return self.model(x)