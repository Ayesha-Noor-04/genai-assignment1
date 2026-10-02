import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .dataset import FS2KDataset
from .models import (
    ConditionalGenerator,
    ConditionalPatchGANDiscriminator,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denormalize(x):
    return (x + 1.0) / 2.0


def save_samples(generator, batch, output_dir, epoch, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()

    with torch.no_grad():
        photo = batch["photo"].to(device)
        style = batch["style"].to(device)

        fake = generator(photo, style)

        image = denormalize(fake[0]).clamp(0, 1)

        save_image(
            image,
            output_dir / f"epoch_{epoch:03d}.png",
        )

    generator.train()


def train_one_epoch(
    generator,
    discriminator,
    loader,
    optimizer_g,
    optimizer_d,
    adversarial_loss,
    reconstruction_loss,
    device,
    lambda_l1,
    scaler_g,
    scaler_d,
):
    generator.train()
    discriminator.train()

    total_g = 0.0
    total_d = 0.0

    use_amp = device.type == "cuda"

    for batch in loader:
        photo = batch["photo"].to(
            device,
            non_blocking=True,
        )

        real_sketch = batch["sketch"].to(
            device,
            non_blocking=True,
        )

        style = batch["style"].to(
            device,
            non_blocking=True,
        )

        # --------------------------------------------------
        # Train discriminator
        # --------------------------------------------------
        optimizer_d.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):
            fake_sketch = generator(
                photo,
                style,
            )

            real_pred = discriminator(
                photo,
                real_sketch,
                style,
            )

            fake_pred = discriminator(
                photo,
                fake_sketch.detach(),
                style,
            )

            real_target = torch.ones_like(real_pred)
            fake_target = torch.zeros_like(fake_pred)

            loss_d_real = adversarial_loss(
                real_pred,
                real_target,
            )

            loss_d_fake = adversarial_loss(
                fake_pred,
                fake_target,
            )

            loss_d = 0.5 * (
                loss_d_real + loss_d_fake
            )

        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()

        # --------------------------------------------------
        # Train generator
        # --------------------------------------------------
        optimizer_g.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp,
        ):
            fake_sketch = generator(
                photo,
                style,
            )

            fake_pred = discriminator(
                photo,
                fake_sketch,
                style,
            )

            target = torch.ones_like(fake_pred)

            loss_g_adv = adversarial_loss(
                fake_pred,
                target,
            )

            loss_g_l1 = reconstruction_loss(
                fake_sketch,
                real_sketch,
            )

            loss_g = (
                loss_g_adv
                + lambda_l1 * loss_g_l1
            )

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        total_g += loss_g.item()
        total_d += loss_d.item()

    return (
        total_g / len(loader),
        total_d / len(loader),
    )


def main(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    dataset = FS2KDataset(
        root=args.data_root,
        split="train",
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    generator = ConditionalGenerator().to(device)

    discriminator = (
        ConditionalPatchGANDiscriminator()
        .to(device)
    )

    adversarial_loss = nn.BCEWithLogitsLoss()

    reconstruction_loss = nn.L1Loss()

    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
    )

    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
    )

    use_amp = device.type == "cuda"

    scaler_g = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    scaler_d = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    output_dir = Path(args.output_dir)

    checkpoint_dir = (
        output_dir / "checkpoints"
    )

    sample_dir = (
        output_dir / "samples"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_batch = next(iter(loader))

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        loss_g, loss_d = train_one_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=loader,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            adversarial_loss=adversarial_loss,
            reconstruction_loss=reconstruction_loss,
            device=device,
            lambda_l1=args.lambda_l1,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"G: {loss_g:.4f} | "
            f"D: {loss_d:.4f}"
        )

        save_samples(
            generator,
            fixed_batch,
            sample_dir,
            epoch,
            device,
        )

        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "args": vars(args),
        }

        torch.save(
            checkpoint,
            checkpoint_dir
            / f"epoch_{epoch:03d}.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="task4/data/FS2K",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="task4/outputs",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr-g",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lr-d",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)