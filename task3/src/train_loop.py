import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src import corruptions as C
from src.datasets import to_tensor


class MoEDataset(Dataset):
    def __init__(self, images, entries):
        self.images = images
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        entry = self.entries[i]

        clean = self.images[entry["image_idx"]]

        corrupted = C.apply_corruption(
            clean,
            entry["corruption"],
            entry["params"],
            entry["seed"],
        )

        x = to_tensor(corrupted)
        y = to_tensor(clean)

        label = entry["label"]

        return x, y, label


def set_experts_eval(model):
    for expert in model.experts:
        expert.eval()


def freeze_experts(model):
    for expert in model.experts:
        for parameter in expert.parameters():
            parameter.requires_grad = False


def unfreeze_experts(model):
    for expert in model.experts:
        for parameter in expert.parameters():
            parameter.requires_grad = True


def ssim_loss(x, y):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)

    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (
        (2 * mu_x * mu_y + c1)
        * (2 * sigma_xy + c2)
    ) / (
        (mu_x * mu_x + mu_y * mu_y + c1)
        * (sigma_x + sigma_y + c2)
    )

    return 1 - ssim.mean()


def compute_loss(
    output,
    target,
    weights,
    logits,
    alpha,
    lam_c,
    lam_b,
    temperature,
):
    l1 = F.l1_loss(output, target)

    ssim = ssim_loss(output, target)

    reconstruction = (
        alpha * l1
        + (1 - alpha) * ssim
    )

    classification = F.cross_entropy(
        logits / temperature,
        target.new_tensor(
            torch.argmax(target, dim=1)
        ).long(),
    )

    balance = (
        weights.mean(dim=0) - 0.25
    ).pow(2).sum()

    loss = (
        reconstruction
        + lam_c * classification
        + lam_b * balance
    )

    return loss, l1, ssim, classification, balance


def train_epoch(
    model,
    loader,
    optimizer,
    alpha,
    lam_c,
    lam_b,
    temperature,
    device,
):
    model.train()
    set_experts_eval(model)

    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    total_classification = 0
    total_balance = 0
    count = 0

    for x, target, labels in loader:
        x = x.to(device)
        target = target.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output, weights, logits = model(x)

        l1 = F.l1_loss(output, target)
        ssim = ssim_loss(output, target)

        reconstruction = (
            alpha * l1
            + (1 - alpha) * ssim
        )

        classification = F.cross_entropy(
            logits / temperature,
            labels,
        )

        balance = (
            weights.mean(dim=0) - 0.25
        ).pow(2).sum()

        loss = (
            reconstruction
            + lam_c * classification
            + lam_b * balance
        )

        loss.backward()
        optimizer.step()

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_l1 += l1.item() * batch_size
        total_ssim += ssim.item() * batch_size
        total_classification += classification.item() * batch_size
        total_balance += balance.item() * batch_size
        count += batch_size

    return {
        "loss": total_loss / count,
        "l1": total_l1 / count,
        "ssim_loss": total_ssim / count,
        "classification": total_classification / count,
        "balance": total_balance / count,
    }


def evaluate(
    model,
    loader,
    alpha,
    lam_c,
    lam_b,
    temperature,
    device,
):
    model.eval()

    total_loss = 0
    total_l1 = 0
    total_ssim = 0
    total_classification = 0
    total_balance = 0
    count = 0

    with torch.no_grad():
        for x, target, labels in loader:
            x = x.to(device)
            target = target.to(device)
            labels = labels.to(device)

            output, weights, logits = model(x)

            l1 = F.l1_loss(output, target)
            ssim = ssim_loss(output, target)

            reconstruction = (
                alpha * l1
                + (1 - alpha) * ssim
            )

            classification = F.cross_entropy(
                logits / temperature,
                labels,
            )

            balance = (
                weights.mean(dim=0) - 0.25
            ).pow(2).sum()

            loss = (
                reconstruction
                + lam_c * classification
                + lam_b * balance
            )

            batch_size = x.size(0)

            total_loss += loss.item() * batch_size
            total_l1 += l1.item() * batch_size
            total_ssim += ssim.item() * batch_size
            total_classification += classification.item() * batch_size
            total_balance += balance.item() * batch_size
            count += batch_size

    return {
        "loss": total_loss / count,
        "l1": total_l1 / count,
        "ssim_loss": total_ssim / count,
        "classification": total_classification / count,
        "balance": total_balance / count,
    }


def build_loaders(cfg):
    processed_dir = cfg["data"]["processed_dir"]
    manifests_dir = cfg["data"]["manifests_dir"]

    trainval = np.load(
        processed_dir + "/trainval_128.npy"
    )

    with open(manifests_dir + "/val_manifest.json", "r") as f:
        import json
        val_manifest = json.load(f)

    with open(manifests_dir + "/test_manifest.json", "r") as f:
        import json
        test_manifest = json.load(f)

    val_entries = val_manifest["entries"]
    test_entries = test_manifest["entries"]

    train_dataset = MoEDataset(
        trainval,
        test_entries,
    )

    val_dataset = MoEDataset(
        trainval,
        val_entries,
    )

    batch_size = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def save_checkpoint(model, optimizer, epoch, score, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score": score,
        },
        path,
    )