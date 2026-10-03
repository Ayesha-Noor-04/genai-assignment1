import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMoE(nn.Module):
    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()

        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.temperature = temperature

    def forward(self, x):
        logits = self.gate(x)

        weights = F.softmax(
            logits / self.temperature,
            dim=1,
        )

        outputs = [x]

        for expert in self.experts:
            outputs.append(expert(x))

        y = torch.zeros_like(x)

        for k in range(4):
            y = y + weights[:, k, None, None, None] * outputs[k]

        return y, weights, logits


def balance_loss(weights):
    mean_weights = weights.mean(dim=0)
    target = torch.full_like(mean_weights, 0.25)

    return ((mean_weights - target) ** 2).sum()


def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )