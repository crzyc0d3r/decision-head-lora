"""Per-question held-out temperature selection and reliability measurements."""

import torch
from torch.nn import functional as F


def fit_temperatures(logits, labels):
    # Includes T=1 so the selected held-out NLL cannot exceed the original NLL.
    candidates = torch.cat((torch.logspace(-.6, .6, 81), torch.ones(1)))
    temperatures = []
    for i, scores in enumerate(logits):
        scores = scores.detach()
        losses = torch.stack([F.cross_entropy(scores / t, labels[:, i]) for t in candidates])
        temperatures.append(float(candidates[losses.argmin()]))
    return temperatures


def metrics(logits, labels, temperatures):
    report = []
    for i, (scores, temperature) in enumerate(zip(logits, temperatures)):
        probs = (scores / temperature).softmax(-1)
        confidence, choices = probs.max(-1)
        correct = choices.eq(labels[:, i])
        ece = torch.zeros(())
        for low in torch.arange(0, 1, .1):
            mask = (confidence > low) & (confidence <= low + .1)
            if mask.any():
                ece += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
        report.append({"accuracy": float(correct.float().mean()),
                       "nll": float(F.cross_entropy(scores / temperature, labels[:, i])),
                       "ece_10_bins": float(ece)})
    return report
