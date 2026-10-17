"""CPU pretraining, adapter training, calibration, and untouched test evaluation."""

import argparse
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .calibration import fit_temperatures, metrics
from .data import QUESTIONS, Vocabulary, split_data, tensor_batch
from .model import DecisionModel, TinyBackbone, TinyLanguageModel, multi_question_loss


def train(output=Path("artifacts"), steps=60, pretrain_steps=60, seed=17):
    if steps < 1 or pretrain_steps < 1:
        raise ValueError("Both training phases require at least one step")
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = random.Random(seed)
    splits = split_data(seed)
    vocabulary = Vocabulary.fit(splits["train"])
    backbone = TinyBackbone(len(vocabulary.tokens))
    language_model = TinyLanguageModel(backbone, len(vocabulary.tokens))
    optimizer = torch.optim.AdamW(language_model.parameters(), lr=.003)
    lm_loss = None
    for _ in range(pretrain_steps):
        tokens, _ = tensor_batch(rng.sample(splits["train"], 24), vocabulary)
        scores = language_model(tokens[:, :-1])
        lm_loss = F.cross_entropy(scores.reshape(-1, scores.shape[-1]), tokens[:, 1:].reshape(-1), ignore_index=0)
        optimizer.zero_grad()
        lm_loss.backward()
        optimizer.step()

    # Discard the generation head completely; it is absent from the saved model.
    model = DecisionModel(backbone, vocabulary)
    del language_model, optimizer
    frozen = {name: param.detach().clone() for name, param in model.named_parameters() if not param.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.004)
    test_tokens, test_labels = tensor_batch(splits["test"], vocabulary)
    model.eval()
    with torch.no_grad():
        before = metrics(model(test_tokens), test_labels, [1.] * len(QUESTIONS))
    model.train()
    losses = []
    for _ in range(steps):
        tokens, labels = tensor_batch(rng.sample(splits["train"], 24), vocabulary)
        loss = multi_question_loss(model(tokens), labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    frozen_unchanged = all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    if not frozen_unchanged:
        raise RuntimeError("A frozen backbone weight changed during adapter training")
    model.eval()
    calibration_tokens, calibration_labels = tensor_batch(splits["calibration"], vocabulary)
    with torch.no_grad():
        calibration_logits = model(calibration_tokens)
        test_logits = model(test_tokens)
        temperatures = fit_temperatures(calibration_logits, calibration_labels)
        test_raw = metrics(test_logits, test_labels, [1.] * len(QUESTIONS))
        test_calibrated = metrics(test_logits, test_labels, temperatures)
    report = {
        "seed": seed, "pretrain_steps": pretrain_steps, "decision_steps": steps,
        "pretrain_final_loss": float(lm_loss.detach()),
        "decision_first_loss": losses[0], "decision_last_loss": losses[-1],
        "frozen_backbone_unchanged": frozen_unchanged,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "split_counts": {key: len(rows) for key, rows in splits.items()},
        "temperatures": dict(zip([q.name for q in QUESTIONS], temperatures)),
        "test_before_training": dict(zip([q.name for q in QUESTIONS], before)),
        "test_uncalibrated": dict(zip([q.name for q in QUESTIONS], test_raw)),
        "test_calibrated": dict(zip([q.name for q in QUESTIONS], test_calibrated)),
        "calibration_uncalibrated": metrics(calibration_logits, calibration_labels, [1.] * len(QUESTIONS)),
        "calibration_calibrated": metrics(calibration_logits, calibration_labels, temperatures),
        "split_ids": {key: [row.id for row in rows] for key, rows in splits.items()},
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "vocabulary": vocabulary.tokens,
                "temperatures": temperatures, "seed": seed}, output / "model.pt")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--pretrain-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = train(args.output, args.steps, args.pretrain_steps, args.seed)
    # The full split manifest is saved locally; terminal output stays readable.
    print(json.dumps({k: v for k, v in report.items() if k != "split_ids"}, indent=2))


if __name__ == "__main__":
    main()
