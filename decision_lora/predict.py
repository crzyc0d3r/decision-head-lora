"""Load a locally trained checkpoint and return bounded distributions."""

import argparse
import json
from pathlib import Path

import torch

from .data import QUESTIONS, Vocabulary
from .model import DecisionModel, TinyBackbone, probabilities


def predict(checkpoint: Path, text: str):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    vocabulary = Vocabulary(saved["vocabulary"])
    model = DecisionModel(TinyBackbone(len(vocabulary.tokens)), vocabulary)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    with torch.no_grad():
        results = probabilities(model(vocabulary.batch([text])), saved["temperatures"])
    return {question.name: {
        "choice": question.choices[int(probs[0].argmax())],
        "probabilities": dict(zip(question.choices, probs[0].tolist())),
    } for question, probs in zip(QUESTIONS, results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    parser.add_argument("--text", default="I need help with duplicate charge. Please refund the payment.")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.text), indent=2))


if __name__ == "__main__":
    main()
