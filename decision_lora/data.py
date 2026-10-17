"""Synthetic support text, disjoint partitions, and a training-only vocabulary."""

from dataclasses import dataclass
import random
import re

import torch


@dataclass(frozen=True)
class Question:
    name: str
    prompt: str
    choices: tuple[str, ...]


QUESTIONS = (
    Question("team", "Which team handles this ticket?", ("billing", "technical", "sales")),
    Question("refund", "Did the customer request a refund?", ("yes", "no")),
)


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    labels: tuple[int, ...]


def examples() -> list[Example]:
    topics = (
        ("invoice payment", "duplicate charge", "billing statement", "card transaction"),
        ("application error", "software crash", "server outage", "broken login"),
        ("enterprise pricing", "product quote", "sales demo", "purchase license"),
    )
    requests = (
        ("Please refund the payment.", "I want my money returned.",
         "Requesting a reimbursement.", "Return the money to my card."),
        ("Please explain the next steps.", "I only need information.",
         "Send me instructions.", "Please help me understand this."),
    )
    contexts = ("I need help with", "My question concerns", "Can you investigate", "I am contacting you about")
    rows = []
    for team, team_topics in enumerate(topics):
        for refund, refund_requests in enumerate(requests):
            for topic_id, topic in enumerate(team_topics):
                for request_id, request in enumerate(refund_requests):
                    for context_id, context in enumerate(contexts):
                        key = f"{team}-{refund}-{topic_id}-{request_id}-{context_id}"
                        rows.append(Example(key, f"{context} {topic}. {request}", (team, refund)))
    return rows


def split_data(seed: int = 17) -> dict[str, list[Example]]:
    # Stratify every team/refund pair; every exact ticket belongs to one split.
    grouped = {}
    for row in examples():
        grouped.setdefault(row.labels, []).append(row)
    splits = {"train": [], "calibration": [], "test": []}
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
        n_train, n_cal = int(len(group) * .6), int(len(group) * .2)
        splits["train"].extend(group[:n_train])
        splits["calibration"].extend(group[n_train:n_train + n_cal])
        splits["test"].extend(group[n_train + n_cal:])
    sets = [set(row.id for row in split) for split in splits.values()]
    assert all(not (left & right) for i, left in enumerate(sets) for right in sets[i + 1:])
    return splits


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+|[?.]", text.lower())


class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = {word: i for i, word in enumerate(tokens)}

    @classmethod
    def fit(cls, training: list[Example]) -> "Vocabulary":
        text = [row.text for row in training]
        text.extend(q.prompt + " " + " ".join(q.choices) for q in QUESTIONS)
        return cls(["<pad>", "<unk>", "<bos>", "<eos>"] + sorted({w for t in text for w in words(t)}))

    def encode(self, text: str) -> list[int]:
        return [2] + [self.index.get(w, 1) for w in words(text)] + [3]

    def batch(self, texts: list[str], max_length: int = 96) -> torch.Tensor:
        sequences = [self.encode(text) for text in texts]
        if not sequences or any(len(row) > max_length for row in sequences):
            raise ValueError(f"Provide nonempty text batches of at most {max_length} tokens per input")
        result = torch.zeros(len(sequences), max(map(len, sequences)), dtype=torch.long)
        for i, row in enumerate(sequences):
            result[i, :len(row)] = torch.tensor(row)
        return result


def tensor_batch(rows: list[Example], vocabulary: Vocabulary):
    return vocabulary.batch([row.text for row in rows]), torch.tensor([row.labels for row in rows])
