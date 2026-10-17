"""Causal pretraining, frozen attention with LoRA, and option cross-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import QUESTIONS, Vocabulary


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.a), self.b) * self.scale


class CausalBlock(nn.Module):
    def __init__(self, width=48, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, inputs):
        batch, length, width = inputs.shape
        normalized = self.norm1(inputs)
        def project(layer):
            return layer(normalized).reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
        # Right padding cannot affect valid tokens in a causal backbone.
        evidence = F.scaled_dot_product_attention(project(self.q), project(self.k), project(self.v), is_causal=True)
        hidden = inputs + self.out(evidence.transpose(1, 2).reshape(batch, length, width))
        return hidden + self.ff(self.norm2(hidden))


class TinyBackbone(nn.Module):
    def __init__(self, vocab_size: int, width: int = 48, max_length: int = 96):
        super().__init__()
        self.width = width
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(max_length, width)
        self.blocks = nn.ModuleList([CausalBlock(width), CausalBlock(width)])
        self.norm = nn.LayerNorm(width)

    def forward(self, token_ids):
        length = token_ids.shape[1]
        if length > self.max_length:
            raise ValueError("Input exceeds the backbone context length")
        hidden = self.embedding(token_ids) + self.position(torch.arange(length, device=token_ids.device))
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)

    def freeze_and_add_lora(self):
        self.requires_grad_(False)
        for block in self.blocks:
            block.q = LoRALinear(block.q)
            block.v = LoRALinear(block.v)


class TinyLanguageModel(nn.Module):
    def __init__(self, backbone: TinyBackbone, vocab_size: int):
        super().__init__()
        self.backbone = backbone
        self.lm_head = nn.Linear(backbone.width, vocab_size)

    def forward(self, token_ids):
        return self.lm_head(self.backbone(token_ids))


class DecisionHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.question = nn.Linear(width, width)
        self.option = nn.Linear(width, width)
        self.choice_interaction = nn.MultiheadAttention(width, 4, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, state, padding_mask, question_vectors, option_vectors):
        all_scores = []
        for question, options in zip(question_vectors, option_vectors):
            queries = (self.question(question) + self.option(options)).unsqueeze(0).expand(state.shape[0], -1, -1)
            # Options interact within a question; separate questions retain their own distributions.
            interactions, _ = self.choice_interaction(queries, queries, queries, need_weights=False)
            queries = queries + interactions
            evidence, _ = self.cross_attention(queries, state, state, key_padding_mask=padding_mask, need_weights=False)
            all_scores.append(self.score(self.norm(queries + evidence)).squeeze(-1))
        return all_scores


class DecisionModel(nn.Module):
    def __init__(self, backbone: TinyBackbone, vocabulary: Vocabulary):
        super().__init__()
        backbone.freeze_and_add_lora()
        self.backbone = backbone
        self.head = DecisionHead(backbone.width)
        # These text embeddings are frozen buffers; only the projection/head and LoRA learn.
        for i, question in enumerate(QUESTIONS):
            with torch.no_grad():
                def embed(text):
                    ids = torch.tensor(vocabulary.encode(text)[1:-1])
                    return backbone.embedding(ids).mean(dim=0)
                self.register_buffer(f"question_{i}", embed(question.prompt))
                self.register_buffer(f"options_{i}", torch.stack([embed(choice) for choice in question.choices]))

    def forward(self, token_ids):
        # Keep autograd enabled: frozen weights still propagate gradients into LoRA.
        state = self.backbone(token_ids)
        question_vectors = [getattr(self, f"question_{i}") for i in range(len(QUESTIONS))]
        option_vectors = [getattr(self, f"options_{i}") for i in range(len(QUESTIONS))]
        return self.head(state, token_ids.eq(0), question_vectors, option_vectors)


def multi_question_loss(logits, labels):
    return torch.stack([F.cross_entropy(scores, labels[:, i]) for i, scores in enumerate(logits)]).mean()


def probabilities(logits, temperatures):
    if len(logits) != len(temperatures) or any(t <= 0 for t in temperatures):
        raise ValueError("Each question requires one positive temperature")
    return [(scores / temperature).softmax(dim=-1) for scores, temperature in zip(logits, temperatures)]
