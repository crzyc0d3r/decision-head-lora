"""Regression checks for the learning boundaries and decision probabilities."""

import unittest

import torch
from torch.nn import functional as F

from decision_lora.calibration import fit_temperatures
from decision_lora.data import QUESTIONS, Example, Vocabulary, split_data, tensor_batch
from decision_lora.model import DecisionModel, TinyBackbone, multi_question_loss, probabilities


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.splits = split_data()
        self.vocabulary = Vocabulary.fit(self.splits["train"])
        self.model = DecisionModel(TinyBackbone(len(self.vocabulary.tokens)), self.vocabulary)

    def test_splits_and_vocabulary_have_no_held_out_fit(self):
        sets = [{row.id for row in rows} for rows in self.splits.values()]
        texts = [{row.text for row in rows} for rows in self.splits.values()]
        for groups in (sets, texts):
            for i, left in enumerate(groups):
                for right in groups[i + 1:]:
                    self.assertFalse(left & right)
        vocabulary = Vocabulary.fit([Example("only", "seen", (0, 0))])
        self.assertEqual(vocabulary.encode("unseen")[1], 1)
        for rows in self.splits.values():
            self.assertEqual(len({row.labels for row in rows}), 6)

    def test_only_lora_and_head_train_and_frozen_values_stay_identical(self):
        frozen = {name: p.detach().clone() for name, p in self.model.named_parameters() if not p.requires_grad}
        self.assertGreater(len(frozen), 0)
        trainable = {name for name, p in self.model.named_parameters() if p.requires_grad}
        self.assertTrue(all(name.startswith("head.") or name.endswith((".a", ".b")) for name in trainable))
        tokens, labels = tensor_batch(self.splits["train"][::10], self.vocabulary)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.01)
        # Zero-initialized LoRA B makes A's first gradient zero; verify after two updates.
        for _ in range(2):
            optimizer.zero_grad()
            loss = multi_question_loss(self.model(tokens), labels)
            loss.backward()
            optimizer.step()
        for name, param in self.model.named_parameters():
            if name in frozen:
                self.assertIsNone(param.grad, name)
                self.assertTrue(torch.equal(param, frozen[name]), name)
        for suffix in (".a", ".b"):
            gradients = [p.grad.abs().sum() for n, p in self.model.named_parameters() if n.endswith(suffix)]
            self.assertTrue(all(float(g) > 0 for g in gradients))
        self.assertGreater(float(self.model.head.score[-1].weight.grad.abs().sum()), 0)

    def test_question_distributions_are_separate_and_ignore_padding(self):
        short = "duplicate charge"
        long = "I need help with invoice payment. Please explain the next steps."
        self.model.eval()
        with torch.no_grad():
            individual = self.model(self.vocabulary.batch([short]))
            batched = self.model(self.vocabulary.batch([short, long]))
        distributions = probabilities(batched, [1., 1.])
        self.assertEqual([tuple(p.shape) for p in distributions], [(2, 3), (2, 2)])
        for raw, mixed, probs in zip(individual, batched, distributions):
            torch.testing.assert_close(raw[0], mixed[0], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(probs.sum(-1), torch.ones(2))
        modified = [batched[0], batched[1] * 10]
        torch.testing.assert_close(probabilities(modified, [1., 1.])[0], distributions[0])

    def test_temperature_uses_detached_held_out_scores_and_preserves_choices(self):
        # Confident errors should lead to a softer held-out distribution.
        logits = [torch.tensor([[8., 0., 0.], [0., 8., 0.], [0., 0., 8.]], requires_grad=True),
                  torch.tensor([[8., 0.], [0., 8.], [8., 0.]], requires_grad=True)]
        labels = torch.tensor([[0, 0], [0, 0], [2, 1]])
        temperatures = fit_temperatures(logits, labels)
        for i, (scores, temperature) in enumerate(zip(logits, temperatures)):
            self.assertGreater(temperature, 1.)
            self.assertIsNone(scores.grad)
            self.assertLessEqual(float(F.cross_entropy(scores.detach() / temperature, labels[:, i])),
                                 float(F.cross_entropy(scores.detach(), labels[:, i])))
            self.assertTrue(torch.equal(scores.argmax(-1), (scores / temperature).argmax(-1)))
        with self.assertRaises(ValueError):
            probabilities(logits, [0., 1.])

    def test_loss_is_mean_of_all_question_cross_entropies(self):
        tokens, labels = tensor_batch(self.splits["train"][:3], self.vocabulary)
        logits = self.model(tokens)
        expected = sum(F.cross_entropy(logits[i], labels[:, i]) for i in range(len(QUESTIONS))) / len(QUESTIONS)
        torch.testing.assert_close(multi_question_loss(logits, labels), expected)


if __name__ == "__main__":
    unittest.main()
