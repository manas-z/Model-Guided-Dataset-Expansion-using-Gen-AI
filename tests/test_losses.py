import unittest

import torch

from src.losses import FocalLoss


class FocalLossTest(unittest.TestCase):
    def test_gamma_zero_matches_cross_entropy(self):
        logits = torch.tensor([[2.0, 0.5], [0.2, 1.4]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)

        focal = FocalLoss(gamma=0.0)
        ce = torch.nn.CrossEntropyLoss()

        self.assertAlmostEqual(float(focal(logits, targets)), float(ce(logits, targets)), places=6)

    def test_alpha_weights_classes(self):
        logits = torch.tensor([[2.0, 0.5], [0.2, 1.4]], dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.long)
        alpha = torch.tensor([1.0, 3.0], dtype=torch.float32)

        unweighted = FocalLoss(gamma=2.0)(logits, targets)
        weighted = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)

        self.assertGreater(float(weighted), float(unweighted))


if __name__ == "__main__":
    unittest.main()
