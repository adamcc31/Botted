import unittest
import math

from src.fair_probability import FairProbabilityEngine


class SigmaEstimatorTests(unittest.TestCase):
    def test_sigma_zero_for_constant_closes(self):
        closes = [100.0 for _ in range(60)]
        sigma = FairProbabilityEngine._realized_sigma_ann_from_closes(
            closes, window_n=30
        )
        self.assertGreaterEqual(sigma, 0.0)
        self.assertEqual(sigma, 0.0)

    def test_sigma_positive_for_nonconstant_series(self):
        # Alternating multiplicative moves create non-zero variance in log returns.
        step = 0.002
        closes = [100.0]
        for i in range(60):
            closes.append(closes[-1] * math.exp(step if i % 2 == 0 else -step))

        sigma = FairProbabilityEngine._realized_sigma_ann_from_closes(
            closes, window_n=30
        )
        self.assertGreater(sigma, 0.0)


if __name__ == "__main__":
    unittest.main()

