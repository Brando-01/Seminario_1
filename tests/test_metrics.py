import unittest

import numpy as np

from src.evaluation import classification_metrics, select_threshold_at_fp_budget


class MetricTests(unittest.TestCase):
    def test_operating_threshold_respects_validation_false_positive_budget(self):
        labels = np.array([0, 0, 0, 0, 0, 1, 1])
        scores = np.array([0.90, 0.40, 0.30, 0.20, 0.10, 0.95, 0.35])
        threshold = select_threshold_at_fp_budget(labels, scores, max_false_positives_per_1000=200)
        metrics = classification_metrics(labels, scores, threshold)
        self.assertLessEqual(metrics["false_positives_per_1000"], 200)
        self.assertEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
