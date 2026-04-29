import unittest

from src.publication_metrics import aggregate_runs, best_by_metric


class PublicationMetricsTest(unittest.TestCase):
    def test_aggregate_runs_reports_mean_and_std_for_each_method(self):
        runs = [
            {
                "seed": 42,
                "methods": {
                    "baseline": {"accuracy": 0.87, "macro_f1": 0.88},
                    "gan_augmented": {"accuracy": 0.89, "macro_f1": 0.90},
                },
            },
            {
                "seed": 123,
                "methods": {
                    "baseline": {"accuracy": 0.89, "macro_f1": 0.86},
                    "gan_augmented": {"accuracy": 0.91, "macro_f1": 0.92},
                },
            },
        ]

        rows = aggregate_runs(runs, metrics=("accuracy", "macro_f1"))

        self.assertEqual(
            rows,
            [
                {
                    "method": "baseline",
                    "n": 2,
                    "accuracy_mean": 0.88,
                    "accuracy_std": 0.014142,
                    "macro_f1_mean": 0.87,
                    "macro_f1_std": 0.014142,
                },
                {
                    "method": "gan_augmented",
                    "n": 2,
                    "accuracy_mean": 0.9,
                    "accuracy_std": 0.014142,
                    "macro_f1_mean": 0.91,
                    "macro_f1_std": 0.014142,
                },
            ],
        )

    def test_best_by_metric_returns_highest_metric_value(self):
        rows = [
            {"method": "baseline", "macro_f1_mean": 0.87},
            {"method": "gan_augmented", "macro_f1_mean": 0.91},
        ]

        self.assertEqual(best_by_metric(rows, "macro_f1_mean")["method"], "gan_augmented")


if __name__ == "__main__":
    unittest.main()
