"""Check split leakage, hand-calculated metrics and saved prediction consistency."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib

from src.evaluate import evaluate_model, evaluate_predictions
from src.train import run_majority


class MajorityTests(unittest.TestCase):
    def test_metrics_match_hand_calculation_with_unpredicted_classes(self):
        metrics = evaluate_predictions(
            ["negative", "neutral", "positive", "neutral"], ["neutral"] * 4,
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 9)
        self.assertAlmostEqual(metrics["weighted_f1"], 1 / 3)
        self.assertEqual(metrics["per_class"]["positive"]["precision"], 0)
        self.assertEqual(metrics["confusion_matrix"]["counts"], [[0, 1, 0], [0, 2, 0], [0, 1, 0]])

    def test_majority_comes_from_train_even_when_dev_majority_differs(self):
        # Both splits intentionally have a different majority to expose leakage.
        training = [{"tweet": "ጥሩ", "label": label} for label in (
            "neutral", "neutral", "neutral", "negative", "positive",
        )]
        development = [{"tweet": "ነው", "label": label} for label in (
            "positive", "positive", "negative",
        )]
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.train.load_split", side_effect=[training, development]) as loader:
                run_dir, metrics = run_majority(output_dir=Path(directory))
            self.assertEqual([call.args[0] for call in loader.call_args_list], ["train", "dev"])
            self.assertEqual(metrics["accuracy"], 0)
            with (run_dir / "predictions.csv").open(newline="") as handle:
                predictions = list(csv.DictReader(handle))
            self.assertEqual([row["predicted_label"] for row in predictions], ["neutral"] * 3)
            restored_metrics, restored_predictions = evaluate_model(
                joblib.load(run_dir / "model.joblib"), development,
            )
            self.assertEqual(restored_metrics, metrics)
            self.assertEqual(restored_predictions, ["neutral"] * 3)
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["training_label_counts"]["neutral"], 3)
            self.assertFalse(metadata["test_split_loaded"])
            self.assertTrue((run_dir / "source_snapshot/src/train.py").exists())

    def test_invalid_split_is_rejected_before_loading_data(self):
        for train, evaluation in (("train", "test"), ("dev", "dev")):
            with self.subTest(train=train, evaluation=evaluation):
                with patch("src.train.load_split") as loader:
                    with self.assertRaises(ValueError):
                        run_majority(train_split=train, eval_split=evaluation)
                    loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
