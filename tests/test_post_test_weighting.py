"""Check the predeclared follow-up and its duplicate-safe train-only folds."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import ROOT, overlap_key
from src.post_test_weighting import (
    configured_model, make_folds, run, sqrt_inverse_weights, validate_plan,
)


class PostTestWeightingTests(unittest.TestCase):
    def test_plan_rejects_changed_candidate_before_data_load(self):
        plan, _ = validate_plan()
        changed = dict(plan)
        changed["C"] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed_plan.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with patch("src.post_test_weighting.load_split") as loader:
                with self.assertRaisesRegex(ValueError, "candidate set changed"):
                    run(plan_path=path, report_path=Path(directory) / "result.json")
                loader.assert_not_called()

    def test_sqrt_weights_use_only_given_training_labels(self):
        labels = ["negative"] * 3 + ["neutral"] * 2 + ["positive"]
        weights = sqrt_inverse_weights(labels)
        self.assertAlmostEqual(weights["negative"], math.sqrt(6 / (3 * 3)))
        self.assertAlmostEqual(weights["neutral"], 1.0)
        self.assertAlmostEqual(weights["positive"], math.sqrt(6 / (3 * 1)))
        _, baseline = validate_plan()
        config, model = configured_model(baseline, 2.0, "sqrt_inverse", labels)
        self.assertEqual(model.named_steps["classifier"].class_weight, weights)
        self.assertIsNone(baseline["classifier"]["class_weight"])
        self.assertEqual(config["preprocessing"], "identity")

    def test_normalized_duplicate_stays_within_one_fold(self):
        plan, _ = validate_plan()
        texts = [f"ልዩ-{i}" for i in range(30)]
        labels = ["negative"] * 10 + ["neutral"] * 10 + ["positive"] * 10
        texts[0] = "ጥሩ  ነው"
        texts[1] = "ጥሩ ነው"
        folds = make_folds(texts, labels, plan)
        self.assertEqual(len(folds), 5)
        self.assertEqual(overlap_key(texts[0]), overlap_key(texts[1]))
        for training, validation in folds:
            self.assertFalse(
                {overlap_key(texts[i]) for i in training}
                & {overlap_key(texts[i]) for i in validation}
            )

    def test_existing_report_blocks_rerun_before_data_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{}", encoding="utf-8")
            with patch("src.post_test_weighting.load_split") as loader:
                with self.assertRaises(FileExistsError):
                    run(report_path=path)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
