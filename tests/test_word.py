"""Check TF-IDF leakage, frozen inference, full candidate retention and tie rules."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from src.data import ROOT
from src.evaluate import evaluate_model
from src.models import build_model, select_word_candidate
from src.train import run_word_grid


def training_examples():
    return [
        {"tweet": text, "label": label}
        for text, label in (("ጥሩ ነው", "positive"), ("መጥፎ ነው", "negative"), ("መረጃ ነው", "neutral"))
        for _ in range(2)
    ]


class WordTests(unittest.TestCase):
    def test_train_only_vocabulary_and_idf_remain_frozen_after_reload(self):
        config = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text())
        model = build_model(config, "word_tfidf", 1.0)
        training = training_examples()
        development = [
            {"tweet": "የልምምድብቻ ጥሩ", "label": "positive"},
            {"tweet": "የልምምድብቻ መጥፎ", "label": "negative"},
        ]
        with threadpool_limits(limits=1):
            model.fit([row["tweet"] for row in training], [row["label"] for row in training])
        vectorizer = model.named_steps["tfidf"]
        self.assertIn("ጥሩ", vectorizer.vocabulary_)
        self.assertNotIn("የልምምድብቻ", vectorizer.vocabulary_)
        self.assertAlmostEqual(vectorizer.idf_[vectorizer.vocabulary_["ነው"]], 1.0)
        vocabulary, idf = dict(vectorizer.vocabulary_), vectorizer.idf_.copy()
        original_metrics, original_predictions = evaluate_model(model, development)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
        restored_vectorizer = restored.named_steps["tfidf"]
        with patch.object(restored_vectorizer, "fit", side_effect=AssertionError("Inference must not fit")), patch.object(restored_vectorizer, "fit_transform", side_effect=AssertionError("Inference must not fit")):
            metrics, predictions = evaluate_model(restored, development)
        self.assertEqual(metrics, original_metrics)
        self.assertEqual(predictions, original_predictions)
        self.assertEqual(restored_vectorizer.vocabulary_, vocabulary)
        np.testing.assert_array_equal(restored_vectorizer.idf_, idf)

    def test_grid_retains_all_candidates_and_reads_train_dev_only(self):
        training = training_examples()
        development = [dict(row) for row in training]
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.train.load_split", side_effect=lambda split: {"train": training, "dev": development}[split]) as loader:
                grid, summary = run_word_grid(output_dir=Path(directory))
            self.assertEqual([call.args[0] for call in loader.call_args_list], ["train", "dev"] * 3)
            self.assertEqual([row["C"] for row in summary["candidates"]], [0.5, 1.0, 2.0])
            self.assertEqual(summary["selected_C"], 0.5)  # All three achieve the same score here.
            for candidate in summary["candidates"]:
                child = grid / candidate["run_id"]
                metadata = json.loads((child / "metadata.json").read_text())
                self.assertEqual(metadata["model"], "word_tfidf")
                self.assertEqual(metadata["model_parameters"]["classifier"]["parameters"]["C"], candidate["C"])
                self.assertTrue((child / "predictions.csv").exists())
                self.assertTrue(metadata["converged"])
            self.assertEqual(json.loads((grid / "selection.json").read_text()), summary)

    def test_rounded_score_tie_prefers_lower_C(self):
        candidates = [
            {"C": 2.0, "metrics": {"macro_f1": 0.50000049}},
            {"C": 0.5, "metrics": {"macro_f1": 0.5000004}},
        ]
        policy = {"score_rounding_decimals": 6, "tied_C_order": "ascending"}
        self.assertEqual(select_word_candidate(candidates, policy)["C"], 0.5)
        candidates[0]["metrics"]["macro_f1"] = 0.51
        self.assertEqual(select_word_candidate(candidates, policy)["C"], 2.0)

    def test_grid_rejects_test_evaluation_before_loading_data(self):
        with patch("src.train.load_split") as loader:
            with self.assertRaises(ValueError):
                run_word_grid(eval_split="test")
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
