"""Check character features, split boundaries and cross-family selection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import yaml
from threadpoolctl import threadpool_limits

from src.data import ROOT
from src.evaluate import evaluate_model
from src.models import build_model, select_family_candidate
from src.train import run_char_grid


def training_examples():
    return [
        {"tweet": text, "label": label}
        for text, label in (("ጥሩ ነው", "positive"), ("መጥፎ ነው", "negative"), ("መረጃ ነው", "neutral"))
        for _ in range(2)
    ]


class CharacterTests(unittest.TestCase):
    def test_character_pipeline_keeps_dev_text_out_of_vocabulary(self):
        config = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text())
        model = build_model(config, "char_tfidf", 1.0)
        training = training_examples()
        development = [{"tweet": "የልምምድብቻ ጥሩ", "label": "positive"}]
        with threadpool_limits(limits=1):
            model.fit([row["tweet"] for row in training], [row["label"] for row in training])
        features = model.named_steps["tfidf"]
        self.assertEqual(features.analyzer, "char")
        self.assertEqual(features.ngram_range, (3, 5))
        self.assertIn("ጥሩ ", features.vocabulary_)
        self.assertNotIn("የልም", features.vocabulary_)
        metrics, predictions = evaluate_model(model, development)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
        self.assertEqual(evaluate_model(restored, development), (metrics, predictions))
        self.assertEqual(restored.named_steps["tfidf"].vocabulary_, features.vocabulary_)

    def test_character_grid_keeps_all_candidates_and_reads_no_test_split(self):
        training = training_examples()
        with tempfile.TemporaryDirectory() as directory:
            with patch("src.train.load_split", side_effect=lambda split: {"train": training, "dev": training}[split]) as loader:
                grid, summary = run_char_grid(output_dir=Path(directory))
            self.assertEqual([call.args[0] for call in loader.call_args_list], ["train", "dev"] * 3)
            self.assertEqual([row["C"] for row in summary["candidates"]], [0.5, 1.0, 2.0])
            self.assertEqual(summary["selected_C"], 0.5)
            for candidate in summary["candidates"]:
                metadata = json.loads((grid / candidate["run_id"] / "metadata.json").read_text())
                self.assertEqual(metadata["model"], "char_tfidf")
                self.assertGreater(metadata["vocabulary_size"], 0)
                self.assertTrue(metadata["converged"])
                self.assertFalse(metadata["test_split_loaded"])
        with patch("src.train.load_split") as loader:
            with self.assertRaises(ValueError):
                run_char_grid(eval_split="test")
            loader.assert_not_called()

    def test_family_selection_obeys_declared_tie_order_and_metric(self):
        policy = {"score_rounding_decimals": 6,
                  "tied_model_order": ["majority", "word_tfidf", "char_tfidf"]}
        candidates = [
            {"model": "majority", "metrics": {"macro_f1": 0.2}},
            {"model": "word_tfidf", "metrics": {"macro_f1": 0.5000004}},
            {"model": "char_tfidf", "metrics": {"macro_f1": 0.50000049}},
        ]
        self.assertEqual(select_family_candidate(candidates, policy)["model"], "word_tfidf")
        candidates[2]["metrics"]["macro_f1"] = 0.51
        self.assertEqual(select_family_candidate(candidates, policy)["model"], "char_tfidf")
        with self.assertRaises(ValueError):
            select_family_candidate(candidates[:2], policy)


if __name__ == "__main__":
    unittest.main()
