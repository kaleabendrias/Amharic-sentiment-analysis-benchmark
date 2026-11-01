"""Protect the new error-review and preprocessing experiment boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import yaml
from threadpoolctl import threadpool_limits

from src.data import ROOT, load_split
from src.error_analysis import read_predictions, selected_run
from src.models import build_model
from src.preprocess import WhitespaceNormalizer
from src.train import run_model

FROZEN_RUN = ROOT / json.loads((ROOT / "configs/final_selection.json").read_text(encoding="utf-8"))["selected_run_directory"]


class StageFiveTests(unittest.TestCase):
    def test_whitespace_transform_is_small_and_saved_pipeline_is_stable(self):
        self.assertEqual(
            WhitespaceNormalizer().transform(["  ጥሩ   ነው\t።  ", "ኣበባ::"]),
            ["ጥሩ ነው ።", "ኣበባ::"],
        )
        config = yaml.safe_load((ROOT / "configs/whitespace_ablation.yaml").read_text())
        model = build_model(config, "char_tfidf", 2.0)
        texts = ["ጥሩ   ነው", "ጥሩ  ነው", "መጥፎ   ነው", "መጥፎ  ነው", "መረጃ  ነው", "መረጃ   ነው"]
        labels = ["positive", "positive", "negative", "negative", "neutral", "neutral"]
        with threadpool_limits(limits=1):
            model.fit(texts, labels)
        before = model.predict(["ጥሩ    ነው", "መጥፎ\tነው"]).tolist()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            self.assertEqual(restored.predict(["ጥሩ    ነው", "መጥፎ\tነው"]).tolist(), before)

    @unittest.skipUnless((FROZEN_RUN / "predictions.csv").is_file(),
                         "Saved development predictions are not included in a clean clone.")
    def test_saved_predictions_reject_different_development_labels(self):
        rows = load_split("dev")
        altered = [dict(row) for row in rows]
        altered[0]["label"] = "positive" if rows[0]["label"] != "positive" else "negative"
        with self.assertRaisesRegex(ValueError, "do not match"):
            read_predictions(selected_run("char"), altered)

    def test_ablation_cannot_load_test_for_selection(self):
        with patch("src.train.load_split") as loader:
            with self.assertRaises(ValueError):
                run_model(
                    config_path=ROOT / "configs/whitespace_ablation.yaml",
                    model_name="char_tfidf", C=2.0, eval_split="test",
                )
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
