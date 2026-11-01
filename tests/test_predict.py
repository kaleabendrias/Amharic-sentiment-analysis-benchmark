"""Check saved-model inference against the existing development evaluation path."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

from src.data import ROOT, load_split
from src.evaluate import evaluate_model
from src.predict import load_experimental_model, load_model, main, predict_texts

FROZEN_RUN = ROOT / json.loads((ROOT / "configs/final_selection.json").read_text(encoding="utf-8"))["selected_run_directory"]
HAS_FROZEN_MODEL = (FROZEN_RUN / "model.joblib").is_file()
EXPERIMENTAL_REPORT = ROOT / "reports/post_test_weighting.json"
EXPERIMENTAL_RUN = ROOT / json.loads(EXPERIMENTAL_REPORT.read_text(encoding="utf-8"))["selected_run_directory"]
HAS_EXPERIMENTAL_MODEL = (EXPERIMENTAL_RUN / "model.joblib").is_file()


class PredictionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_FROZEN_MODEL, "The original frozen model artifact is not included in a clean clone.")
    def test_inference_matches_evaluation_without_refitting(self):
        model, run_id, source = load_model()
        self.assertEqual(source, "frozen")
        self.assertEqual(run_id, "char_tfidf_20261028T075729163499Z_3b98ba")
        rows = load_split("dev")[:25]
        _, evaluation_predictions = evaluate_model(model, rows)
        with patch.object(model, "fit", side_effect=AssertionError("inference must not fit")):
            inference_predictions = predict_texts(model, [row["tweet"] for row in rows])
        self.assertEqual(inference_predictions, evaluation_predictions)
        recreated, _, recreated_source = load_model(ROOT / "artifacts" / "char_grid_20261028T075720208937Z_ebc8fc" / run_id)
        self.assertEqual(recreated_source, "locally_recreated")
        self.assertEqual(predict_texts(recreated, [row["tweet"] for row in rows]), evaluation_predictions)

    def test_reject_external_artifact_and_blank_input(self):
        with patch("src.predict.joblib.load") as loader:
            with self.assertRaisesRegex(ValueError, "under this repository"):
                load_model("/tmp/untrusted-run")
            loader.assert_not_called()
        with self.assertRaisesRegex(ValueError, "nonempty"):
            predict_texts(object(), ["  "])

    def test_missing_frozen_artifact_explains_recreation(self):
        with patch("src.predict.Path.is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "--run-dir"):
                load_model()

    @unittest.skipUnless(HAS_FROZEN_MODEL, "The original frozen model artifact is not included in a clean clone.")
    def test_json_cli_returns_labels_without_confidence_claim(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--text", "ጥሩ ነው", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["model_source"], "frozen")
        self.assertEqual(len(payload["predictions"]), 1)
        self.assertEqual(set(payload["predictions"][0]), {"index", "label"})
        self.assertIn(payload["predictions"][0]["label"], {"negative", "neutral", "positive"})

    @unittest.skipUnless(HAS_EXPERIMENTAL_MODEL, "The exploratory model artifact is not included in a clean clone.")
    def test_experimental_model_matches_saved_development_inference(self):
        model, run_id, source = load_experimental_model()
        self.assertEqual(source, "post_test_exploratory")
        self.assertEqual(run_id, EXPERIMENTAL_RUN.name)
        self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")
        rows = load_split("dev")[:25]
        _, evaluation_predictions = evaluate_model(model, rows)
        self.assertEqual(predict_texts(model, [row["tweet"] for row in rows]), evaluation_predictions)

    @unittest.skipUnless(HAS_EXPERIMENTAL_MODEL, "The exploratory model artifact is not included in a clean clone.")
    def test_experimental_json_marks_test_score_unavailable(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--text", "ጥሩ ነው", "--experimental-report", "--format", "json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["model_source"], "post_test_exploratory")
        self.assertIs(payload["independent_test_score_available"], False)


if __name__ == "__main__":
    unittest.main()
