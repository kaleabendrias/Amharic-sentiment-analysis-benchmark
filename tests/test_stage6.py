"""Guard the frozen choice and final-test boundary without reading real test rows."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data import ROOT
from src.final_evaluate import run_final, validate_freeze

FROZEN_RUN = ROOT / json.loads((ROOT / "configs/final_selection.json").read_text(encoding="utf-8"))["selected_run_directory"]


@unittest.skipUnless((FROZEN_RUN / "model.joblib").is_file(),
                     "The original frozen model artifact is not included in a clean clone.")
class StageSixTests(unittest.TestCase):
    def test_frozen_choice_matches_selected_saved_run(self):
        freeze, selection, ablation, run_dir = validate_freeze()
        self.assertEqual(freeze["selected_run_id"], selection["selected_run_id"])
        self.assertEqual(run_dir.name, freeze["selected_run_id"])
        self.assertLess(ablation["macro_f1_delta"], 0)

    def test_changed_model_hash_is_rejected_before_test_load(self):
        freeze = json.loads((ROOT / "configs/final_selection.json").read_text(encoding="utf-8"))
        freeze["model_file_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong_freeze.json"
            path.write_text(json.dumps(freeze), encoding="utf-8")
            with patch("src.final_evaluate.load_split") as loader:
                with self.assertRaisesRegex(ValueError, "model file differs"):
                    run_final(
                        freeze_path=path,
                        report_path=Path(directory) / "report.json",
                        results_path=Path(directory) / "results.csv",
                        local_dir=Path(directory) / "local",
                    )
                loader.assert_not_called()

    def test_synthetic_rows_use_one_prediction_pass_and_second_run_is_blocked(self):
        fake_rows = [
            {"tweet": "ጥሩ ነው", "label": "positive"},
            {"tweet": "መጥፎ ነው", "label": "negative"},
            {"tweet": "መረጃ ነው", "label": "neutral"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            kwargs = {
                "report_path": base / "report.json",
                "results_path": base / "results.csv",
                "local_dir": base / "local",
            }
            with patch("src.final_evaluate.load_split", return_value=fake_rows) as loader:
                result = run_final(**kwargs)
                loader.assert_called_once_with("test")
            self.assertEqual(result["test_metrics"]["evaluation_rows"], 3)
            self.assertEqual(sum(result["true_label_counts"].values()), 3)
            with (base / "local/predictions.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            with (base / "results.csv").open(encoding="utf-8", newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 5)
            self.assertEqual(records[-1]["evaluation_split"], "test")
            with patch("src.final_evaluate.load_split") as loader:
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    run_final(**kwargs)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
