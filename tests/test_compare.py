"""Check that cross-family selection rejects mismatched data provenance."""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from src.compare import compare_development
from src.data import ROOT


class ComparisonTests(unittest.TestCase):
    def test_comparison_selects_by_dev_metric_and_rejects_changed_dataset(self):
        word = json.loads((ROOT / "reports/word_dev.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            grid = parent / "char_grid_example"
            run = grid / "char_run_example"
            run.mkdir(parents=True)
            candidates = deepcopy(word["candidates"])
            for index, candidate in enumerate(candidates):
                candidate["run_id"] = f"char_run_{index}"
            candidates[-1]["run_id"] = run.name
            candidates[-1]["metrics"]["macro_f1"] = 0.51
            selection = {
                "model": "char_tfidf", "train_split": "train", "evaluation_split": "dev",
                "primary_metric": "macro_f1", "selection_policy": word["selection_policy"],
                "config_sha256": "synthetic", "candidates": candidates,
                "selected_C": 2.0, "selected_run_id": run.name,
                "test_split_loaded": False,
            }
            (grid / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            (run / "metrics.json").write_text(json.dumps(candidates[-1]["metrics"]), encoding="utf-8")
            metadata = deepcopy(word["selected_metadata"])
            metadata.update(model="char_tfidf", candidate_C=2.0)
            (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            result = compare_development(grid, output_dir=parent / "reports")
            self.assertEqual((result["selected_model"], result["selected_C"]), ("char_tfidf", 2.0))
            self.assertFalse(result["test_split_loaded"])
            self.assertEqual(len((parent / "reports/development_results.csv").read_text().splitlines()), 8)

            metadata["dataset"]["files"]["dev"]["sha256"] = "changed-data"
            (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data differ"):
                compare_development(grid, output_dir=parent / "reports")


if __name__ == "__main__":
    unittest.main()
