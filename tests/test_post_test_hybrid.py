"""Check the declared architecture study and fold-local feature fitting."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sklearn.pipeline import FeatureUnion

from src.post_test_hybrid import build_candidate, run, validate_plan


class PostTestHybridTests(unittest.TestCase):
    def test_tampered_candidate_rejected_before_data_load(self):
        plan, _ = validate_plan()
        plan["candidates"][1]["C"] = 3.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with patch("src.post_test_hybrid.load_split") as loader:
                with self.assertRaisesRegex(ValueError, "candidate protocol"):
                    run(plan_path=path, report_path=Path(directory) / "report.json")
                loader.assert_not_called()

    def test_hybrid_vocabulary_fits_only_on_supplied_text(self):
        plan, baseline = validate_plan()
        spec = plan["candidates"][1]
        model = build_candidate(spec, baseline)
        self.assertIsInstance(model.named_steps["features"], FeatureUnion)
        self.assertEqual(model.named_steps["classifier"].class_weight, "balanced")
        model.fit(["ጥሩ ነው", "መጥፎ ነው", "አዎ ጥሩ", "አይ መጥፎ"],
                  ["positive", "negative", "positive", "negative"])
        word = model.named_steps["features"].transformer_list[0][1]
        self.assertNotIn("unseenvalidationtoken", word.vocabulary_)

    def test_existing_result_blocks_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "result.json"
            report.write_text("{}", encoding="utf-8")
            with patch("src.post_test_hybrid.load_split") as loader:
                with self.assertRaises(FileExistsError):
                    run(report_path=report)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
