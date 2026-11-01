"""Verify the fixed transformer plan before any data or optional model import."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.post_test_transformer import run, validate_plan


class PostTestTransformerTests(unittest.TestCase):
    def test_changed_training_setting_is_rejected_before_data_load(self):
        plan = validate_plan()
        plan["max_token_length"] = 256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with patch("src.post_test_transformer.load_split") as loader:
                with self.assertRaisesRegex(ValueError, "protocol changed"):
                    run(path, Path(directory) / "report.json")
                loader.assert_not_called()

    def test_existing_report_prevents_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}", encoding="utf-8")
            with patch("src.post_test_transformer.load_split") as loader:
                with self.assertRaises(FileExistsError):
                    run(report_path=report)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
