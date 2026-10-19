"""Guard against corrupt downloads, malformed rows and Ethiopic text loss."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.data import overlap_key, read_tsv, verify_file


class DataIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.tsv"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")
        return self.path

    def test_text_survives_utf8_tsv_loading(self):
        text = "እወደዋለሁ 😊 good!"
        self.write(f"tweet\tlabel\n{text}\tpositive\n")
        self.assertEqual(read_tsv(self.path), [{"tweet": text, "label": "positive"}])

    def test_corrupt_download_is_rejected(self):
        self.write("original")
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        verify_file(self.path, digest)
        self.write("changed")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            verify_file(self.path, digest)

    def test_malformed_rows_are_rejected(self):
        examples = [
            "text\tlabel\nጥሩ\tpositive\n",
            "tweet\tlabel\nጥሩ\tunknown\n",
            "tweet\tlabel\n \tneutral\n",
            "tweet\tlabel\nጥሩ\tpositive\textra\n",
            "tweet\tlabel\nጥሩ\n",
            "tweet\tlabel\n",
        ]
        for text in examples:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    read_tsv(self.write(text))

    def test_overlap_key_preserves_script_emoji_and_negation(self):
        self.assertEqual(overlap_key("  ጥሩ\nአይደለም 😊  "), "ጥሩ አይደለም 😊")
        self.assertNotEqual(overlap_key("ሀ"), overlap_key("ሐ"))


if __name__ == "__main__":
    unittest.main()
