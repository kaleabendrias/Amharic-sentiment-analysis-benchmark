"""Strict loading of the original, unmodified AfriSenti TSV files."""

import csv
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("negative", "neutral", "positive")
SPLITS = ("train", "dev", "test")


def manifest():
    return json.loads((ROOT / "configs/dataset.json").read_text(encoding="utf-8"))


def verify_file(path, expected_sha256):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Checksum mismatch: {path}. Restore the pinned original file.")
    return actual


def read_tsv(path):
    """Keep the original text and reject malformed rows or unexpected labels."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["tweet", "label"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if set(row) != {"tweet", "label"} or row["tweet"] is None:
                raise ValueError(f"Malformed row at {path}:{line}")
            if row["label"] not in LABELS:
                raise ValueError(f"Unexpected label at {path}:{line}: {row['label']!r}")
            if not row["tweet"].strip():
                raise ValueError(f"Empty text at {path}:{line}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def load_split(split):
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    entry = next(f for f in manifest()["files"] if f["local_path"] == f"data/raw/amh/{split}.tsv")
    path = ROOT / entry["local_path"]
    verify_file(path, entry["sha256"])
    return read_tsv(path)


def overlap_key(text):
    """Audit only: NFC plus collapsed whitespace; no Ethiopic character mapping."""
    return " ".join(unicodedata.normalize("NFC", text).split())
