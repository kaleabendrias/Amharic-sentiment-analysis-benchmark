"""Inspect a small, balanced train/dev sample without fitting a model."""

import argparse
import json
import random
import re
import unicodedata
from collections import Counter

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import LABELS, ROOT, load_split, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be at least 1")

    config = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text(encoding="utf-8"))
    seed = config["seed"]
    rng = random.Random(seed)
    word_config = dict(config["word_tfidf"])
    word_config["ngram_range"] = tuple(word_config["ngram_range"])
    vectorizer = TfidfVectorizer(**word_config)
    tokenize = vectorizer.build_tokenizer()
    preprocess = vectorizer.build_preprocessor()
    samples = []
    split_sizes = {}

    # Deliberately inspect train/dev only. Final test examples stay reserved.
    for split in ("train", "dev"):
        rows = load_split(split)
        split_sizes[split] = len(rows)
        for label in LABELS:
            candidates = [(index, row) for index, row in enumerate(rows) if row["label"] == label]
            if len(candidates) < args.per_class:
                parser.error(f"Not enough {label} examples in {split}")
            for index, row in rng.sample(candidates, args.per_class):
                text = row["tweet"]
                samples.append({
                    "split": split,
                    "row_index_zero_based": index,
                    "label": label,
                    "tweet": text,
                    "word_tokens": tokenize(preprocess(text)),
                    "characters": len(text),
                    "has_ethiopic_characters": any("ETHIOPIC" in unicodedata.name(c, "") for c in text),
                    "has_ascii_latin_letters": bool(re.search(r"[A-Za-z]", text)),
                    "has_punctuation": any(unicodedata.category(c).startswith("P") for c in text),
                    "has_unicode_symbols": any(unicodedata.category(c).startswith("S") for c in text),
                })

    # Raw messages and their tokens remain local under the ignored artifacts/ directory.
    local = ROOT / "artifacts/data_inspection/samples.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_revision": manifest()["revision"],
        "seed": seed,
        "sample_size": len(samples),
        "examples_per_class_per_split": args.per_class,
        "sampling": "Fixed-seed random sampling within each label in train and dev",
        "population_split_sizes": split_sizes,
        "selected_labels_by_split": {
            split: dict(Counter(row["label"] for row in samples if row["split"] == split))
            for split in ("train", "dev")
        },
        "selected_rows_with": {
            flag: sum(row[flag] for row in samples)
            for flag in (
                "has_ethiopic_characters", "has_ascii_latin_letters",
                "has_punctuation", "has_unicode_symbols",
            )
        },
        "selected_rows_with_no_word_tokens": sum(not row["word_tokens"] for row in samples),
        "word_tokenizer_config": config["word_tfidf"],
        "raw_sample_file": "artifacts/data_inspection/samples.json",
        "test_split_loaded": False,
        "model_fitted": False,
        "interpretation": "A small label-balanced inspection sample, not a prevalence estimate or a performance result. Latin letters may be names, URLs or other text; they do not prove code-switching. Unicode symbols are not an emoji detector.",
    }
    destination = ROOT / "reports/data_inspection.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
