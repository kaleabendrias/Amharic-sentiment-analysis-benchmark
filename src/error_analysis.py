"""Review saved word/character development errors without publishing raw tweets."""

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data import ROOT, load_split, manifest


def selected_run(family):
    report = json.loads((ROOT / "reports" / f"{family}_dev.json").read_text(encoding="utf-8"))
    if report["model"] != f"{family}_tfidf" or report["test_split_loaded"]:
        raise ValueError(f"Invalid {family} selection report.")
    return ROOT / "artifacts" / report["reference_grid_id"] / report["selected_run_id"]


def read_predictions(run_dir, rows):
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    source = manifest()
    if metadata["train_split"] != "train" or metadata["evaluation_split"] != "dev":
        raise ValueError("Only train-fitted development predictions may be inspected.")
    if metadata["test_split_loaded"] or metadata["dataset"]["revision"] != source["revision"]:
        raise ValueError("Run provenance does not match the declared split protocol.")
    for split in ("train", "dev"):
        expected = next(f for f in source["files"] if f["local_path"] == f"data/raw/amh/{split}.tsv")
        if metadata["dataset"]["files"][split]["sha256"] != expected["sha256"]:
            raise ValueError(f"Saved {split} hash differs from the pinned dataset.")
    with (run_dir / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["row_index_zero_based", "true_label", "predicted_label"]:
            raise ValueError("Unexpected prediction columns.")
        predictions = []
        for index, prediction in enumerate(reader):
            if index >= len(rows) or int(prediction["row_index_zero_based"]) != index:
                raise ValueError("Prediction rows are missing, duplicated, or out of order.")
            if prediction["true_label"] != rows[index]["label"]:
                raise ValueError("Prediction labels do not match the current development file.")
            predictions.append(prediction["predicted_label"])
    if len(predictions) != len(rows):
        raise ValueError("Prediction count differs from development row count.")
    return predictions, metadata


def sample_indices(rows, word_predictions, char_predictions, seed=42):
    """Five from each directed character confusion plus ten model disagreements."""
    rng = random.Random(seed)
    by_confusion = defaultdict(list)
    disagreements = []
    for index, (row, word, char) in enumerate(zip(rows, word_predictions, char_predictions, strict=True)):
        if char != row["label"]:
            by_confusion[f"{row['label']}->{char}"].append(index)
        if word != char:
            disagreements.append(index)
    chosen = []
    for key in sorted(by_confusion):
        chosen.extend(rng.sample(by_confusion[key], min(5, len(by_confusion[key]))))
    remaining = [index for index in disagreements if index not in set(chosen)]
    chosen.extend(rng.sample(remaining, min(10, len(remaining))))
    return chosen, by_confusion, disagreements


def run(output_dir=ROOT / "artifacts/error_analysis", report_path=ROOT / "reports/error_analysis.json"):
    rows = load_split("dev")  # The held-out test split is never loaded here.
    word_dir, char_dir = selected_run("word"), selected_run("char")
    word, word_meta = read_predictions(word_dir, rows)
    char, char_meta = read_predictions(char_dir, rows)
    if word_meta["dataset"]["files"]["dev"]["sha256"] != char_meta["dataset"]["files"]["dev"]["sha256"]:
        raise ValueError("The model runs used different development data.")
    chosen, by_confusion, disagreements = sample_indices(rows, word, char)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Raw review text and row-level predictions stay under ignored artifacts/.
    with (output_dir / "sample40.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["row_index_zero_based", "true_label", "word_label", "char_label", "tweet"])
        for index in chosen:
            writer.writerow([index, rows[index]["label"], word[index], char[index], rows[index]["tweet"]])
    summary = {
        "train_split": "train", "evaluation_split": "dev", "test_split_loaded": False,
        "word_run_id": word_dir.name, "char_run_id": char_dir.name,
        "dataset_revision": manifest()["revision"],
        "development_sha256": char_meta["dataset"]["files"]["dev"]["sha256"],
        "development_rows": len(rows),
        "char_error_count": sum(len(group) for group in by_confusion.values()),
        "char_error_confusions": {key: len(group) for key, group in sorted(by_confusion.items())},
        "nonneutral_to_neutral_errors": sum(
            len(group) for key, group in by_confusion.items() if key.endswith("->neutral")
        ),
        "word_char_disagreements": len(disagreements),
        "char_correct_word_wrong": sum(
            char[i] == rows[i]["label"] and word[i] != rows[i]["label"] for i in disagreements
        ),
        "word_correct_char_wrong": sum(
            word[i] == rows[i]["label"] and char[i] != rows[i]["label"] for i in disagreements
        ),
        "development_rows_with_repeated_whitespace": sum(
            bool(re.search(r"\s{2,}", row["tweet"])) for row in rows
        ),
        "sample_size": len(chosen),
        "sample_repeated_whitespace": sum(bool(re.search(r"\s{2,}", rows[i]["tweet"])) for i in chosen),
        "sample_latin_letters": sum(bool(re.search(r"[A-Za-z]", rows[i]["tweet"])) for i in chosen),
        "sample_question_marks": sum("?" in rows[i]["tweet"] for i in chosen),
        "sampling_note": "Stratified error and disagreement sample; not representative prevalence estimates.",
        "local_review_file": "artifacts/error_analysis/sample40.tsv",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
