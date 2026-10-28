"""Compare the three train-fitted model families using development results only."""

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.data import ROOT
from src.models import select_family_candidate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def dataset_identity(dataset):
    return (dataset["revision"], tuple(
        dataset["files"][split]["sha256"] for split in ("train", "dev")
    ))


def compare_development(char_grid_dir, majority_report=ROOT / "reports/majority_dev.json",
                        word_report=ROOT / "reports/word_dev.json",
                        config_path=ROOT / "configs/baselines.yaml",
                        output_dir=ROOT / "reports"):
    """Validate provenance, then write aggregate character and family reports."""
    majority = read_json(majority_report)
    word = read_json(word_report)
    char_grid_dir = Path(char_grid_dir)
    char = read_json(char_grid_dir / "selection.json")
    selected_char = next(
        item for item in char["candidates"] if item["run_id"] == char["selected_run_id"]
    )
    char_run_dir = char_grid_dir / char["selected_run_id"]
    char_metadata = read_json(char_run_dir / "metadata.json")
    if read_json(char_run_dir / "metrics.json") != selected_char["metrics"]:
        raise ValueError("Character grid metrics differ from the saved selected run.")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if [majority["model"], word["model"], char["model"]] != [
        "majority", "word_tfidf", "char_tfidf"
    ]:
        raise ValueError("The reports must contain the three expected model families.")
    for report in (majority, word, char):
        if (report["train_split"], report["evaluation_split"]) != ("train", "dev"):
            raise ValueError("Only train-fitted development results can be compared.")
        if report["test_split_loaded"]:
            raise ValueError("A report has loaded the test split.")
    if not (word["primary_metric"] == char["primary_metric"] == config["primary_metric"]):
        raise ValueError("The primary metric differs across experiments.")
    if not (word["selection_policy"] == char["selection_policy"] == config["selection"]):
        raise ValueError("The declared selection policy differs across experiments.")
    if [item["C"] for item in word["candidates"]] != [item["C"] for item in char["candidates"]]:
        raise ValueError("The word and character C grids differ.")
    if [item["C"] for item in char["candidates"]] != sorted(config["classifier"]["C_candidates"]):
        raise ValueError("Candidate C values differ from the current specification.")
    if not all(item["run_id"] != "" for item in char["candidates"]):
        raise ValueError("A character run has no ID.")
    if selected_char["C"] != char["selected_C"]:
        raise ValueError("Character winner C and run ID disagree.")
    selected_word = next(
        item for item in word["candidates"] if item["run_id"] == word["selected_run_id"]
    )
    if selected_word["C"] != word["selected_C"]:
        raise ValueError("Word winner C and run ID disagree.")
    identities = [dataset_identity(report["dataset"]) for report in (
        majority, word["selected_metadata"], char_metadata
    )]
    if len(set(identities)) != 1:
        raise ValueError("Training or development data differ across model families.")
    for expected_model, expected_C, metadata in (
        ("word_tfidf", selected_word["C"], word["selected_metadata"]),
        ("char_tfidf", selected_char["C"], char_metadata),
    ):
        if metadata["model"] != expected_model or metadata["candidate_C"] != expected_C:
            raise ValueError("A selected model's metadata disagrees with its family or C value.")
        if (metadata["train_split"], metadata["evaluation_split"]) != ("train", "dev"):
            raise ValueError("A selected model used unexpected split roles.")
        if metadata["test_split_loaded"] or metadata["seed"] != config["seed"]:
            raise ValueError("A selected model used test data or a different seed.")
        if metadata["labels"] != config["labels"]:
            raise ValueError("The label order differs from the specification.")
    if majority["seed"] != config["seed"]:
        raise ValueError("The majority run used a different seed.")

    winners = [
        {"model": "majority", "C": None, "run_id": majority["reference_run_id"],
         "metrics": majority["metrics"]},
        {"model": "word_tfidf", "C": selected_word["C"],
         "run_id": selected_word["run_id"], "metrics": selected_word["metrics"]},
        {"model": "char_tfidf", "C": selected_char["C"],
         "run_id": selected_char["run_id"], "metrics": selected_char["metrics"]},
    ]
    winner = select_family_candidate(winners, config["selection"])
    char_report = {
        "model": "char_tfidf", "reference_grid_id": char_grid_dir.name,
        "train_split": "train", "evaluation_split": "dev", "primary_metric": "macro_f1",
        "selection_policy": char["selection_policy"], "config_sha256": char["config_sha256"],
        "candidates": char["candidates"], "selected_C": char["selected_C"],
        "selected_run_id": char["selected_run_id"], "selected_metadata": char_metadata,
        "test_split_loaded": False,
    }
    selection = {
        "evaluation_split": "dev", "training_split": "train", "primary_metric": "macro_f1",
        "selection_policy": config["selection"],
        "dataset_revision": identities[0][0],
        "train_sha256": identities[0][1][0], "dev_sha256": identities[0][1][1],
        "family_winners": winners,
        "selected_model": winner["model"], "selected_C": winner["C"],
        "selected_run_id": winner["run_id"],
        "test_split_loaded": False,
        "selection_scope": "Development selection only; held-out test evaluation has not run.",
    }
    rows = []
    for model, candidates in (
        ("majority", winners[:1]), ("word_tfidf", word["candidates"]),
        ("char_tfidf", char["candidates"]),
    ):
        for candidate in candidates:
            metrics = candidate["metrics"]
            rows.append({
                "model": model, "C": candidate.get("C"), "evaluation_split": "dev",
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "selected_within_family": candidate["run_id"] == next(
                    item["run_id"] for item in winners if item["model"] == model
                ),
                "selected_overall": candidate["run_id"] == winner["run_id"],
                "reference_run_id": candidate["run_id"],
            })
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "char_dev.json", char_report)
    save_json(output_dir / "baseline_selection.json", selection)
    with (output_dir / "development_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--char-grid-dir", type=Path, required=True)
    parser.add_argument("--majority-report", type=Path, default=ROOT / "reports/majority_dev.json")
    parser.add_argument("--word-report", type=Path, default=ROOT / "reports/word_dev.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    selection = compare_development(
        args.char_grid_dir, args.majority_report, args.word_report, args.config, args.output_dir
    )
    print(json.dumps({
        "selected_model": selection["selected_model"],
        "selected_C": selection["selected_C"],
        "selected_run_id": selection["selected_run_id"],
        "test_split_loaded": selection["test_split_loaded"],
    }, indent=2))


if __name__ == "__main__":
    main()
