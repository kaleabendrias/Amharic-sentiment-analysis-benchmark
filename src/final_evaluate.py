"""Evaluate the pre-frozen train-only model on the held-out test split once."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml

from src.data import LABELS, ROOT, load_split, manifest
from src.evaluate import evaluate_model


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_freeze(freeze_path=ROOT / "configs/final_selection.json"):
    """Check the development decision and artifact without opening the test TSV."""
    freeze_path = Path(freeze_path)
    freeze = read_json(freeze_path)
    if (freeze["training_split"], freeze["selection_split"], freeze["final_evaluation_split"]) != (
        "train", "dev", "test"
    ):
        raise ValueError("Frozen split roles changed.")
    if freeze["retrain_on_development"] or freeze["test_metrics_known_at_freeze"]:
        raise ValueError("Frozen protocol must use the train-only model before test results exist.")
    if (freeze["model"], freeze["C"], freeze["preprocessing"], freeze["primary_metric"]) != (
        "char_tfidf", 2.0, "identity", "macro_f1"
    ) or freeze["labels"] != list(LABELS):
        raise ValueError("Frozen model, preprocessing, or metric changed.")

    source = manifest()
    if source["revision"] != freeze["dataset_revision"]:
        raise ValueError("Dataset revision differs from the freeze.")
    for split in ("train", "dev", "test"):
        entry = next(f for f in source["files"] if f["local_path"] == f"data/raw/amh/{split}.tsv")
        if entry["sha256"] != freeze[f"{split}_sha256"]:
            raise ValueError(f"Pinned {split} hash differs from the freeze.")

    paths = {
        "baseline_config_sha256": ROOT / "configs/baselines.yaml",
        "baseline_selection_report_sha256": ROOT / "reports/baseline_selection.json",
        "ablation_report_sha256": ROOT / "reports/whitespace_ablation.json",
    }
    for key, path in paths.items():
        if sha256(path) != freeze[key]:
            raise ValueError(f"Frozen evidence changed: {path.name}")
    config = yaml.safe_load(paths["baseline_config_sha256"].read_text(encoding="utf-8"))
    if config["preprocessing"] != "identity" or config["primary_metric"] != "macro_f1":
        raise ValueError("The baseline configuration no longer matches the freeze.")
    selection = read_json(paths["baseline_selection_report_sha256"])
    if selection["test_split_loaded"] or (
        selection["selected_model"], selection["selected_C"], selection["selected_run_id"]
    ) != (freeze["model"], freeze["C"], freeze["selected_run_id"]):
        raise ValueError("Development selection no longer matches the freeze.")
    if (selection["dataset_revision"], selection["train_sha256"], selection["dev_sha256"]) != (
        freeze["dataset_revision"], freeze["train_sha256"], freeze["dev_sha256"]
    ):
        raise ValueError("Development selection used different data.")
    ablation = read_json(paths["ablation_report_sha256"])
    if ablation["test_split_loaded"] or ablation["baseline_run_id"] != freeze["selected_run_id"]:
        raise ValueError("Ablation provenance no longer matches the freeze.")
    if ablation["fixed_C"] != freeze["C"] or ablation["macro_f1_delta"] >= 0:
        raise ValueError("Ablation no longer supports the frozen choice.")

    run_dir = (ROOT / freeze["selected_run_directory"]).resolve()
    if not run_dir.is_relative_to((ROOT / "artifacts").resolve()) or run_dir.name != freeze["selected_run_id"]:
        raise ValueError("Frozen model path is outside the expected artifact directory.")
    metadata = read_json(run_dir / "metadata.json")
    if (metadata["run_id"], metadata["model"], metadata["candidate_C"],
            metadata["train_split"], metadata["evaluation_split"]) != (
        freeze["selected_run_id"], freeze["model"], freeze["C"], "train", "dev"
    ) or metadata["test_split_loaded"] or not metadata["converged"]:
        raise ValueError("Saved model metadata does not match the freeze.")
    if metadata["config_sha256"] != freeze["baseline_config_sha256"]:
        raise ValueError("Saved model used a different baseline configuration.")
    for split in ("train", "dev"):
        if metadata["dataset"]["files"][split]["sha256"] != freeze[f"{split}_sha256"]:
            raise ValueError(f"Saved model used different {split} data.")
    winner = next(x for x in selection["family_winners"] if x["model"] == freeze["model"])
    if read_json(run_dir / "metrics.json") != winner["metrics"]:
        raise ValueError("Saved development metrics differ from the selected result.")
    if sha256(run_dir / "model.joblib") != freeze["model_file_sha256"]:
        raise ValueError("Saved model file differs from the freeze.")
    return freeze, selection, ablation, run_dir


def run_final(freeze_path=ROOT / "configs/final_selection.json",
              report_path=ROOT / "reports/final_test.json",
              results_path=ROOT / "reports/results.csv",
              local_dir=ROOT / "artifacts/final_test"):
    """Write one aggregate test result and keep row-level predictions local."""
    report_path, results_path, local_dir = map(Path, (report_path, results_path, local_dir))
    if any(path.exists() for path in (report_path, results_path, local_dir)):
        raise FileExistsError("A final-test output already exists; do not rerun the held-out evaluation.")
    freeze, selection, ablation, run_dir = validate_freeze(freeze_path)
    model = joblib.load(run_dir / "model.joblib")
    if list(model.named_steps) != ["tfidf", "classifier"]:
        raise ValueError("The frozen model unexpectedly contains a preprocessing step.")
    if model.named_steps["tfidf"].analyzer != "char" or model.named_steps["classifier"].C != freeze["C"]:
        raise ValueError("The loaded estimator does not match the frozen model.")

    # This is the only call that opens the held-out test split for final scoring.
    rows = load_split("test")
    metrics, predictions = evaluate_model(model, rows, freeze["labels"])
    test_counts = Counter(row["label"] for row in rows)
    prediction_counts = Counter(predictions)
    report = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "One evaluation of the saved train-only development winner; no retraining or test-based selection.",
        "training_split": "train", "selection_split": "dev", "evaluation_split": "test",
        "dataset_revision": freeze["dataset_revision"],
        "test_sha256": freeze["test_sha256"],
        "frozen_selection_sha256": sha256(freeze_path),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "selected_run_id": freeze["selected_run_id"],
        "model_file_sha256": freeze["model_file_sha256"],
        "preprocessing": freeze["preprocessing"],
        "C": freeze["C"],
        "true_label_counts": {label: test_counts[label] for label in LABELS},
        "prediction_counts": {label: prediction_counts[label] for label in LABELS},
        "development_metrics": next(x["metrics"] for x in selection["family_winners"] if x["model"] == freeze["model"]),
        "test_metrics": metrics,
        "row_level_predictions": "artifacts/final_test/predictions.csv (ignored by Git)",
        "test_informed_changes_allowed_in_original_experiment": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=False)
    with (local_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["row_index_zero_based", "true_label", "predicted_label"])
        writer.writerows((i, row["label"], predicted) for i, (row, predicted) in
                         enumerate(zip(rows, predictions, strict=True)))
    fields = ["model", "preprocessing", "C", "evaluation_split", "selected_overall",
              "accuracy", "macro_f1", "weighted_f1", "reference_run_id"]
    table = []
    for candidate in selection["family_winners"]:
        table.append({
            "model": candidate["model"], "preprocessing": "identity", "C": candidate["C"],
            "evaluation_split": "dev", "selected_overall": candidate["run_id"] == freeze["selected_run_id"],
            **{key: candidate["metrics"][key] for key in ("accuracy", "macro_f1", "weighted_f1")},
            "reference_run_id": candidate["run_id"],
        })
    table.append({
        "model": "char_tfidf", "preprocessing": "whitespace_collapse", "C": freeze["C"],
        "evaluation_split": "dev", "selected_overall": False,
        **{key: ablation["ablation_metrics"][key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "reference_run_id": ablation["ablation_run_id"],
    })
    table.append({
        "model": freeze["model"], "preprocessing": freeze["preprocessing"], "C": freeze["C"],
        "evaluation_split": "test", "selected_overall": True,
        **{key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")},
        "reference_run_id": freeze["selected_run_id"],
    })
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "configs/final_selection.json")
    args = parser.parse_args()
    report = run_final(freeze_path=args.freeze)
    print(json.dumps({
        "selected_run_id": report["selected_run_id"],
        "test_rows": report["test_metrics"]["evaluation_rows"],
        "accuracy": report["test_metrics"]["accuracy"],
        "macro_f1": report["test_metrics"]["macro_f1"],
        "weighted_f1": report["test_metrics"]["weighted_f1"],
    }, indent=2))


if __name__ == "__main__":
    main()
