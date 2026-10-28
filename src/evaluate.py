"""Score predictions, or re-evaluate a saved model on development data."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data import LABELS, load_split, manifest


def evaluate_predictions(true_labels, predicted_labels, labels=LABELS):
    """Return JSON-ready metrics with an explicit, fixed class order."""
    true_labels, predicted_labels = list(true_labels), list(predicted_labels)
    labels = list(labels)
    if not true_labels or len(true_labels) != len(predicted_labels):
        raise ValueError("Evaluation requires equally sized, nonempty label lists.")
    if (set(true_labels) | set(predicted_labels)) - set(labels):
        raise ValueError("Evaluation contains an unexpected label.")
    report = classification_report(
        true_labels, predicted_labels, labels=labels,
        output_dict=True, zero_division=0,
    )
    return {
        "evaluation_rows": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
        "confusion_matrix": {
            "labels": labels,
            "rows_are": "true labels",
            "columns_are": "predicted labels",
            "counts": confusion_matrix(true_labels, predicted_labels, labels=labels).tolist(),
        },
        "zero_division": 0,
    }


def evaluate_model(model, rows, labels=LABELS):
    """Predict from an already-fitted model. No fitting happens here."""
    predictions = [str(label) for label in model.predict([row["tweet"] for row in rows])]
    metrics = evaluate_predictions([row["label"] for row in rows], predictions, labels)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["dev"], default="dev")
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata["train_split"] != "train" or metadata["evaluation_split"] != "dev":
        raise ValueError("This lesson supports train-fitted, development-evaluated runs only.")
    current = manifest()
    dev_entry = next(f for f in current["files"] if f["local_path"] == "data/raw/amh/dev.tsv")
    if (current["revision"] != metadata["dataset"]["revision"]
            or dev_entry["sha256"] != metadata["dataset"]["files"]["dev"]["sha256"]):
        raise ValueError("The current development dataset differs from the saved run.")
    model = joblib.load(args.run_dir / "model.joblib")
    rows = load_split(args.eval_split)
    metrics, _ = evaluate_model(model, rows, metadata["labels"])
    saved = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    if metrics != saved:
        raise ValueError("Re-evaluated metrics differ from the saved metrics.")
    print(json.dumps({"matches_saved_metrics": True, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
