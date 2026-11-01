"""Run the one declared whitespace ablation against the fixed character C=2 baseline."""

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.data import ROOT
from src.train import run_model


def predictions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["predicted_label"] for row in csv.DictReader(handle)]


def run(config_path=ROOT / "configs/whitespace_ablation.yaml",
        report_path=ROOT / "reports/whitespace_ablation.json"):
    baseline_report = json.loads((ROOT / "reports/char_dev.json").read_text(encoding="utf-8"))
    baseline_run = ROOT / "artifacts" / baseline_report["reference_grid_id"] / baseline_report["selected_run_id"]
    baseline_config = yaml.safe_load((baseline_run / "config_source.yaml").read_text(encoding="utf-8"))
    ablation_config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if baseline_config["preprocessing"] != "identity" or ablation_config["preprocessing"] != "whitespace_collapse":
        raise ValueError("Expected identity baseline and whitespace-only ablation.")
    if {k: v for k, v in ablation_config.items() if k != "preprocessing"} != {
        k: v for k, v in baseline_config.items() if k != "preprocessing"
    }:
        raise ValueError("The ablation must change only the preprocessing option.")
    if baseline_report["model"] != "char_tfidf" or baseline_report["test_split_loaded"]:
        raise ValueError("Only the selected character development run is eligible.")
    C = baseline_report["selected_C"]  # Fixed in advance; no new C search.
    baseline_metrics = json.loads((baseline_run / "metrics.json").read_text(encoding="utf-8"))
    baseline_metadata = json.loads((baseline_run / "metadata.json").read_text(encoding="utf-8"))
    run_dir, ablated_metrics = run_model(
        config_path=Path(config_path), train_split="train", eval_split="dev",
        model_name="char_tfidf", C=C,
    )
    ablated_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    for split in ("train", "dev"):
        if (ablated_metadata["dataset"]["files"][split]["sha256"]
                != baseline_metadata["dataset"]["files"][split]["sha256"]):
            raise ValueError(f"The {split} dataset changed between experiments.")
    if not ablated_metadata["converged"]:
        raise RuntimeError("Ablation classifier did not converge; do not compare scores yet.")
    old_predictions = predictions(baseline_run / "predictions.csv")
    new_predictions = predictions(run_dir / "predictions.csv")
    if len(old_predictions) != len(new_predictions) or len(old_predictions) != baseline_metrics["evaluation_rows"]:
        raise ValueError("Prediction row counts differ.")
    summary = {
        "intervention": "Collapse internal Unicode whitespace to one ASCII space and trim edges before character TF-IDF.",
        "other_settings_changed": False,
        "fixed_C": C,
        "train_split": "train", "evaluation_split": "dev", "test_split_loaded": False,
        "baseline_run_id": baseline_run.name, "ablation_run_id": run_dir.name,
        "ablation_run_directory": str(run_dir.relative_to(ROOT)),
        "dataset_revision": ablated_metadata["dataset"]["revision"],
        "train_sha256": ablated_metadata["dataset"]["files"]["train"]["sha256"],
        "dev_sha256": ablated_metadata["dataset"]["files"]["dev"]["sha256"],
        "baseline_metrics": baseline_metrics, "ablation_metrics": ablated_metrics,
        "macro_f1_delta": ablated_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "accuracy_delta": ablated_metrics["accuracy"] - baseline_metrics["accuracy"],
        "weighted_f1_delta": ablated_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
        "changed_predictions": sum(a != b for a, b in zip(old_predictions, new_predictions, strict=True)),
        "comparison_note": "One development-set preprocessing ablation; no final-test inference or significance claim.",
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/whitespace_ablation.yaml")
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({
        "ablation_run_directory": result["ablation_run_directory"],
        "baseline_macro_f1": result["baseline_metrics"]["macro_f1"],
        "ablation_macro_f1": result["ablation_metrics"]["macro_f1"],
        "macro_f1_delta": result["macro_f1_delta"],
        "changed_predictions": result["changed_predictions"],
    }, indent=2))
