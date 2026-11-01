"""Classify local Amharic text with a saved, train-fitted character model."""

import argparse
import json
import sys
from pathlib import Path

import joblib

from src.data import LABELS, ROOT, manifest
from src.final_evaluate import read_json, sha256, validate_freeze


def _check_pipeline(model, C):
    if list(model.named_steps) != ["tfidf", "classifier"]:
        raise ValueError("Expected the original-text TF-IDF classifier pipeline.")
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    if (vectorizer.analyzer, vectorizer.ngram_range, vectorizer.lowercase) != ("char", (3, 5), False):
        raise ValueError("Saved character features do not match the selected settings.")
    if classifier.C != C or set(model.classes_) != set(LABELS):
        raise ValueError("Saved classifier or label set differs from the selected settings.")


def load_model(run_dir=None):
    """Load the pinned artifact, or a locally recreated run with matching settings."""
    if run_dir is None:
        frozen = read_json(ROOT / "configs/final_selection.json")
        if not (ROOT / frozen["selected_run_directory"] / "model.joblib").is_file():
            raise FileNotFoundError(
                "The original frozen model is local and is not included in Git. "
                "Train the character grid, then pass its selected run with --run-dir."
            )
        freeze, _, _, path = validate_freeze()
        source = "frozen"
    else:
        freeze = read_json(ROOT / "configs/final_selection.json")
        path = Path(run_dir)
        path = (path if path.is_absolute() else ROOT / path).resolve()
        if not path.is_relative_to((ROOT / "artifacts").resolve()):
            raise ValueError("A recreated run must be under this repository's artifacts directory.")
        metadata = read_json(path / "metadata.json")
        if (metadata["model"], metadata["candidate_C"], metadata["train_split"],
                metadata["evaluation_split"], metadata["config_sha256"]) != (
            freeze["model"], freeze["C"], "train", "dev", freeze["baseline_config_sha256"]
        ) or metadata["test_split_loaded"] or not metadata["converged"]:
            raise ValueError("The recreated run is not a converged train-only character C=2 baseline.")
        if metadata["run_id"] != path.name or sha256(path / "config_source.yaml") != freeze["baseline_config_sha256"]:
            raise ValueError("The recreated run's ID or source configuration differs from its metadata.")
        source_manifest = manifest()
        if metadata["dataset"]["revision"] != source_manifest["revision"] != freeze["dataset_revision"]:
            raise ValueError("The recreated run used a different dataset revision.")
        for split in ("train", "dev"):
            expected = freeze[f"{split}_sha256"]
            source_entry = next(
                item for item in source_manifest["files"]
                if item["local_path"] == f"data/raw/amh/{split}.tsv"
            )
            if (source_entry["sha256"] != expected
                    or metadata["dataset"]["files"][split]["sha256"] != expected):
                raise ValueError(f"The recreated run used different {split} data.")
        source = "locally_recreated"
    model = joblib.load(path / "model.joblib")
    _check_pipeline(model, freeze["C"])
    return model, path.name, source


def load_experimental_model(report_path=ROOT / "reports/post_test_weighting.json"):
    """Load the separate post-test follow-up using its saved development report."""
    report_path = Path(report_path)
    report_path = (report_path if report_path.is_absolute() else ROOT / report_path).resolve()
    if not report_path.is_relative_to(ROOT.resolve()):
        raise ValueError("An experimental report must be inside this repository.")
    report = read_json(report_path)
    plan_path = ROOT / "configs/post_test_weighting.json"
    plan = read_json(plan_path)
    if (report["experiment_id"], report["plan_sha256"], report["model"],
            report["C"], report["preprocessing"]) != (
        plan["experiment_id"], sha256(plan_path), "char_tfidf", 2.0, "identity"
    ):
        raise ValueError("The experimental report does not match the declared study.")
    if (not report["original_test_result_known_before_experiment"]
            or report["original_test_split_loaded_for_this_study"]):
        raise ValueError("The experimental report has invalid test-use provenance.")
    if (report["dataset_revision"], report["train_sha256"], report["dev_sha256"]) != (
        plan["dataset_revision"], plan["train_sha256"], plan["dev_sha256"]
    ):
        raise ValueError("The experimental report used different data.")
    selected = min(report["candidate_results"], key=lambda item: (
        -round(item["mean_fold_macro_f1"], 6),
        plan["tie_order"].index(item["weight_name"]),
    ))
    if selected["weight_name"] != report["selected_weight_name"]:
        raise ValueError("The reported experimental selection differs from the fold scores.")
    path = (ROOT / report["selected_run_directory"]).resolve()
    if not path.is_relative_to((ROOT / "artifacts").resolve()) or path.name != report["selected_run_id"]:
        raise ValueError("The experimental model path is outside the expected artifact directory.")
    if not (path / "model.joblib").is_file():
        raise FileNotFoundError("The experimental model is local. Reproduce the study and pass its new report path.")
    metadata = read_json(path / "metadata.json")
    if (metadata["run_id"], metadata["model"], metadata["candidate_C"],
            metadata["train_split"], metadata["evaluation_split"], metadata["config_sha256"]) != (
        report["selected_run_id"], "char_tfidf", 2.0, "train", "dev",
        report["selected_config_sha256"]
    ) or metadata["test_split_loaded"] or not metadata["converged"]:
        raise ValueError("The experimental saved run differs from its report.")
    if metadata["dataset"]["revision"] != report["dataset_revision"]:
        raise ValueError("The experimental run used a different dataset revision.")
    if sha256(path / "config_source.yaml") != report["selected_config_sha256"]:
        raise ValueError("The experimental run used a different configuration.")
    if read_json(path / "metrics.json") != report["selected_development_metrics"]:
        raise ValueError("The experimental saved development metrics differ from its report.")
    for split in ("train", "dev"):
        if metadata["dataset"]["files"][split]["sha256"] != report[f"{split}_sha256"]:
            raise ValueError(f"The experimental run used different {split} data.")
    model = joblib.load(path / "model.joblib")
    _check_pipeline(model, report["C"])
    if model.named_steps["classifier"].class_weight != report["selected_full_train_class_weight"]:
        raise ValueError("The experimental classifier has different class weights.")
    return model, path.name, "post_test_exploratory"


def predict_texts(model, texts):
    """Run inference without fitting or changing the original input strings."""
    texts = list(texts)
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Provide at least one nonempty text; blank lines are not allowed.")
    labels = [str(value) for value in model.predict(texts)]
    if len(labels) != len(texts) or set(labels) - set(LABELS):
        raise ValueError("The saved model returned an unexpected number or label of predictions.")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text", action="append", metavar="AMHARIC", help="Repeat for multiple texts.")
    inputs.add_argument("--stdin", action="store_true", help="Read one text per nonblank input line.")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--run-dir", type=Path, help="Use a locally recreated baseline C=2 run under artifacts/.")
    models.add_argument(
        "--experimental-report", nargs="?", type=Path,
        const=ROOT / "reports/post_test_weighting.json",
        help="Use the exploratory weighted model named by a saved report (default: current study).",
    )
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args(argv)
    texts = sys.stdin.read().splitlines() if args.stdin else args.text
    if not texts or any(not text.strip() for text in texts):
        parser.error("Provide at least one nonblank text; blank input lines are not allowed.")
    try:
        if args.experimental_report is not None:
            model, run_id, source = load_experimental_model(args.experimental_report)
        else:
            model, run_id, source = load_model(args.run_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    labels = predict_texts(model, texts)
    if source == "post_test_exploratory" and args.format == "plain":
        print("Exploratory model: no independent test score for this revision.", file=sys.stderr)
    if args.format == "json":
        result = {
            "model_run_id": run_id,
            "model_source": source,
            "predictions": [{"index": index, "label": label} for index, label in enumerate(labels)],
        }
        if source == "post_test_exploratory":
            result["independent_test_score_available"] = False
        print(json.dumps(result, ensure_ascii=False))
    else:
        for label in labels:
            print(label)


if __name__ == "__main__":
    main()
