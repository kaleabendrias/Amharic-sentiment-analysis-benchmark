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
    parser.add_argument("--run-dir", type=Path, help="Use a locally recreated C=2 run under artifacts/.")
    parser.add_argument("--format", choices=("plain", "json"), default="plain")
    args = parser.parse_args(argv)
    texts = sys.stdin.read().splitlines() if args.stdin else args.text
    if not texts or any(not text.strip() for text in texts):
        parser.error("Provide at least one nonblank text; blank input lines are not allowed.")
    try:
        model, run_id, source = load_model(args.run_dir)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    labels = predict_texts(model, texts)
    if args.format == "json":
        print(json.dumps({
            "model_run_id": run_id,
            "model_source": source,
            "predictions": [{"index": index, "label": label} for index, label in enumerate(labels)],
        }, ensure_ascii=False))
    else:
        for label in labels:
            print(label)


if __name__ == "__main__":
    main()
