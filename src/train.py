"""Fit majority or word TF-IDF baselines on train and evaluate development."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import yaml
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from src.data import LABELS, ROOT, load_split, manifest
from src.evaluate import evaluate_model
from src.models import build_model, model_parameters, select_word_candidate


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def code_provenance(run_dir):
    """Record Git state plus actual source snapshots, including uncommitted code."""
    def git(*arguments):
        try:
            result = subprocess.run(
                ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False,
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    hashes = {}
    for relative in (
        "src/__init__.py", "src/data.py", "src/train.py", "src/evaluate.py", "src/models.py",
        "configs/dataset.json", "requirements.txt",
    ):
        content = (ROOT / relative).read_bytes()
        destination = run_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "working_tree_dirty": None if status is None else bool(status),
        "working_tree_status": status,
        "source_sha256": hashes,
        "snapshot_directory": "source_snapshot",
    }


def run_model(config_path=ROOT / "configs/baselines.yaml", train_split="train",
              eval_split="dev", output_dir=ROOT / "artifacts", model_name="majority", C=None):
    # Validate boundaries before loading any dataset or creating output files.
    if train_split != "train" or eval_split != "dev":
        raise ValueError("This step fits train only and evaluates development only.")
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["train_split"] != train_split or config["selection_split"] != eval_split:
        raise ValueError("Requested splits do not match the experiment configuration.")
    if config["labels"] != list(LABELS):
        raise ValueError("Configuration must preserve the project's fixed label order.")
    model = build_model(config, model_name, C)

    train_rows = load_split(train_split)
    dev_rows = load_split(eval_split)
    started = perf_counter()
    # Pipeline.fit learns both vocabulary/IDF and classifier from training only.
    # DummyClassifier accepts the same texts but ignores their feature values.
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        model.fit([row["tweet"] for row in train_rows], [row["label"] for row in train_rows])
    fit_seconds = perf_counter() - started
    started = perf_counter()
    metrics, predictions = evaluate_model(model, dev_rows, config["labels"])
    evaluation_seconds = perf_counter() - started

    timestamp = datetime.now(timezone.utc)
    run_id = f"{model_name}_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:6]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, run_dir / "model.joblib")
    save_json(run_dir / "metrics.json", metrics)
    resolved_config = dict(config)
    resolved_config["run"] = {
        "model": model_name, "train_split": train_split, "evaluation_split": eval_split,
        "model_parameters": model_parameters(model),
    }
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8",
    )
    (run_dir / "config_source.yaml").write_bytes(config_bytes)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index_zero_based", "true_label", "predicted_label"])
        for index, (row, predicted) in enumerate(zip(dev_rows, predictions, strict=True)):
            writer.writerow([index, row["label"], predicted])

    source = manifest()
    used_files = {
        split: next(f for f in source["files"] if f["local_path"] == f"data/raw/amh/{split}.tsv")
        for split in (train_split, eval_split)
    }
    metadata = {
        "run_id": run_id,
        "created_at_utc": timestamp.isoformat(),
        "model": model_name,
        "model_parameters": model_parameters(model),
        "candidate_C": C,
        "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_) if model_name == "word_tfidf" else None,
        "classifier_iterations": model.named_steps["classifier"].n_iter_.tolist() if model_name == "word_tfidf" else None,
        "training_warnings": [{"category": warning.category.__name__, "message": str(warning.message)} for warning in caught],
        "converged": not any(issubclass(warning.category, ConvergenceWarning) for warning in caught),
        "fit_thread_limit": 1,
        "seed": config["seed"],
        "train_split": train_split,
        "evaluation_split": eval_split,
        "labels": config["labels"],
        "training_label_counts": dict(Counter(row["label"] for row in train_rows)),
        "development_label_counts": dict(Counter(row["label"] for row in dev_rows)),
        "prediction_counts": dict(Counter(predictions)),
        "fit_seconds": fit_seconds,
        "evaluation_seconds": evaluation_seconds,
        "timing_scope": "Fit excludes data loading; evaluation includes prediction and metrics. Neither is a latency benchmark.",
        "dataset": {"repository": source["repository"], "revision": source["revision"], "files": used_files},
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted(
            (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        )),
        "code": code_provenance(run_dir),
        "test_split_loaded": False,
    }
    save_json(run_dir / "metadata.json", metadata)
    return run_dir, metrics


def run_majority(config_path=ROOT / "configs/baselines.yaml", train_split="train",
                 eval_split="dev", output_dir=ROOT / "artifacts"):
    return run_model(config_path, train_split, eval_split, output_dir, model_name="majority")


def run_word_grid(config_path=ROOT / "configs/baselines.yaml", train_split="train",
                  eval_split="dev", output_dir=ROOT / "artifacts"):
    if train_split != "train" or eval_split != "dev":
        raise ValueError("The word experiment fits train only and evaluates development only.")
    config_bytes = Path(config_path).read_bytes()
    config = yaml.safe_load(config_bytes)
    if config["primary_metric"] != "macro_f1":
        raise ValueError("The predefined selection metric is development macro-F1.")
    values = config["classifier"]["C_candidates"]
    if not values or any(not isinstance(C, (int, float)) or not math.isfinite(C) or C <= 0 for C in values):
        raise ValueError("C candidates must be finite positive numbers.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate C candidates are not allowed.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    grid_dir = Path(output_dir) / f"word_grid_{timestamp}_{uuid4().hex[:6]}"
    # Validate settings and split roles before creating the grid directory.
    if config["train_split"] != train_split or config["selection_split"] != eval_split:
        raise ValueError("Requested splits do not match the experiment configuration.")
    build_model(config, "word_tfidf", values[0])
    grid_dir.mkdir(parents=True, exist_ok=False)
    candidates = []
    for C in sorted(values):
        run_dir, metrics = run_model(config_path, train_split, eval_split, grid_dir, "word_tfidf", C)
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        if metadata["config_sha256"] != hashlib.sha256(config_bytes).hexdigest():
            raise ValueError("Configuration changed during the experiment; selection stopped.")
        if not metadata["converged"]:
            raise RuntimeError(f"C={C} did not converge. Inspect {run_dir}/metadata.json before selection.")
        candidates.append({
            "C": C, "run_id": run_dir.name, "metrics": metrics,
            "fit_seconds": metadata["fit_seconds"], "vocabulary_size": metadata["vocabulary_size"],
            "classifier_iterations": metadata["classifier_iterations"],
            "training_warnings": metadata["training_warnings"],
        })
    selected = select_word_candidate(candidates, config["selection"])
    summary = {
        "model": "word_tfidf", "train_split": train_split, "evaluation_split": eval_split,
        "primary_metric": "macro_f1", "selection_policy": config["selection"],
        "selection_scope": "Best word candidate only; character and cross-family selection are later steps.",
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "candidates": candidates, "selected_C": selected["C"],
        "selected_run_id": selected["run_id"], "test_split_loaded": False,
    }
    save_json(grid_dir / "selection.json", summary)
    return grid_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["majority", "word_tfidf"], default="majority")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines.yaml")
    parser.add_argument("--train-split", choices=["train"], default="train")
    parser.add_argument("--eval-split", choices=["dev"], default="dev")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.model == "majority":
        run_dir, metrics = run_majority(args.config, args.train_split, args.eval_split, args.output_dir)
        print(json.dumps({"run_directory": str(run_dir), "metrics": metrics}, indent=2))
    else:
        grid_dir, selection = run_word_grid(args.config, args.train_split, args.eval_split, args.output_dir)
        print(json.dumps({"grid_directory": str(grid_dir), "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
