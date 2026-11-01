"""Exploratory post-test class-weight study using train-only grouped CV."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import warnings
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from src.data import LABELS, ROOT, load_split, manifest, overlap_key
from src.evaluate import evaluate_predictions
from src.models import build_model
from src.train import run_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_plan(plan_path=ROOT / "configs/post_test_weighting.json"):
    """Validate the predeclared settings without opening train, dev, or test TSVs."""
    plan = read_json(plan_path)
    if (plan["experiment_id"], plan["feature_model"], plan["C"],
            plan["preprocessing"], plan["candidate_weights"]) != (
        "post_test_class_weighting_v1", "char_tfidf", 2.0, "identity",
        ["none", "sqrt_inverse", "balanced"]
    ):
        raise ValueError("The declared model or candidate set changed.")
    if plan["selection_metric"] != "mean fold macro-F1, rounded to six decimals":
        raise ValueError("The declared selection measure changed.")
    if plan["tie_order"] != plan["candidate_weights"]:
        raise ValueError("The declared tie order changed.")
    cv = plan["cross_validation"]
    if (cv["splitter"], cv["n_splits"], cv["shuffle"], cv["random_state"]) != (
        "StratifiedGroupKFold", 5, True, 42
    ):
        raise ValueError("The declared fold protocol changed.")
    if sha256(ROOT / "configs/baselines.yaml") != plan["baseline_config_sha256"]:
        raise ValueError("The frozen baseline configuration changed.")
    if sha256(ROOT / "reports/final_test.json") != plan["original_final_test_report_sha256"]:
        raise ValueError("The prior final-test report changed.")
    source = manifest()
    if source["revision"] != plan["dataset_revision"]:
        raise ValueError("The pinned dataset revision changed.")
    for split in ("train", "dev"):
        entry = next(item for item in source["files"]
                     if item["local_path"] == f"data/raw/amh/{split}.tsv")
        if entry["sha256"] != plan[f"{split}_sha256"]:
            raise ValueError(f"The pinned {split} hash changed.")
    baseline = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text(encoding="utf-8"))
    if (baseline["preprocessing"], baseline["classifier"]["class_weight"],
            baseline["primary_metric"]) != ("identity", None, "macro_f1"):
        raise ValueError("The baseline configuration is not the expected unweighted model.")
    return plan, baseline


def sqrt_inverse_weights(labels):
    counts = Counter(labels)
    if set(counts) != set(LABELS):
        raise ValueError("Each training fold must contain every sentiment class.")
    total = len(labels)
    return {label: math.sqrt(total / (len(LABELS) * counts[label])) for label in LABELS}


def configured_model(baseline, C, weight_name, training_labels):
    config = deepcopy(baseline)
    if weight_name == "none":
        config["classifier"]["class_weight"] = None
    elif weight_name == "sqrt_inverse":
        config["classifier"]["class_weight"] = sqrt_inverse_weights(training_labels)
    elif weight_name == "balanced":
        config["classifier"]["class_weight"] = "balanced"
    else:
        raise ValueError(f"Unrecognized weight candidate: {weight_name}")
    return config, build_model(config, "char_tfidf", C)


def make_folds(texts, labels, plan):
    groups = [overlap_key(text) for text in texts]
    cv = plan["cross_validation"]
    splitter = StratifiedGroupKFold(
        n_splits=cv["n_splits"], shuffle=cv["shuffle"], random_state=cv["random_state"]
    )
    folds = list(splitter.split(texts, labels, groups))
    seen = set()
    for train_index, valid_index in folds:
        train_groups = {groups[i] for i in train_index}
        valid_groups = {groups[i] for i in valid_index}
        if train_groups & valid_groups:
            raise ValueError("A normalized duplicate crossed a CV fold boundary.")
        if seen & set(valid_index):
            raise ValueError("A training row was validated more than once.")
        seen.update(valid_index)
    if seen != set(range(len(texts))):
        raise ValueError("Cross-validation did not cover every training row.")
    return folds


def run(plan_path=ROOT / "configs/post_test_weighting.json",
        report_path=ROOT / "reports/post_test_weighting.json",
        artifact_root=ROOT / "artifacts"):
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError("This declared exploratory study already has a saved report.")
    plan, baseline = validate_plan(plan_path)
    rows = load_split("train")  # Original test data is never loaded in this follow-up.
    texts = [row["tweet"] for row in rows]
    labels = [row["label"] for row in rows]
    folds = make_folds(texts, labels, plan)
    run_root = Path(artifact_root) / f"post_test_weighting_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:6]}"
    run_root.mkdir(parents=True, exist_ok=False)
    candidates = []
    for name in plan["candidate_weights"]:
        oof = [None] * len(rows)
        fold_results = []
        for number, (train_index, valid_index) in enumerate(folds, start=1):
            fit_labels = [labels[i] for i in train_index]
            _, model = configured_model(baseline, plan["C"], name, fit_labels)
            started = perf_counter()
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                model.fit([texts[i] for i in train_index], fit_labels)
            fit_seconds = perf_counter() - started
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError(f"{name} fold {number} did not converge.")
            predictions = [str(value) for value in model.predict([texts[i] for i in valid_index])]
            for index, prediction in zip(valid_index, predictions, strict=True):
                oof[int(index)] = prediction
            metrics = evaluate_predictions([labels[i] for i in valid_index], predictions)
            fold_results.append({
                "fold": number, "training_rows": len(train_index),
                "validation_rows": len(valid_index),
                "training_label_counts": dict(Counter(fit_labels)),
                "metrics": metrics, "fit_seconds": fit_seconds,
                "classifier_iterations": model.named_steps["classifier"].n_iter_.tolist(),
                "warnings": [f"{item.category.__name__}: {item.message}" for item in caught],
            })
        if any(value is None for value in oof):
            raise ValueError("An out-of-fold prediction is missing.")
        with (run_root / f"{name}_oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["row_index_zero_based", "true_label", "predicted_label"])
            writer.writerows((i, label, prediction) for i, (label, prediction) in
                             enumerate(zip(labels, oof, strict=True)))
        fold_scores = [item["metrics"]["macro_f1"] for item in fold_results]
        candidates.append({
            "weight_name": name,
            "mean_fold_macro_f1": statistics.fmean(fold_scores),
            "fold_macro_f1_std": statistics.pstdev(fold_scores),
            "oof_metrics": evaluate_predictions(labels, oof),
            "folds": fold_results,
        })
    tie_order = plan["tie_order"]
    selected = min(candidates, key=lambda item: (
        -round(item["mean_fold_macro_f1"], 6), tie_order.index(item["weight_name"])
    ))
    selected_name = selected["weight_name"]
    selected_config, _ = configured_model(baseline, plan["C"], selected_name, labels)
    selected_config_path = run_root / "selected_config.yaml"
    selected_config_path.write_text(
        yaml.safe_dump(selected_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    saved_run, dev_metrics = run_model(
        config_path=selected_config_path, train_split="train", eval_split="dev",
        output_dir=run_root, model_name="char_tfidf", C=plan["C"],
    )
    metadata = read_json(saved_run / "metadata.json")
    if not metadata["converged"] or metadata["test_split_loaded"]:
        raise RuntimeError("The selected full-training run did not meet the declared protocol.")
    if metadata["dataset"]["files"]["train"]["sha256"] != plan["train_sha256"] or (
        metadata["dataset"]["files"]["dev"]["sha256"] != plan["dev_sha256"]
    ):
        raise ValueError("The selected run used different train or development data.")
    original = read_json(ROOT / "reports/final_test.json")["development_metrics"]
    report = {
        "experiment_id": plan["experiment_id"],
        "status": plan["status"],
        "plan_sha256": sha256(plan_path),
        "code_sha256": sha256(Path(__file__)),
        "original_test_result_known_before_experiment": True,
        "original_test_split_loaded_for_this_study": False,
        "selection_data": "five grouped folds of original train split",
        "follow_up_evaluation_data": "original development split, selected candidate only",
        "dataset_revision": plan["dataset_revision"],
        "train_sha256": plan["train_sha256"], "dev_sha256": plan["dev_sha256"],
        "model": plan["feature_model"], "C": plan["C"],
        "preprocessing": plan["preprocessing"],
        "candidate_results": candidates,
        "selected_weight_name": selected_name,
        "selected_full_train_class_weight": selected_config["classifier"]["class_weight"],
        "selected_config_sha256": sha256(selected_config_path),
        "selected_run_id": saved_run.name,
        "selected_run_directory": str(saved_run.relative_to(ROOT)),
        "original_development_metrics": original,
        "selected_development_metrics": dev_metrics,
        "development_macro_f1_delta": dev_metrics["macro_f1"] - original["macro_f1"],
        "claim_policy": plan["claim_policy"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/post_test_weighting.json")
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps({
        "selected_weight": result["selected_weight_name"],
        "selected_run_directory": result["selected_run_directory"],
        "development_macro_f1": result["selected_development_metrics"]["macro_f1"],
        "development_macro_f1_delta": result["development_macro_f1_delta"],
        "original_test_split_loaded_for_this_study": False,
    }, indent=2))


if __name__ == "__main__":
    main()
