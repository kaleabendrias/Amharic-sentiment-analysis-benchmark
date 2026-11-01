"""Predeclared, train-only grouped comparison of balanced linear text models."""

import argparse
import json
import statistics
import warnings
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from src.data import ROOT, load_split, manifest
from src.evaluate import evaluate_predictions
from src.models import build_model
from src.post_test_weighting import make_folds, read_json, sha256


PLAN_PATH = ROOT / "configs/post_test_hybrid.json"
REPORT_PATH = ROOT / "reports/post_test_hybrid.json"


def validate_plan(path=PLAN_PATH):
    plan = read_json(path)
    expected = [
        ("char_logreg", "char", "LogisticRegression", 2.0),
        ("hybrid_logreg", "word_char", "LogisticRegression", 2.0),
        ("char_svm", "char", "LinearSVC", 1.0),
        ("hybrid_svm", "word_char", "LinearSVC", 1.0),
    ]
    actual = [(c["name"], c["features"], c["estimator"], c["C"])
              for c in plan["candidates"]]
    if (plan["experiment_id"], plan["preprocessing"], plan["class_weight"], actual,
            plan["tie_order"], plan["selection_metric"]) != (
        "post_test_hybrid_v1", "identity", "balanced", expected,
        [item[0] for item in expected], "mean fold macro-F1, rounded to six decimals"
    ):
        raise ValueError("The predeclared candidate protocol changed.")
    cv = plan["cross_validation"]
    if (cv["splitter"], cv["n_splits"], cv["shuffle"], cv["random_state"]) != (
        "StratifiedGroupKFold", 5, True, 42
    ):
        raise ValueError("The predeclared fold protocol changed.")
    for filename, key in [
        ("configs/baselines.yaml", "baseline_config_sha256"),
        ("reports/final_test.json", "original_final_test_report_sha256"),
        ("reports/post_test_weighting.json", "prior_weighting_report_sha256"),
    ]:
        if sha256(ROOT / filename) != plan[key]:
            raise ValueError(f"Prior file changed: {filename}")
    source = manifest()
    if source["revision"] != plan["dataset_revision"]:
        raise ValueError("Dataset revision changed.")
    for split in ("train", "dev"):
        entry = next(item for item in source["files"]
                     if item["local_path"] == f"data/raw/amh/{split}.tsv")
        if entry["sha256"] != plan[f"{split}_sha256"]:
            raise ValueError(f"Pinned {split} checksum changed.")
    baseline = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text(encoding="utf-8"))
    if baseline["preprocessing"] != "identity" or baseline["seed"] != 42:
        raise ValueError("Baseline feature settings changed.")
    return plan, baseline


def build_candidate(spec, baseline):
    """Build a fresh pipeline so vocabulary fitting happens within each fold."""
    config = deepcopy(baseline)
    config["classifier"]["class_weight"] = "balanced"
    if spec["features"] == "char" and spec["estimator"] == "LogisticRegression":
        return build_model(config, "char_tfidf", spec["C"])

    def vectorizer(name):
        settings = dict(config[f"{name}_tfidf"])
        settings["ngram_range"] = tuple(settings["ngram_range"])
        return TfidfVectorizer(**settings)

    if spec["features"] == "char":
        features = vectorizer("char")
    elif spec["features"] == "word_char":
        features = FeatureUnion([("word", vectorizer("word")),
                                 ("char", vectorizer("char"))])
    else:
        raise ValueError("Unknown feature family.")
    if spec["estimator"] == "LinearSVC":
        classifier = LinearSVC(C=spec["C"], class_weight="balanced",
                               max_iter=10000, random_state=42)
    elif spec["estimator"] == "LogisticRegression":
        classifier = build_model(config, "char_tfidf", spec["C"]).named_steps["classifier"]
    else:
        raise ValueError("Unknown classifier.")
    return Pipeline([("features", features), ("classifier", classifier)])


def fit_checked(model, texts, labels):
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(texts, labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("A candidate failed to converge; results cannot be selected.")
    return perf_counter() - started


def run(plan_path=PLAN_PATH, report_path=REPORT_PATH, artifact_root=ROOT / "artifacts"):
    report_path = Path(report_path)
    if report_path.exists():
        raise FileExistsError("The exploratory study already has a report.")
    plan, baseline = validate_plan(plan_path)
    train = load_split("train")
    texts = [row["tweet"] for row in train]
    labels = [row["label"] for row in train]
    folds = make_folds(texts, labels, plan)
    results = []
    for spec in plan["candidates"]:
        oof = [None] * len(train)
        scores = []
        fold_details = []
        for number, (fit_idx, valid_idx) in enumerate(folds, 1):
            model = build_candidate(spec, baseline)
            seconds = fit_checked(model, [texts[i] for i in fit_idx],
                                  [labels[i] for i in fit_idx])
            predictions = model.predict([texts[i] for i in valid_idx])
            for index, prediction in zip(valid_idx, predictions, strict=True):
                oof[int(index)] = str(prediction)
            metrics = evaluate_predictions([labels[i] for i in valid_idx], predictions)
            scores.append(metrics["macro_f1"])
            fold_details.append({"fold": number, "training_rows": len(fit_idx),
                                 "validation_rows": len(valid_idx),
                                 "macro_f1": metrics["macro_f1"],
                                 "accuracy": metrics["accuracy"],
                                 "fit_seconds": seconds})
        if any(prediction is None for prediction in oof):
            raise ValueError("An out-of-fold prediction is missing.")
        results.append({"name": spec["name"],
                        "mean_fold_macro_f1": statistics.fmean(scores),
                        "fold_macro_f1_std": statistics.pstdev(scores),
                        "oof_metrics": evaluate_predictions(labels, oof),
                        "folds": fold_details})
    winner = min(results, key=lambda item: (
        -round(item["mean_fold_macro_f1"], 6),
        plan["tie_order"].index(item["name"])))
    selected = next(spec for spec in plan["candidates"] if spec["name"] == winner["name"])
    model = build_candidate(selected, baseline)
    full_fit_seconds = fit_checked(model, texts, labels)
    dev = load_split("dev")
    dev_metrics = evaluate_predictions([row["label"] for row in dev],
                                       model.predict([row["tweet"] for row in dev]))
    artifact_dir = Path(artifact_root) / (
        "post_test_hybrid_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_" + uuid4().hex[:6])
    artifact_dir.mkdir(parents=True, exist_ok=False)
    model_path = artifact_dir / "model.joblib"
    joblib.dump(model, model_path)
    prior = read_json(ROOT / "reports/post_test_weighting.json")
    report = {
        "experiment_id": plan["experiment_id"], "status": plan["status"],
        "plan_sha256": sha256(plan_path), "code_sha256": sha256(Path(__file__)),
        "dataset_revision": plan["dataset_revision"],
        "train_sha256": plan["train_sha256"], "dev_sha256": plan["dev_sha256"],
        "original_test_result_known_before_experiment": True,
        "original_test_split_loaded_for_this_study": False,
        "selection_data": "five grouped folds of original train split",
        "follow_up_evaluation_data": "original development split, selected candidate only",
        "candidate_results": results, "selected_candidate": selected,
        "selected_run_directory": str(artifact_dir.relative_to(ROOT)),
        "selected_model_sha256": sha256(model_path),
        "selected_model_bytes": model_path.stat().st_size,
        "full_train_fit_seconds": full_fit_seconds,
        "selected_development_metrics": dev_metrics,
        "prior_balanced_cv_macro_f1": next(
            item["mean_fold_macro_f1"] for item in prior["candidate_results"]
            if item["weight_name"] == "balanced"),
        "prior_balanced_dev_macro_f1": prior["selected_development_metrics"]["macro_f1"],
        "development_macro_f1_delta_vs_prior": (
            dev_metrics["macro_f1"] - prior["selected_development_metrics"]["macro_f1"]),
        "claim_policy": plan["claim_policy"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    report = run(args.plan, args.report)
    print(json.dumps({"selected_candidate": report["selected_candidate"],
                      "development_macro_f1": report["selected_development_metrics"]["macro_f1"],
                      "development_macro_f1_delta_vs_prior": report["development_macro_f1_delta_vs_prior"]},
                     indent=2))


if __name__ == "__main__":
    main()
