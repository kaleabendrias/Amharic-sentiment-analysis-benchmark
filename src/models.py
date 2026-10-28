"""Construct baselines and apply the declared development-only selection rules."""

import json
import math

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(config, model_name, C=None):
    if model_name == "majority":
        if config["majority"] != {"strategy": "most_frequent"}:
            raise ValueError("The majority reference must use most_frequent.")
        return DummyClassifier(strategy="most_frequent", random_state=config["seed"])
    if model_name not in {"word_tfidf", "char_tfidf"}:
        raise ValueError(f"Model not implemented: {model_name}")
    expected_analyzer = "word" if model_name == "word_tfidf" else "char"
    if config["preprocessing"] != "identity" or config[model_name]["analyzer"] != expected_analyzer:
        raise ValueError(f"This experiment requires unmodified text and {expected_analyzer} features.")
    classifier = config["classifier"]
    if classifier["estimator"] != "LogisticRegression" or C not in classifier["C_candidates"]:
        raise ValueError("Use logistic regression with a predefined C candidate.")
    features = dict(config[model_name])
    features["ngram_range"] = tuple(features["ngram_range"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**features)),
        ("classifier", LogisticRegression(
            C=C, solver=classifier["solver"], max_iter=classifier["max_iter"],
            class_weight=classifier["class_weight"], random_state=config["seed"],
        )),
    ])


def model_parameters(model):
    """Include actual defaults while converting parameter types to JSON-safe values."""
    if isinstance(model, Pipeline):
        parameters = {
            name: {"estimator": type(step).__name__, "parameters": step.get_params(deep=False)}
            for name, step in model.steps
        }
    else:
        parameters = model.get_params(deep=False)

    def encode(value):
        if isinstance(value, type):
            return f"{value.__module__}.{value.__qualname__}"
        raise TypeError(f"Cannot record model parameter: {type(value).__name__}")

    return json.loads(json.dumps(parameters, default=encode))


def select_tfidf_candidate(candidates, policy):
    """Highest rounded dev macro-F1, then lower C, for either TF-IDF family."""
    if not candidates or policy["tied_C_order"] != "ascending":
        raise ValueError("Need candidates and the predefined ascending-C tie rule.")
    decimals = policy["score_rounding_decimals"]
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("Score rounding must use a nonnegative integer.")
    for candidate in candidates:
        if not math.isfinite(candidate["metrics"]["macro_f1"]):
            raise ValueError("Cannot select a candidate with a non-finite score.")
    return min(candidates, key=lambda candidate: (
        -round(candidate["metrics"]["macro_f1"], decimals), candidate["C"],
    ))


def select_family_candidate(candidates, policy):
    """Choose one family winner by rounded development macro-F1 and declared order."""
    order = policy["tied_model_order"]
    if not candidates or len(order) != len(set(order)):
        raise ValueError("Need candidates and a unique model tie order.")
    if len(candidates) != len(order) or {item["model"] for item in candidates} != set(order):
        raise ValueError("Compare exactly one selected candidate from each declared family.")
    decimals = policy["score_rounding_decimals"]
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("Score rounding must use a nonnegative integer.")
    for candidate in candidates:
        if not math.isfinite(candidate["metrics"]["macro_f1"]):
            raise ValueError("Cannot select a candidate with a non-finite score.")
    return min(candidates, key=lambda candidate: (
        -round(candidate["metrics"]["macro_f1"], decimals), order.index(candidate["model"]),
    ))
