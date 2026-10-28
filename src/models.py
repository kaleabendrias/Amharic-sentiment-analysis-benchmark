"""Construct the implemented baselines and apply the declared word-model tie rule."""

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
    if model_name != "word_tfidf":
        raise ValueError(f"Model not implemented: {model_name}")
    if config["preprocessing"] != "identity" or config["word_tfidf"]["analyzer"] != "word":
        raise ValueError("This experiment requires unmodified text and word features.")
    classifier = config["classifier"]
    if classifier["estimator"] != "LogisticRegression" or C not in classifier["C_candidates"]:
        raise ValueError("Use logistic regression with a predefined C candidate.")
    features = dict(config["word_tfidf"])
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


def select_word_candidate(candidates, policy):
    """Highest rounded dev macro-F1, then lower C; no test score is accepted."""
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
