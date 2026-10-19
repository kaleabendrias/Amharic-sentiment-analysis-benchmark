"""Check installed tools and Ethiopic text features; no dataset models are fitted."""

import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy
import pandas
import requests
import scipy
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import ROOT, overlap_key


def main():
    sample = "እወደዋለሁ ጥሩ ነው 😊"
    config = yaml.safe_load((ROOT / "configs/baselines.yaml").read_text(encoding="utf-8"))
    word_config = dict(config["word_tfidf"])
    char_config = dict(config["char_tfidf"])
    word_config["ngram_range"] = tuple(word_config["ngram_range"])
    char_config["ngram_range"] = tuple(char_config["ngram_range"])
    vectorizer = TfidfVectorizer(**word_config)
    tokens = vectorizer.build_analyzer()(sample)
    if "እወደዋለሁ" not in tokens or "ጥሩ" not in tokens:
        raise RuntimeError("Ethiopic word tokenization failed")
    if not any("😊" in feature for feature in TfidfVectorizer(**char_config).build_analyzer()(sample)):
        raise RuntimeError("Character analyzer lost emoji")
    if overlap_key("  ጥሩ\nነው  ") != "ጥሩ ነው":
        raise RuntimeError("Whitespace normalization failed")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Exercise plotting without creating a fake result chart.
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.canvas.draw()
    plt.close(figure)
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "joblib", "PyYAML", "requests")
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "system_ram_gib": round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1),
        "isolated_environment": sys.prefix != sys.base_prefix,
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "checks": {"ethiopic_word_tokens": "passed", "emoji_character_analyzer": "passed", "headless_plotting": "passed"},
        "dataset_training_performed": False,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/environment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
