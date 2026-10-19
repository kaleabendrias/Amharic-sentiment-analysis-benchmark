"""Produce aggregate integrity reports without publishing tweet text or training models."""

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations

from src.data import LABELS, ROOT, SPLITS, load_split, manifest, overlap_key


def summarize(rows):
    texts = [r["tweet"] for r in rows]
    exact = Counter(texts)
    normalized = Counter(map(overlap_key, texts))
    labels_by_text = defaultdict(set)
    for row in rows:
        labels_by_text[overlap_key(row["tweet"])].add(row["label"])
    lengths = [len(text) for text in texts]
    return {
        "rows": len(rows),
        "labels": {label: sum(r["label"] == label for r in rows) for label in LABELS},
        "empty_texts": sum(not text.strip() for text in texts),
        "unicode_replacement_character_rows": sum("\ufffd" in text for text in texts),
        "exact_duplicate_extra_rows": sum(n - 1 for n in exact.values()),
        "normalized_duplicate_extra_rows": sum(n - 1 for n in normalized.values()),
        "normalized_texts_with_conflicting_labels": sum(len(s) > 1 for s in labels_by_text.values()),
        "character_lengths": {
            "min": min(lengths),
            "median": statistics.median(lengths),
            "p95": statistics.quantiles(lengths, n=100, method="inclusive")[94] if len(lengths) > 1 else lengths[0],
            "max": max(lengths),
        },
    }


def main():
    splits = {name: load_split(name) for name in SPLITS}
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": manifest()["revision"],
        "raw_files_modified": False,
        "normalization_for_overlap_audit_only": "Unicode NFC and collapsed whitespace",
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "cross_split_overlap": {},
    }
    for a, b in combinations(SPLITS, 2):
        left, right = splits[a], splits[b]
        exact = {r["tweet"] for r in left} & {r["tweet"] for r in right}
        shared = {overlap_key(r["tweet"]) for r in left} & {overlap_key(r["tweet"]) for r in right}
        report["cross_split_overlap"][f"{a}/{b}"] = {
            "exact_shared_unique_texts": len(exact),
            "normalized_shared_unique_texts": len(shared),
            f"{a}_rows_affected": sum(overlap_key(r["tweet"]) in shared for r in left),
            f"{b}_rows_affected": sum(overlap_key(r["tweet"]) in shared for r in right),
        }
    directory = ROOT / "reports"
    directory.mkdir(exist_ok=True)
    (directory / "data_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Data audit", "", f"Source revision: `{report['source_revision']}`.", "",
        "Raw files remain unchanged. All files passed SHA-256, UTF-8, column, nonempty-text and label validation.", "",
        "| Split | Rows | Negative | Neutral | Positive | Exact duplicate extra rows | NFC/whitespace duplicate extra rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in report["splits"].items():
        counts = stats["labels"]
        lines.append(f"| {name} | {stats['rows']} | {counts['negative']} | {counts['neutral']} | {counts['positive']} | {stats['exact_duplicate_extra_rows']} | {stats['normalized_duplicate_extra_rows']} |")
    lines += ["", "Duplicate counts are extra rows beyond one occurrence per text, regardless of label.", "",
              "## Cross-split overlap", "", "Counts below refer to distinct shared texts, not pairs of rows.", "",
              "| Pair | Exact shared texts | NFC/whitespace shared texts |", "|---|---:|---:|"]
    for pair, counts in report["cross_split_overlap"].items():
        lines.append(f"| {pair} | {counts['exact_shared_unique_texts']} | {counts['normalized_shared_unique_texts']} |")
    lines += ["", "## Interpretation", "",
              "Training and development are majority-neutral; test is majority-negative. Report macro-F1 and per-class metrics alongside accuracy.", "",
              "Keep the released splits as the primary benchmark. If overlaps exist, document them and preregister a separate contamination sensitivity analysis before fitting models. Do not silently deduplicate or reshuffle the official files.", "",
              "NFC/whitespace keys are used only for this integrity check. No model preprocessing or Ethiopic orthographic normalization has been selected.", "",
              "The JSON companion includes label conflicts, replacement characters, length percentiles and affected-row counts. This audit does not assess semantic duplicates or annotation correctness.", "",
              "Only structural aggregates were produced for test; no test predictions, model selection or test error inspection were performed.", ""]
    (directory / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
