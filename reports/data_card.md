# Data card: AfriSenti Amharic

Prepared 19 October 2026. This describes the exact files downloaded for this project; it does not claim new data collection or annotation.

## Source and version

- Creators' repository: [AfriSenti-SemEval 2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023).
- Pinned revision: `5aec3cfcf87b59f885e98641f72e52182776429d`.
- Selected files: [`data/amh/train.tsv`, `dev.tsv`, `test.tsv`](https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/5aec3cfcf87b59f885e98641f72e52182776429d/data/amh).
- Language identifier: `amh` in this directory. Other upstream locations use different abbreviations; the downloader uses these explicit paths.
- Provenance: [configs/dataset.json](../configs/dataset.json) records byte lengths, SHA-256 digests and download URLs via source path + revision. A local copy of the upstream README is retained under ignored `data/source/`.
- Upstream license statement: [CC BY 4.0 in the pinned README](https://github.com/afrisenti-semeval/afrisent-semeval-2023/blob/5aec3cfcf87b59f885e98641f72e52182776429d/README.md#license). Attribute the creators and identify changes. See [CITATIONS.md](../CITATIONS.md) for the requested papers.

## Schema and labels

The files are UTF-8 tab-separated tables with exactly `tweet` and `label` columns. `tweet` is the original social-media text; `label` is one of the source's three categories: `negative`, `neutral`, `positive`. These are the existing annotation labels, not labels inferred by a model. For annotation methodology and borderline cases, consult the dataset publications; no new interpretation of their guidelines is imposed here.

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 1,548 | 3,104 | 1,332 | 5,984 |
| Development | 388 | 776 | 333 | 1,497 |
| Test | 1,337 | 224 | 438 | 1,999 |

## Integrity and processing

The loader verifies each file hash before use, validates columns and labels, rejects empty/whitespace-only text, and preserves the original string. The initial audit found no replacement characters and no within-split label conflicts for NFC/whitespace-equivalent texts. One exact duplicate extra row exists within training; two extra rows exist after NFC/whitespace comparison. Development and test have no duplicate extra rows under those checks. All three split pairs have zero shared texts under exact and NFC/whitespace comparisons.

No rows were deleted, labels remapped, splits reshuffled, or model preprocessing applied. The overlap audit uses Unicode NFC plus collapsed whitespace only to construct comparison keys. It does not establish that there are no paraphrases, shared users, thread relationships or near duplicates. See the generated [audit](data_audit.md) and [JSON details](data_audit.json).

## Evaluation boundaries

Fit vocabulary, IDF and classifier parameters on train only. Select hyperparameters and conduct error analysis on development. Reserve test for evaluation after model choices are frozen. The setup audit used only structural test aggregates and did not inspect test mistakes or fit a classifier.

The pronounced class-distribution difference between train/dev and test is an observed property, not an explanation of its cause. Report macro-F1, weighted-F1, per-class metrics and accuracy. Preserve the official files for the primary benchmark; any deduplicated or rebalanced alternative must be separately named and documented.

## Scope and limitations

This is a social-media sentiment benchmark. Performance on it does not establish performance on news, customer reviews, regional varieties or other Amharic domains. Short messages, context dependence and label ambiguity may affect predictions; investigate these using development data rather than asserting findings in advance. Native-language review should inform any later orthographic normalization.

Keep raw text and row-level predictions out of Git by default. The committed audit contains aggregates, not copied tweets. The first deliverable is a reproducible offline classification study; a future model card must describe the actual model and measured limitations.
