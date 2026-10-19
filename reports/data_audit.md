# Data audit

Source revision: `5aec3cfcf87b59f885e98641f72e52182776429d`.

Raw files remain unchanged. All files passed SHA-256, UTF-8, column, nonempty-text and label validation.

| Split | Rows | Negative | Neutral | Positive | Exact duplicate extra rows | NFC/whitespace duplicate extra rows |
|---|---:|---:|---:|---:|---:|---:|
| train | 5984 | 1548 | 3104 | 1332 | 1 | 2 |
| dev | 1497 | 388 | 776 | 333 | 0 | 0 |
| test | 1999 | 1337 | 224 | 438 | 0 | 0 |

Duplicate counts are extra rows beyond one occurrence per text, regardless of label.

## Cross-split overlap

Counts below refer to distinct shared texts, not pairs of rows.

| Pair | Exact shared texts | NFC/whitespace shared texts |
|---|---:|---:|
| train/dev | 0 | 0 |
| train/test | 0 | 0 |
| dev/test | 0 | 0 |

## Interpretation

Training and development are majority-neutral; test is majority-negative. Report macro-F1 and per-class metrics alongside accuracy.

Keep the released splits as the primary benchmark. If overlaps exist, document them and preregister a separate contamination sensitivity analysis before fitting models. Do not silently deduplicate or reshuffle the official files.

NFC/whitespace keys are used only for this integrity check. No model preprocessing or Ethiopic orthographic normalization has been selected.

The JSON companion includes label conflicts, replacement characters, length percentiles and affected-row counts. This audit does not assess semantic duplicates or annotation correctness.

Only structural aggregates were produced for test; no test predictions, model selection or test error inspection were performed.
