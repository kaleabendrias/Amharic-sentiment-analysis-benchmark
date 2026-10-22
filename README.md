# Amharic Sentiment Analysis Benchmark

Study how word and character TF-IDF classifiers compare on Amharic sentiment analysis, then investigate their errors and resource costs. The CPU baseline is the first milestone; a pretrained transformer is an optional later comparison.

**Current stage:** groundwork and first train/development data inspection complete. All 24 environment packages are installed and verified. The majority-class baseline is the next small task. Start with [explanation.md](explanation.md) for the beginner lesson from 21 October 2026, or [NEXT_STEPS.md](NEXT_STEPS.md) for the implementation checklist.

## Dataset

Uses the Amharic `data/amh` distribution from the creators' [AfriSenti repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023/tree/5aec3cfcf87b59f885e98641f72e52182776429d/data/amh). It contains existing sentiment annotations, not new annotations collected for this project.

| Split | Examples | Use |
|---|---:|---|
| Train | 5,984 | Fit features and classifiers |
| Development | 1,497 | Select hyperparameters and analyze errors |
| Test | 1,999 | Final evaluation after choices are frozen |
| Total | 9,480 | Three sentiment classes |

Labels are `negative`, `neutral`, `positive`. The test set is predominantly negative while train/dev are predominantly neutral. The audit found no exact or NFC/whitespace text overlap between splits. Training contains one extra exact duplicate row and two extra rows under NFC/whitespace comparison. Raw files remain unchanged.

See [data card](reports/data_card.md), [audit](reports/data_audit.md), [source manifest](configs/dataset.json) and [attribution](CITATIONS.md). The upstream README declares CC BY 4.0. Keep attribution and identify modifications when releasing derivative artifacts.

## Setup and verification

Tested on Linux with Python 3.12.3. The current workspace already has `.venv` and the downloaded data. To recreate the setup from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.download_data
python -m scripts.audit_data
python -m scripts.check_environment
python -m unittest discover -s tests -v
python -m pip check
```

`requirements.in` lists direct dependencies; `requirements.txt` pins the tested versions, including transitive dependencies. The environment check imports the installed libraries, verifies Ethiopic tokenization and emoji handling, and exercises plotting without training a model. Its output is saved to [reports/environment.json](reports/environment.json).

The downloader checks existing files and fetches only missing files from the pinned revision. A checksum mismatch stops execution; move the changed file aside and rerun the command to restore the original. Network failures are retried up to three times. No account, API key, tweet hydration or GPU is needed for the downloaded files and CPU stage.

## Files available now

```text
configs/dataset.json         upstream revision, source paths and SHA-256 hashes
configs/baselines.yaml       next-stage experiment specification, not a trainer
src/data.py                 strict UTF-8 TSV loader and integrity checks
scripts/download_data.py    reproducible download/verification command
scripts/audit_data.py       class balance, duplicates, overlaps and length audit
scripts/check_environment.py dependency, tokenization and plotting checks
scripts/inspect_data.py     small label-balanced train/dev inspection
tests/test_data.py          corruption, malformed-input and text-preservation tests
reports/                   aggregate data and environment reports
data/raw/amh/               original train/dev/test TSVs; ignored by Git
data/source/README.md       original upstream README; ignored by Git
```

## Modeling protocol

Compare the training majority class, word TF-IDF + logistic regression, and character TF-IDF + logistic regression. Fit vectorizers inside a training-only scikit-learn pipeline; tune the small predefined grid on development data. The initial feature and classifier choices are recorded in [configs/baselines.yaml](configs/baselines.yaml). The future trainer must read this configuration and record its resolved values with each run.

Use macro-F1 as the primary project metric, plus per-class precision/recall/F1, weighted-F1 and accuracy. Macro-F1 and weighted-F1 are different measures: use a matching protocol when comparing with published work. Freeze the selected configuration before final test evaluation.

Protocol confirmed on 21 October 2026, before any classifiers were fitted: use the pinned dataset revision above, retain original rows and labels, and seed sampling/learned estimators with 42 where supported. Compare the majority reference and the word/character settings in `configs/baselines.yaml`, with logistic-regression `C` candidates 0.5, 1.0 and 2.0. Rank candidates by development macro-F1 rounded to six decimal places. For ties, use the fixed order majority, word TF-IDF, character TF-IDF, then smaller `C` within a model family. This rule makes selection deterministic without consulting test results. The future trainer must implement these configuration fields.

The first [inspection report](reports/data_inspection.md) covers 18 train/development examples. Rerun it with `python -m scripts.inspect_data`; raw sampled messages and tokens stay in ignored `artifacts/data_inspection/samples.json`.

The word analyzer retains Unicode word characters, including Ethiopic text, numbers, underscores and single-character words. Punctuation such as slashes and exclamation marks, and symbols such as emoji, are excluded from its word tokens. The character analyzer can retain punctuation and emoji. Neither uses an English stopword list or stemmer. No Ethiopic character folding has been chosen. Consult the [official scikit-learn text-feature guide](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) when implementing the pipelines.

Only aggregate structure was audited for the test split. There are no measured model results yet. For the next session, begin with the majority baseline and development-set evaluation in [NEXT_STEPS.md](NEXT_STEPS.md).
