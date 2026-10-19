# Amharic Sentiment Analysis Benchmark

**Status:** Groundwork complete: data downloaded and audited, dependencies installed and verified. CPU model implementation and evaluation are next.  
**Prepared:** 19 October 2026.  
**Purpose:** Build a reproducible study of Amharic sentiment classification that demonstrates sound NLP experiments, language-aware error analysis, and practical model tradeoffs.

The dataset and setup utilities are now implemented in this folder. See [NEXT_STEPS.md](NEXT_STEPS.md) for the current handoff and [README.md](README.md) for reproducible setup commands. The model experiments below are future implementation tasks. Do not claim results until the corresponding experiments run successfully.

## What makes this project worthwhile

Answer a specific question: **How much predictive quality do inexpensive character-based models provide for Amharic sentiment, and when does a larger pretrained model justify its resource cost?**

The distinctive work is a controlled comparison, an Amharic-specific error analysis, and reproducible artifacts. A model that fails to improve on a strong baseline can still produce an informative study. The deliverable is a benchmark and report, not another generic chatbot or an assumed state-of-the-art result.

## Feasible scope and resources

| Stage | Deliverable | Resources | Planning estimate |
|---|---|---|---|
| A: Minimum complete study | Data audit; majority, word-TF-IDF, and character-TF-IDF baselines; evaluation; short report | Ordinary CPU laptop, Python, scikit-learn | About 1–2 focused days after dataset access |
| B: Strong portfolio version | Controlled preprocessing ablation; saved error analysis; inference demo; reproducible release | CPU laptop | About 1–2 additional days |
| C: Optional research extension | One transformer comparison; quality/latency/model-size analysis; repeated runs if feasible | GPU preferred; optional hosted notebook | About 2–4 additional days, subject to compute |

Do not make Stage A depend on a GPU, paid API, large new data collection, translation service, or deployment account. Defer transformer training if resources are unavailable. These estimates depend on data availability and prior setup; they are not deadlines.

## Data and access

Use the **Amharic subset of AfriSenti**. The creators provide data and task resources through their [official repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023). Follow the selected files' usage conditions and requested citations. Record the exact source revision, language identifier used by the chosen distribution, and file hashes. Do not imply that you collected or annotated this benchmark.

Before modeling:

- [x] Verify actual train/development/test file availability and labels. The selected `data/amh` files include all three labeled splits.
- [x] Preserve the released split protocol; no custom held-out split is needed.
- [x] Record per-split sample counts, missing/empty texts, class balance, label mappings, and text-length distributions.
- [x] Detect exact duplicate texts and audit cross-split overlap. Original rows remain unchanged; see `reports/data_audit.md`.
- [x] Keep original raw files ignored by Git and provide a download command pinned to the source revision and SHA-256 digests.
- [x] Save a data card with version, processing status, attribution, labels, domain limitations, and upstream usage terms.

## Stage A — the complete CPU baseline

### 1. Build a small reproducible pipeline

Use Python, pandas, NumPy, scikit-learn and a plotting library. Keep a small dependency set and record tested versions when implementation begins.

Organize the workflow into data audit, train, evaluate, predict, and report commands. Store experiment configuration beside each result. Use a fixed seed for the initial comparison and fit feature extraction only on training data. A scikit-learn pipeline helps keep transformations inside the training boundary. See [official text-feature documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [evaluation pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).

### 2. Compare a small set of models

| Model | Role | Initial configuration to investigate |
|---|---|---|
| Majority-class classifier | Minimum reference | Predict the training majority class |
| Word TF-IDF + logistic regression | Classical lexical baseline | Word 1–2 grams; Amharic-compatible tokenization; bounded vocabulary |
| Character TF-IDF + logistic regression or linear SVM | Main comparison | Character 3–5 grams; bounded vocabulary |

Inspect tokenization on actual Ethiopic text and punctuation before accepting defaults. Tune a small, predefined grid on development data, such as a few regularization values and at most two feature configurations. Compare the same estimator for word/character features first so the feature effect is easier to interpret. A classifier comparison can be a separate experiment.

Start with conservative whitespace/Unicode handling. Preserve negation, emoji and code-switching until their effect is understood; they may contain sentiment information. Do not apply English stemming or stopword lists to Amharic. Specify word and character analyzer choices exactly in the report.

### 3. Define evaluation before examining final test results

Primary project metric: **macro-F1**. Also report per-class precision/recall/F1, accuracy, class counts and a confusion matrix. Record the official task metric separately if comparing with published benchmark results; do not equate differently averaged F1 scores.

Use the development set for selecting the model and preprocessing. Freeze those choices, then run the test evaluation. Never use the final test score as a repeated tuning signal. A later revision informed by test errors must be identified as such and evaluated on fresh held-out data where possible.

Save predictions and metrics with the experiment ID, configuration, source-code revision, dataset version, seed, and training duration. No target accuracy or improvement is promised in advance.

### Stage A acceptance criteria

- [ ] Clean setup can run the documented workflow.
- [ ] Data audit and split boundaries are documented.
- [ ] Majority, word and character baselines have comparable development results.
- [ ] A selected configuration has a clearly labeled held-out evaluation.
- [ ] Results JSON/CSV and a confusion matrix are saved.
- [ ] README explains the question, data, methods, commands, findings and limits.
- [ ] No placeholder score is presented as a measured result.

## Stage B — make the study distinctive

### Amharic-specific error analysis

Review roughly 30–50 development-set mistakes, as available, and define a small error taxonomy based on what you observe. Candidate categories to investigate include negation, spelling/orthographic variation, mixed Amharic/English text, sarcasm, short ambiguous messages, and annotation ambiguity. Do not report these as findings before inspecting examples.

Compare word and character model errors. Explain where one succeeds and the other fails. Do not generalize from a handful of examples to all Amharic speakers or all text domains.

### One controlled ablation

Choose one justified normalization decision. Compare minimal normalization against the proposed change while keeping data, classifier and evaluation fixed. Document any Ethiopic character mapping and its linguistic rationale; avoid collapsing characters solely because it raises one development score. A neutral or negative result is acceptable.

### Lightweight inference demo

Provide a CLI first, then optionally a small local Streamlit interface with Amharic input, predicted class, and a concise explanation of the model's limitations. Display probabilities only if the estimator supports them, and do not describe uncalibrated scores as reliable confidence. Keep the demo separate from the scientific evaluation.

### Stage B acceptance criteria

- [ ] Error-analysis notes show real examples and observations within redistribution limits.
- [ ] One ablation has a recorded comparison and interpretation.
- [ ] Demo loads a saved model and produces the same prediction as the evaluation code.
- [ ] A short model card explains intended use, language/domain limits, and failure modes.

## Stage C — optional transformer and efficiency study

Candidate: [AfroXLM-R base, official model card](https://huggingface.co/Davlan/afro-xlmr-base). Verify the card's current language coverage, licensing, architecture, and resource requirements when implementing. Use it as a pretrained starting point, not as a model you trained from scratch.

Fine-tune a classification head and model on the same training split, choosing hyperparameters using development data. Start with one modest run and a short maximum sequence length supported by the observed data. If memory is insufficient, lower batch size or defer this stage rather than making the whole project fail.

Compare with the best CPU baseline using:

- Macro-F1 and per-class results under the same data protocol.
- Measured inference latency on stated hardware, batch size, input set and warm-up procedure; do not compare GPU and CPU latency as if hardware were equal.
- Saved model size and actual training resources/time.
- Several seeds for the learned model if resources allow. Deterministic baselines do not need repeated identical runs just to inflate the experiment count.

Report whether the quality improvement, if any, justifies the extra cost. Do not assume the transformer wins. Keep any confidence interval tied to a specified method and sample; omit it if it is not implemented correctly.

## Proposed repository structure

Create these files when implementation begins; this tree is a plan, not an inventory of existing code.

```text
Amharic-sentiment-analysis-benchmark/
  plan.md
  README.md
  requirements.txt
  configs/
    word_tfidf.yaml
    char_tfidf.yaml
  src/
    data.py
    preprocessing.py
    train.py
    evaluate.py
    predict.py
  tests/
    test_split_integrity.py
    test_preprocessing.py
  reports/
    data_card.md
    results.csv
    error_analysis.md
    model_card.md
    figures/
  data/                  # ignored; source/version documented
  artifacts/             # ignored; trained model and run metadata
  app.py                 # optional after the experiment works
```

Tests should cover meaningful failure modes: accidental split overlap, vectorizer fitting outside training, label mapping mistakes, and preprocessing that deletes Ethiopic text. Do not spend the first day building deployment infrastructure or an extensive test suite that only repeats library behavior.

## Completion and CV wording

**Current CV entry:** **Amharic Sentiment Analysis — Benchmark Design**, under Projects. It credits the completed experiment protocol without a status label. No implemented model or measured result is claimed.

After Stage A is actually complete, a defensible entry could say:

> Built a reproducible Amharic sentiment-classification benchmark using AfriSenti, comparing word- and character-level TF-IDF classifiers with a majority-class baseline. Evaluated the selected model on a fixed held-out split and documented macro-F1, per-class performance and error patterns.

Add actual sample counts, measured scores and an artifact link only after they exist. Add normalization ablations, the demo, or transformer/latency comparisons only after those stages are complete. A pretrained transformer extension is fine-tuning; it is not creating a new foundation model.

## Suggested build order

1. Verify data access and complete the audit.
2. Implement and evaluate the CPU baselines.
3. Write the first report and check reproducibility.
4. Add error analysis and one controlled ablation.
5. Add a minimal demo.
6. Attempt the optional transformer study if time/compute allow.
7. Publish a coherent repository and revise the CV with the actual results.

The project is successful when another person can understand and reproduce a sound comparison. A high headline accuracy, a public deployment, and a positive transformer result are not required.
