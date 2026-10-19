# Next steps: Amharic sentiment analysis

Prepared 19 October 2026. Use this as the handoff for the next implementation session. The full study design remains in [plan.md](plan.md).

## Groundwork completed

- [x] Create the project's isolated `.venv` for Python 3.12.
- [x] Install and verify all 24 pinned CPU-environment packages, including NumPy, pandas, scikit-learn, SciPy and Matplotlib.
- [x] Download all three labeled Amharic AfriSenti splits: 5,984 train, 1,497 development, 1,999 test examples.
- [x] Pin upstream revision and SHA-256 hashes in `configs/dataset.json`.
- [x] Preserve original UTF-8 TSV files under ignored `data/raw/amh/`.
- [x] Implement strict data loading and reproducible download/verification commands.
- [x] Prepare class-balance, duplicate, overlap and text-length auditing.
- [x] Record data provenance, usage terms, citations and evaluation boundaries.
- [x] Define the next-stage CPU baseline settings in `configs/baselines.yaml`.
- [x] Add checks for corrupted files, malformed rows and preservation of Ethiopic text.
- [x] Pass all four integrity tests, `pip check`, package-version matching, Ethiopic/emoji analyzer checks and headless plotting.

Dependency versions and import/plotting verification are recorded in `requirements.txt` and `reports/environment.json`. See `reports/data_audit.md` and its JSON companion for the data checks. No benchmark classifier has been trained, and no performance score is available yet.

## Resume the environment

```bash
cd "/home/mint/Desktop/scholarship with ai/projects/Amharic-sentiment-analysis-benchmark"
source .venv/bin/activate
python -m scripts.download_data
python -m scripts.check_environment
python -m unittest discover -s tests -v
```

For another machine or a deleted environment, follow the installation commands in `README.md`. Do not rerun unpinned dependency installation if you want the tested environment.

## 1. Confirm the experiment protocol

- [ ] Read the data card and audit. Keep the released splits for the primary comparison.
- [ ] Inspect a small selection of **training and development** messages locally to check punctuation, tokenization, code-switching and annotation examples. Do not publish raw tweets in a notebook by default.
- [ ] Keep the one exact training duplicate initially. If a deduplication sensitivity experiment is useful, define it separately and record changed counts.
- [ ] Use macro-F1 as the primary metric; also save weighted-F1, accuracy and per-class precision/recall/F1. Class order is `negative`, `neutral`, `positive`.
- [ ] Keep test data out of feature fitting, hyperparameter selection and early error analysis. A neutral-majority training set and negative-majority test set make accuracy especially easy to misread.

Done when: a short protocol in the README states the data revision, split roles, metrics, candidate settings and tie-breaking rule before model comparison begins.

## 2. Implement the majority reference first

- [ ] Create `src/train.py` and `src/evaluate.py` with explicit train/development arguments and configuration loading.
- [ ] Fit `DummyClassifier(strategy="most_frequent")` on training labels only; evaluate on development.
- [ ] Save a metrics JSON, development predictions and the resolved configuration under ignored `artifacts/<run_id>/`.
- [ ] Save seed, package versions, dataset hashes, code revision and elapsed training time alongside the run. Identify an uncommitted working tree when applicable.

Done when: one documented command reproduces the reference results and every development row has one prediction. The command still needs to be implemented; no `train` command exists at the groundwork stage.

## 3. Build the word and character baselines

- [ ] Use `Pipeline(TfidfVectorizer(...), LogisticRegression(...))` so vocabulary and IDF are fitted on train only.
- [ ] Implement both feature settings in `configs/baselines.yaml`, initially with the same estimator.
- [ ] Run the small development-only regularization grid (`C = 0.5, 1, 2`). Keep all candidates' results rather than just the winner.
- [ ] Verify the word analyzer keeps Ethiopic text. Document that word features exclude punctuation/emoji while character features can capture them.
- [ ] Handle any convergence warning by inspecting the solver/iteration count, then record any configuration change.
- [ ] Save trained pipelines with joblib, local predictions and a compact development comparison table in `reports/`.
- [ ] Add a meaningful check that inference uses the saved fitted vectorizer rather than fitting a new one on development or user input.

Done when: majority, word and character baselines have comparable development metrics, persisted models and complete run metadata. No GPU or pretrained model is needed.

## 4. Investigate development errors and one ablation

- [ ] Compare roughly 30–50 development mistakes across the word and character models.
- [ ] Create `reports/error_analysis.md` with categories supported by actual examples. Candidate topics include negation, spelling variation, mixed-language text and ambiguity.
- [ ] Choose one justified preprocessing change. If using Ethiopic character mappings, document their linguistic rationale before comparing results.
- [ ] Compare that change with the unmodified-text baseline while holding the split, estimator and candidate budget fixed.
- [ ] Report improvement, deterioration or no clear change honestly. Avoid repeatedly adding experiments in pursuit of one high score.

Done when: the report explains observed failures and includes a controlled development-set comparison. Row-level review files remain local under `artifacts/`.

## 5. Freeze the model and evaluate test

- [ ] Select the model and preprocessing using development results and write down the frozen configuration.
- [ ] Evaluate the selected train-fitted model on the original test split. If retraining on train+dev is desired, preregister it as a separate protocol first.
- [ ] Save macro-F1, weighted-F1, accuracy, per-class results, class counts and confusion matrix.
- [ ] Explain the observed class-distribution difference and the limits of this single benchmark.
- [ ] Publish aggregate metrics and methodology in `reports/results.csv` and a short report, keeping row-level outputs ignored.

Done when: every result can be traced to data hashes, code/configuration and a saved run. Further changes informed by test results require a clearly identified revised evaluation protocol.

## 6. Make the result usable

- [ ] Implement `src/predict.py` to load a saved pipeline and classify Amharic input locally.
- [ ] Confirm that the same saved model gives identical predictions through evaluation and inference paths.
- [ ] Optionally add a small Streamlit demo after the CLI works; install its dependencies separately at that stage.
- [ ] Write `reports/model_card.md` with measured performance, intended use, data domain and actual limitations. Do not label uncalibrated probabilities as reliable confidence.
- [ ] Reproduce the workflow in a clean environment, choose a code license and review tracked files before publishing.

Done when: someone else can set up the repository, reproduce the comparison and try the model without private credentials or a GPU.

## 7. Optional extension after the CPU study

Consider fine-tuning [AfroXLM-R base](https://huggingface.co/Davlan/afro-xlmr-base) if GPU resources are available. Verify its current model card and license first. Install PyTorch, Transformers and the chosen training dependencies separately once the hardware is known. They are not required for the current CPU setup.

Use the same split protocol. Compare quality with measured training time, model size and inference latency on stated hardware. Do not assume the transformer wins. If test results already influenced the extension's design, label the later comparison accordingly rather than claiming a pristine final test.

## Recommended next session

Complete steps 1–3: implement the majority, word and character baselines and compare **development** results. This produces the first real modeling deliverable. Add CV claims about implementation, performance, error analysis or transformer fine-tuning only as those results are completed.
