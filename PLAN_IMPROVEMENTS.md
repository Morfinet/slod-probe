# SLoD Experiment: Improvement Plan

## Priority 1 — Test alternative definitions of `meso`

- [X] Compare the current first-sentence rule with the first 2–3 sentences of the section's opening paragraph.
- [X] Include the full opening paragraph as an additional condition, with a consistent length limit.
- [X] For each definition, report length distributions and the number of available examples per class.
- [X] Treat each definition as a separate condition and keep the test set fixed.

## Priority 2 — Add simple feature baselines

- [X] Train a classifier using text length only.
- [X] Add a baseline using simple surface features: word, digit and citation-marker counts.
- [X] Compare the embedding probe with a TF-IDF classifier on the same paper-grouped folds. See `TECHNICAL_REPORT.md`.

## Priority 3 — Measure uncertainty across paper splits and samples

- [X] Completed for the fixed selected dataset: five repetitions of paper-grouped 5-fold evaluation (seeds 42–46) and 95% paper-bootstrap intervals from 2,000 resamples. See `TECHNICAL_REPORT.md` and `results/`.
- [ ] Still pending: repeat the original span selection with different sampling seeds and regenerate embeddings.
- [ ] Repeat evaluation across several paper-level train/test splits and span-sampling seeds.
- [ ] Calculate confidence intervals for macro F1 and for differences between conditions.
- [ ] Resample papers, rather than individual spans, when estimating confidence intervals.

## Priority 4 — Evaluate cross-domain transfer in both directions

- [ ] Run NLP → CV and CV → NLP experiments.
- [ ] Match the number of training examples across directions and conditions.
- [ ] Report per-class precision, recall, and F1, in addition to aggregate metrics.

## Priority 5 — Compare embedding models

- [X] Evaluate MiniLM-L6, MiniLM-L12 and MPNet on the same NLP examples and paper splits.
- [X] Keep the probe and evaluation procedure fixed across encoders.
- [X] Compare results on the same full-text examples and paper splits.
- [X] Verify that cached embeddings correspond to the selected encoder.

## Priority 6 — Tune logistic-regression regularization

- [ ] Evaluate a predefined range of `C` values using grouped cross-validation within the training data.
- [ ] Select `C` without using the test set.
- [ ] Compare the selected model with the current `C = 1` baseline.

## Priority 7 — Study input length and embedding dimensionality separately

- [ ] Compare several predefined input token limits.
- [ ] Report class retention at each limit, especially for `meso`.
- [X] Reduce vectors with PCA fitted on training data only when testing embedding dimensionality.
- [ ] Keep input token length and vector dimensionality as separate experimental factors.

## Priority 8 — Check for source-type shortcuts

- [ ] Repeat the analysis without titles.
- [ ] Report results separately for titles, abstracts, introductions, conclusions, section leads, and detail paragraphs where sample sizes permit.
- [ ] Check whether performance depends on recognizing the structural source of a span.

## Reporting checklist

- [ ] Report macro F1, accuracy, per-class metrics, and confusion matrices.
- [ ] Include the numbers of papers and spans in every condition.
- [ ] Provide confidence intervals for the main results and comparisons.
- [ ] Use the same examples and paper splits when comparing conditions whenever possible.
- [ ] Predefine primary comparisons; label analyses chosen after inspecting test results as exploratory.
