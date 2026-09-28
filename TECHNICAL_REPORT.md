# Is semantic level of detail present in frozen embeddings?

## Introduction

Semantic Level of Detail (SLoD) describes whether a text span gives a paper-level overview (`macro`), a section-level account (`meso`), or a specific method or result (`micro`). I test whether a linear classifier can predict these weak labels from frozen sentence embeddings, whether it transfers from NLP to computer vision (CV), and how it compares with simple feature and TF-IDF baselines.

## Data and weak labels

I used one validation shard of `allenai/peS2o`, derived from S2ORC. The parser scanned 51,323 papers and assigned NLP or CV when exactly one domain keyword list matched the title. Titles, abstracts, first two introduction paragraphs and conclusion paragraphs are `macro`. The **full opening paragraph** of each other section is `meso`. Later paragraphs in methods, experiments, evaluation and results sections are `micro`. References, appendices and acknowledgements were excluded.

Structural sampling selected 250 spans per domain/class cell, capped at four spans per paper and label. The final experiment has **1,500 spans from 591 papers**: 500 each of macro, meso and micro, with 750 NLP and 750 CV spans. These are position-based weak labels, not human judgments of abstraction.

## Embeddings, probe and evaluation

I encoded the final texts with frozen `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional normalized vectors; maximum input length 256 model tokens). The probe is logistic regression after standard scaling (`C=1`, no hyperparameter tuning). For in-domain evaluation, I ran five repetitions of paper-grouped 5-fold cross-validation using seeds 42–46. Each NLP paper is held out once per repetition, with no paper overlap between training and test. The cross-domain condition trains on all NLP spans and tests on all CV spans.

I also evaluated three baselines on the same 750 NLP spans and the same paper-grouped folds: logistic regression on word count alone; logistic regression on word count, digit count and citation-marker count; and logistic regression on word unigram/bigram TF-IDF. Counts are transformed with `log1p` and standardized using training data only. A citation marker is a bracketed numeric reference or a parenthesized expression containing a four-digit year; this heuristic does not find every citation. TF-IDF uses `min_df=2` and sublinear term frequency, with its vocabulary fitted within each training fold. All classifiers use `C=1` without tuning.

I pooled the five held-out folds within each repetition, computed macro F1, and averaged the five scores. Intervals come from 2,000 bootstrap resamples of whole papers and are conditional on the selected corpus, fitted models and splits. Baseline differences use the same sampled papers as the embedding probe. Confusion matrices and per-class metrics pool held-out predictions; their counts are not independent samples.

## Results

| Condition | Accuracy | Macro F1 | 95% paper-bootstrap interval | Majority macro F1 | Evaluation size |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-domain, repeated 5-fold | 0.519 | 0.519 | 0.492–0.545 | 0.305 | 348 papers; 750 unique spans |
| NLP → CV | 0.453 | 0.454 | 0.419–0.489 | 0.167 | 243 papers; 750 spans |

| Condition | Macro P / R | Meso P / R | Micro P / R |
| --- | ---: | ---: | ---: |
| In-domain | 0.679 / 0.676 | 0.407 / 0.402 | 0.471 / 0.479 |
| NLP → CV | 0.699 / 0.436 | 0.345 / 0.316 | 0.416 / 0.608 |

The in-domain score exceeds its majority baseline but remains modest. Transfer to CV is weaker, particularly for `meso` recall (0.316). In the in-domain confusion matrix, 517 of 1,250 held-out true `meso` predictions are classified as `micro`; 482 of 1,250 true `micro` predictions are classified as `meso`. The full paragraphs frequently contain procedural details, making the section-level structural label hard to distinguish from method and result details.

### Simple baselines on the NLP folds

| Model | Accuracy | Macro F1 | 95% paper-bootstrap interval | Difference from embedding probe (95% interval) | Macro / meso / micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Word count only | 0.315 | 0.308 | 0.284–0.334 | −0.210 (−0.248 to −0.175) | 0.357 / 0.259 / 0.316 |
| Word, digit and citation counts | 0.443 | 0.435 | 0.404–0.471 | −0.083 (−0.123 to −0.041) | 0.528 / 0.356 / 0.423 |
| Word unigram/bigram TF-IDF | 0.596 | 0.587 | 0.557–0.617 | +0.068 (+0.033 to +0.105) | 0.784 / 0.419 / 0.559 |
| Frozen MiniLM embeddings | 0.519 | 0.519 | 0.492–0.545 | Reference | 0.677 / 0.405 / 0.475 |

Each row uses 348 NLP papers and 750 unique held-out spans, tested once in each of five repetitions. The count-only model is near the majority baseline (macro F1 0.305). Adding digits and citation markers improves it, but remains below the embedding probe. TF-IDF exceeds the embedding probe by 0.068 macro F1 on these fixed splits. The MiniLM encoder truncates inputs at 256 model tokens while TF-IDF uses the full saved text, so this comparison does not isolate representation quality at a matched input length. These features can also reflect structural source and topic vocabulary; the comparison does not establish that any model recognizes semantic abstraction independently of those cues. Reproduce the comparison with `python src/cross_validate.py --baselines`.

### Encoder and embedding-size comparison

I compared frozen [MiniLM-L6](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [MiniLM-L12](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), and [MPNet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). Each encoder saw the same 750 NLP spans, capped at 256 model tokens, and used the same paper-grouped 5×5 folds and `C=1` logistic probe. Native embeddings have 384, 384 and 768 dimensions respectively. For 16, 32, 64 and 128 dimensions, PCA was fitted on each training fold and then applied to its test fold. The existing MiniLM-L6 cache was checked against the selected encoder and span IDs; new model embeddings were computed without adding cache files to the repository.

| Encoder | 16 | 32 | 64 | 128 | Native |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniLM-L6 | 0.548 | 0.569 | **0.571** | 0.564 | 0.519 (384) |
| MiniLM-L12 | 0.531 | **0.574** | 0.572 | 0.559 | 0.537 (384) |
| MPNet | 0.520 | 0.546 | **0.562** | 0.553 | 0.542 (768) |

Cells are mean held-out macro F1 across the five repetitions. The best tested embedding setting, MiniLM-L12 at 32 dimensions, reached **0.574** (95% paper-bootstrap interval 0.547–0.601; accuracy 0.579). For comparison, TF-IDF scored 0.587 and word count alone 0.308 on the same folds. Reducing dimension improved the point estimate for all three encoders, but the sizes were inspected on these data, so the highest cell is an exploratory selection rather than a separately validated optimum. PCA changes the representation as well as its size; the native points should not be interpreted as a pure dimension effect. The plotted comparison is in [model_size_comparison.ipynb](model_size_comparison.ipynb); reproduce all numbers with `python src/cross_validate.py --model-sizes`.

## Length diagnostics

The plots below are from [test_length.ipynb](test_length.ipynb). They compare token lengths and retention thresholds for the selected `meso` text, with `macro` and `micro` as references. They are **length diagnostics**, not classifier scores. On all 500 `meso` examples, the median is 143 model tokens for the full paragraph. Axis labels and titles are in English.

![Token-length distributions and cumulative distributions](results/meso_length_distribution.png)

![Span retention by token threshold](results/meso_length_retention.png)

## Error analysis and conclusion

The full predictions and confidence-ranked correct and failed examples are in `results/<condition>/`. The highest confusion is between `meso` and `micro`, consistent with opening paragraphs containing concrete technical content. Some `macro` spans also contain detailed procedures despite appearing in an abstract or introduction. These mismatches show why structural source alone is an imperfect proxy for semantic abstraction.

The results show that frozen sentence embeddings contain a linearly decodable signal for these structural labels. MiniLM-L12 with 32-dimensional train-fold PCA was the strongest tested embedding setting (macro F1 0.574), while TF-IDF reached 0.587 on the same NLP folds. The probe is not strong enough to treat as a reliable SLoD labeler. I would use its probabilities only as an auxiliary retrieval feature, alongside relevance. A stronger test would use a paper-disjoint, human-rated abstraction set and controls for source type and topic vocabulary.

## References

Lo, K. et al. (2020). S2ORC: The Semantic Scholar Open Research Corpus. *ACL 2020*. <https://doi.org/10.18653/v1/2020.acl-main.447>

Soldaini, L., & Lo, K. (2023). peS2o (Pretraining Efficiently on S2ORC) Dataset. <https://huggingface.co/datasets/allenai/peS2o>

Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207–219. <https://doi.org/10.1162/coli_a_00422>
