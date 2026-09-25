# Is semantic level of detail present in frozen embeddings?

## Introduction

Semantic Level of Detail (SLoD) describes how broad or specific a text span is. In this experiment `macro` means a paper-level statement, `meso` means a section-level statement, and `micro` means a detailed method or result. Such a signal could help a retrieval system return a mixture of overview and detailed evidence.

The research question is: can logistic regression predict weak SLoD labels from frozen sentence embeddings? I also test whether the result transfers from NLP to computer vision and whether it remains after controlling text length.

## Data and weak labels

I used a validation shard of `allenai/peS2o`, which is derived from S2ORC. The script scanned 51,323 papers. A paper was assigned to NLP or computer vision when its title matched one domain keyword list but not the other.

The labels were assigned as follows:

- macro: title, abstract, first two introduction paragraphs and conclusion paragraphs;
- meso: first sentence of other sections;
- micro: non-leading paragraphs from methods, experiments, evaluation and results sections.

References, appendices and acknowledgements were skipped. I sampled 250 spans for each domain/class pair, giving 1,500 spans in total and exactly 500 per class. They came from 590 papers. Sampling was limited to four spans from one paper for one label so that a few long papers did not dominate the dataset.

These are weak labels. They describe rhetorical position as well as abstraction. In particular, an introduction can contain implementation details and a methods paragraph can still give a high-level definition.

## Embeddings and probe

I used `sentence-transformers/all-MiniLM-L6-v2`. The model was put in evaluation mode, all parameters were frozen, and its normalized 384-dimensional embeddings were cached. The classifier was logistic regression with a `StandardScaler` and `C=1`. I did not tune hyperparameters.

The three conditions were:

1. **In-domain:** train and test on NLP papers using five repetitions of paper-grouped 5-fold cross-validation. Split seeds were fixed at 42–46. Every NLP paper is tested once per repetition, with no paper shared between training and test within a fold.
2. **Cross-domain:** train on all NLP spans and test on all computer-vision spans.
3. **Length-controlled:** use the same 25 NLP paper splits, but truncate every retained span to exactly 24 model tokens and re-embed it. As in the original evaluation, classes were balanced separately inside each training and test fold.

The suggested 100-150 token range was not suitable for these labels because most titles and many section leads are shorter than 100 tokens. Using 24 tokens keeps examples from all three classes while removing token-count variation among retained spans. Across the 25 folds, the full-text test folds contain 133–170 spans and the balanced controlled test folds contain 90–126 spans. There was no paper overlap between train and test in any fold.

For each repetition, I pooled its five held-out folds and calculated macro F1. The primary estimate is the mean of those five scores. I used 2,000 percentile-bootstrap resamples of whole test papers (seed 2026), with the same paper resample applied across the five repetitions, to obtain 95% intervals. These intervals account for clustering of spans within papers on the fixed dataset and fitted models. They do not include uncertainty from collecting a new corpus or selecting new spans. Fold assignments and individual fold scores are saved in `results/paper_folds.json` and `results/<condition>/fold_metrics.json`.

## Results

| Condition | Accuracy | Macro F1 | 95% paper-bootstrap interval | Majority macro F1 | Evaluation size |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-domain, repeated 5-fold | 0.707 | 0.707 | 0.680–0.733 | 0.296 | 347 papers, 750 unique spans |
| Cross-domain, NLP → CV | 0.655 | 0.634 | 0.601–0.665 | 0.167 | 243 CV papers, 750 spans |
| Length-controlled, repeated 5-fold | 0.453 | 0.453 | 0.425–0.483 | 0.167 | 329 papers, 615 unique tested spans |

For the repeated conditions, accuracy and macro F1 are means across five repetitions. Full-text macro F1 by repetition was 0.731, 0.685, 0.712, 0.713 and 0.692; controlled macro F1 was 0.484, 0.459, 0.430, 0.446 and 0.447. Controlled class balancing selects some spans in several repetitions and omits others, yielding 2,583 held-out predictions in total. Per-class metrics and confusion matrices below pool held-out predictions across repetitions, so their support is a count of predictions rather than unique spans.

Per-class precision and recall:

| Condition | Macro P / R | Meso P / R | Micro P / R |
| --- | ---: | ---: | ---: |
| In-domain | 0.732 / 0.734 | 0.726 / 0.722 | 0.662 / 0.664 |
| Cross-domain | 0.741 / 0.332 | 0.751 / 0.784 | 0.562 / 0.848 |
| Length-controlled | 0.534 / 0.524 | 0.375 / 0.387 | 0.454 / 0.448 |

The confusion matrices are saved with the results:

- [in-domain](results/in_domain/confusion_matrix.png)
- [cross-domain](results/cross_domain/confusion_matrix.png)
- [length-controlled](results/length_controlled/confusion_matrix.png)

Cross-domain macro F1 is 0.073 below the mean in-domain result, but the two conditions also have different training-set sizes and test domains. Generalization is uneven: micro recall is high (0.848), while macro recall falls to 0.332. Many CV macro spans were classified as micro, which suggests that writing style and domain vocabulary affect the boundary.

The length-controlled mean macro F1 is 0.253 lower than the full-text mean (0.453 versus 0.707). This comparison is descriptive: truncation also removes context, excludes short spans and changes the class-balanced test subset. It therefore does not isolate text length as the sole cause of the difference. The controlled score remains above its majority baseline, suggesting some signal survives the intervention.

## Error analysis

I selected the three most confident correct and incorrect predictions for each class from the repeated in-domain held-out predictions, keeping distinct spans within each group. Full text, fold and confidence values are in `results/in_domain/qualitative_examples.json`.

| True class | Result | Predicted | Example (short description) |
| --- | --- | --- | --- |
| Macro | correct | macro | introduction motivating language models for climate and health research |
| Macro | correct | macro | conclusion presenting a multilingual NLU dataset |
| Macro | correct | macro | introduction proposing a question-answering approach to semantic parsing |
| Macro | failed | micro | introduction spelling out task-specific MRC query construction |
| Macro | failed | micro | structured abstract about diagnostic statements for pituitary adenomas |
| Macro | failed | micro | introduction discussing assumptions of MRC systems |
| Meso | correct | meso | section lead introducing results of a voting step |
| Meso | correct | meso | section lead introducing datasets and preprocessing |
| Meso | correct | meso | section lead naming a Random Forest SSL model |
| Meso | failed | macro | section lead proposing a conversation-distillation framework |
| Meso | failed | micro | section lead describing a counterfactual example about cats |
| Meso | failed | micro | section lead beginning with algorithm input and output specifications |
| Micro | correct | micro | experiment paragraph reporting a 9:1 train/test split |
| Micro | correct | micro | analysis paragraph with scene-text detection rates |
| Micro | correct | micro | method paragraph defining MRC context and query windows |
| Micro | failed | macro | methods paragraph on temporal-expression detection tools |
| Micro | failed | meso | paragraph defining a palliative trigger index |
| Micro | failed | meso | methods paragraph defining intermediate beam hypotheses |

The mistakes often make sense semantically even though they disagree with the structural label. Some macro errors contain procedural details despite appearing in introductions. Meso errors include section leads that name a specific algorithm or example, while some micro paragraphs read like broader descriptions. In the pooled controlled predictions, meso is recalled least well (0.387), and 328 of 861 micro predictions are meso. These repeated counts refer to held-out predictions, not independent examples.

## Conclusion and next steps

The experiment shows that structural SLoD labels are linearly decodable from frozen MiniLM embeddings on the selected corpus. The five in-domain macro F1 scores range from 0.685 to 0.731, below the previously reported single-split score of 0.780. Performance is lower under the length-control procedure, although this procedure changes both available text and sample composition. Cross-domain transfer is possible, although macro performance changes considerably between NLP and computer vision.

I would use this probe only as an additional routing feature, not as a hard label. A RAG system could combine its probabilities with relevance and request both macro and micro passages. Before that, I would create a small paper-disjoint test set with continuous SLoD scores from several annotators. Other useful controls are matching numeral and citation density, testing more encoders, and constructing pairs that discuss the same topic at different levels of detail.

## References

Lo, K. et al. (2020). S2ORC: The Semantic Scholar Open Research Corpus. *ACL 2020*. <https://doi.org/10.18653/v1/2020.acl-main.447>

Soldaini, L., & Lo, K. (2023). peS2o (Pretraining Efficiently on S2ORC) Dataset. <https://huggingface.co/datasets/allenai/peS2o>

Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207-219. <https://doi.org/10.1162/coli_a_00422>
