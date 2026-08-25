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

I used `sentence-transformers/all-MiniLM-L6-v2`. The model was put in evaluation mode, all parameters were frozen, and its normalized 384-dimensional embeddings were cached. The classifier was logistic regression with a `StandardScaler`, `C=1` and random seed 42. I did not tune hyperparameters.

The three conditions were:

1. **In-domain:** train and test on NLP papers using a grouped split.
2. **Cross-domain:** train on all NLP spans and test on all computer-vision spans.
3. **Length-controlled:** use the same NLP paper split, but truncate every retained span to exactly 24 model tokens and re-embed it. Classes were balanced again inside train and test.

The suggested 100-150 token range was not suitable for these labels because most titles and many section leads are shorter than 100 tokens. Using 24 tokens keeps examples from all three classes while removing token-count variation. The controlled set had 423 training spans and 102 test spans. There was no paper overlap between train and test in any condition.

## Results

| Condition | Accuracy | Macro F1 | Majority macro F1 | Train / test |
| --- | ---: | ---: | ---: | ---: |
| In-domain | 0.780 | 0.780 | 0.167 | 600 / 150 |
| Cross-domain | 0.655 | 0.634 | 0.167 | 750 / 750 |
| Length-controlled | 0.471 | 0.469 | 0.167 | 423 / 102 |

Per-class precision and recall:

| Condition | Macro P / R | Meso P / R | Micro P / R |
| --- | ---: | ---: | ---: |
| In-domain | 0.816 / 0.800 | 0.804 / 0.740 | 0.727 / 0.800 |
| Cross-domain | 0.741 / 0.332 | 0.751 / 0.784 | 0.562 / 0.848 |
| Length-controlled | 0.636 / 0.618 | 0.390 / 0.471 | 0.393 / 0.324 |

The confusion matrices are saved with the results:

- [in-domain](results/in_domain/confusion_matrix.png)
- [cross-domain](results/cross_domain/confusion_matrix.png)
- [length-controlled](results/length_controlled/confusion_matrix.png)

Cross-domain macro F1 is 0.146 below the in-domain result, but still far above the majority baseline. Generalization is uneven: micro recall is high (0.848), while macro recall falls to 0.332. Many CV macro spans were classified as micro, which suggests that writing style and domain vocabulary affect the boundary.

The most important result is the length control. Macro F1 falls by 0.311, from 0.780 to 0.469. Therefore, length explains a large part of the original result. The controlled score is still above the baseline, so there is some remaining signal in the embeddings, but it is much weaker.

## Error analysis

I selected the three most confident correct and incorrect predictions for each class from the in-domain test set. Full text and confidence values are in `results/in_domain/qualitative_examples.json`.

| True class | Result | Predicted | Example (short description) |
| --- | --- | --- | --- |
| Macro | correct | macro | introduction motivating continual improvement of existing NMT models |
| Macro | correct | macro | abstract about zero-shot NLI for Indigenous languages |
| Macro | correct | macro | abstract surveying neural code intelligence |
| Macro | failed | micro | conclusion specifying task-specific MRC query construction |
| Macro | failed | meso | conclusion summarizing an unsupervised cargo-classification model |
| Macro | failed | micro | introduction dominated by multimodal MT data and evaluation details |
| Meso | correct | meso | section lead giving the history of text summarization |
| Meso | correct | meso | section lead describing Twitter proof-of-concept experiments |
| Meso | correct | meso | training-data lead pointing to bitext counts |
| Meso | failed | micro | section lead defining Integrated Gradients |
| Meso | failed | micro | short summary sentence about a rider's team performance |
| Meso | failed | micro | evaluation-metric lead naming entity-level F1 |
| Micro | correct | micro | training setup with class weighting and a learning-rate schedule |
| Micro | correct | micro | pretraining procedure defining context and query windows |
| Micro | correct | micro | ablation removing noisy-token embeddings after encoding |
| Micro | failed | macro | results paragraph dominated by future-work discussion |
| Micro | failed | meso | appendix fragment listing prompt tables |
| Micro | failed | macro | broad interpretation of clustering results |

The mistakes often make sense semantically even though they disagree with the structural label. The most confident macro errors contain dense procedural details despite appearing in introductions or conclusions. Meso errors include short section leads that name a specific method or metric, while some micro paragraphs give broad interpretations rather than implementation details. After length control, meso and micro are the hardest pair: 18 of 34 micro test examples were predicted as meso. This is evidence that the first-sentence rule does not define a clean semantic class.

## Conclusion and next steps

The experiment shows that structural SLoD labels are linearly decodable from frozen MiniLM embeddings, but the uncontrolled score overstates the result because length is a strong confound. Cross-domain transfer is possible, although macro performance changes considerably between NLP and computer vision.

I would use this probe only as an additional routing feature, not as a hard label. A RAG system could combine its probabilities with relevance and request both macro and micro passages. Before that, I would create a small paper-disjoint test set with continuous SLoD scores from several annotators. Other useful controls are matching numeral and citation density, testing more encoders, and constructing pairs that discuss the same topic at different levels of detail.

## References

Lo, K. et al. (2020). S2ORC: The Semantic Scholar Open Research Corpus. *ACL 2020*. <https://doi.org/10.18653/v1/2020.acl-main.447>

Soldaini, L., & Lo, K. (2023). peS2o (Pretraining Efficiently on S2ORC) Dataset. <https://huggingface.co/datasets/allenai/peS2o>

Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207-219. <https://doi.org/10.1162/coli_a_00422>
