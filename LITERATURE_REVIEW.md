# Literature review: probing semantic level of detail

The goal of this project is to test whether a frozen language model embedding contains information about the level of detail of a scientific text span. I use three labels: `macro` for paper-level statements, `meso` for section-level statements and `micro` for implementation or result details. The three papers below are useful for different parts of this problem: Belinkov discusses what probing results mean, Nickel and Kiela study embeddings for hierarchical data, and Ratner et al. describe weak supervision.

## Probing classifiers

Belinkov (2022) reviews probing classifiers as a way to analyse representations. In a typical probing experiment, the encoder is frozen and a separate classifier is trained to predict some property from its output. If a linear classifier performs well, it means that the property can be recovered through a linear decision boundary. This is useful because the probe itself has limited capacity.

However, a good probing score does not prove that the original model uses this information. It also does not prove that the representation has learned the intended linguistic property. The classifier may use an unintended correlation in the data. A complex probe may learn much of the task itself, and a random or simple input baseline may already perform well. For this reason, it is safer to say that a property is *decodable* from an embedding rather than that the model understands or uses it.

Several controls follow from this point. The probe should be simple, the encoder must stay frozen, and the result should be compared with a majority baseline. Train and test examples should also be separated at the document level. Otherwise, similar spans from the same paper may appear on both sides of the split. In this project the main additional control is text length, because section position and paragraph length are strongly related.

For SLoD, logistic regression is a suitable first probe. It cannot establish a causal role of abstraction level inside MiniLM, but it can answer a narrower question: is there a linear signal that predicts the structural labels?

## Hierarchical representations

Nickel and Kiela (2017) show that the geometry of an embedding matters when the data have a hierarchy. Trees grow exponentially with depth, while Euclidean space does not naturally have this property. Poincare embeddings use a negatively curved space, where the available area increases quickly with distance from the centre. This makes it possible to represent both similarity and hierarchical position using relatively few dimensions.

Their experiments on WordNet show that low-dimensional Poincare embeddings reconstruct hierarchical relations better than comparable Euclidean embeddings. This is evidence that a suitable inductive bias can make hierarchy easier to encode.

There is still an important difference between WordNet and scientific level of detail. WordNet contains explicit relations between concepts. SLoD is not a clean tree. An abstract may contain a numerical detail, while a methods paragraph may start with a broad motivation. Also, `all-MiniLM-L6-v2` is a Euclidean sentence encoder trained for semantic similarity, not a model trained to place spans at different levels of a hierarchy.

Therefore, a successful SLoD probe would not show that MiniLM has learned a Poincare-like hierarchy. It would only show that its vectors contain features correlated with the labels. A useful later experiment would compare ordinary sentence embeddings with embeddings trained using an explicit hierarchical objective, or predict a continuous level instead of three classes.

## Weak supervision from document structure

Ratner et al. (2017) propose Snorkel, where users write labeling functions instead of manually labeling every example. These functions can be noisy, can disagree and can abstain. A generative model estimates their accuracy and correlation, then produces probabilistic labels for a downstream model.

The general idea is relevant here because scientific papers already have structure that can be converted into labels. Titles and abstracts usually describe the whole paper, section opening sentences often introduce a local topic, and later methods or results paragraphs usually contain more detail. This gives a large dataset without manual annotation and makes the labeling procedure reproducible.

The limitation is that document structure is not ground truth. In this prototype all labels come from one family of structural rules, so there are no independent labeling functions whose agreement can be analysed. The errors are also systematic rather than random. For example, abstracts tend to be shorter than methods paragraphs, section leads use characteristic phrases, and different fields use different section names. A classifier can learn these regularities without learning semantic abstraction.

A small manually annotated sample would be the best way to estimate label quality. Multiple annotators could score detail on a continuous scale and disagreements could be kept instead of forcing a single label. The weak labels would still be useful for development, but final evaluation should use the human set.

## Application to the SLoD experiment

The literature suggests a deliberately limited claim. I can test whether structural SLoD labels are linearly decodable from frozen embeddings, but not whether SLoD is a causal internal variable of the encoder.

The experiment therefore uses a frozen MiniLM model and logistic regression. Papers, not spans, are the unit of the train/test split. Cross-domain evaluation checks whether a probe trained on NLP papers transfers to computer vision. The length-controlled condition re-embeds spans after cutting them to the same number of model tokens. If performance drops strongly, part of the original result was caused by length.

Length is not the only possible confound. Other likely signals are numerals, citations, tense, discourse phrases, formulas and section-specific vocabulary. Domain selection based on title words may also create a bias. Another problem is the `meso` class: the first sentence of a section can be either a short transition or a detailed description of an algorithm. This makes the class less clearly defined than macro or micro.

If the controlled probe stays above the baseline, it may still be useful as one feature in a retrieval system. For example, a RAG pipeline could request both overview and detailed passages. It should not use the predicted class as a hard filter until the probe has been tested on human labels and additional domains.

## References

Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. *Computational Linguistics*, 48(1), 207-219. <https://doi.org/10.1162/coli_a_00422>

Nickel, M., & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. *NeurIPS 30*, 6338-6347. <https://proceedings.neurips.cc/paper_files/paper/2017/hash/59dfa2df42d9e3d41f5b02bfc32229dd-Abstract.html>

Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Re, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *PVLDB*, 11(3), 269-282. <https://doi.org/10.14778/3157794.3157797>
