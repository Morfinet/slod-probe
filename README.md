# SLoD probe

This repository checks whether a frozen text embedding contains enough information for a linear classifier to distinguish three levels of detail in scientific papers:

- `macro`: titles, abstracts, introduction and conclusion paragraphs;
- `meso`: the full opening paragraph of a regular section;
- `micro`: detailed paragraphs from methods, experiments and results.

The labels are produced from document structure, so they are weak labels rather than human annotation.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install -r requirements.txt
```

The repository already contains the selected spans and cached embeddings. The command requested in the assignment can therefore be run directly:

```bash
python src/probe.py --train --eval --condition in_domain
```

The cross-domain condition is:

```bash
python src/probe.py --train --eval --condition cross_domain
```

The in-domain command runs five repetitions of paper-grouped 5-fold cross-validation (split seeds 42–46). Each repetition tests every NLP paper once. The reported macro F1 is the mean of the five out-of-fold macro F1 scores, not the mean of 25 individual fold scores. A 95% percentile interval is calculated from 2,000 bootstrap resamples of whole papers (seed 2026), using the same sampled papers across the five repetitions. This interval is conditional on the selected spans, fitted models and five split assignments; it does not cover a new draw of papers or spans from the source corpus.

Each command writes `metrics.json`, a confusion matrix, predictions and qualitative examples to `results/<condition>/`. The in-domain condition also writes `fold_metrics.json`; `results/paper_folds.json` records every train/test paper assignment. Confusion matrices and per-class counts pool predictions from all five repetitions, so each full-text span appears five times. The cross-domain condition remains a single NLP-to-CV train/test experiment, with a paper-bootstrap interval on the CV test papers.

To reproduce the length, surface-count and TF-IDF baselines on the same NLP paper folds, run `python src/cross_validate.py --baselines`. It prints their metrics and paired paper-bootstrap differences from the saved embedding-probe predictions. The comparison is summarized in `TECHNICAL_REPORT.md`.

## Rebuilding the data

The source is one validation shard of `allenai/peS2o`, which is derived from S2ORC. The download is about 493 MB.

```bash
python src/dataset.py --download
python src/embed.py
python src/probe.py --condition all
```

`dataset.py` scans the shard, assigns a domain using title keywords, creates structural labels with full opening paragraphs for `meso`, and samples up to 250 spans for every domain/class combination. The **saved evaluation cohort** contains 1,500 spans from 591 papers: 500 each of macro, meso and micro. The reported numbers are reproducible from the saved `data/spans/spans.jsonl` and embedding cache.

I used `sentence-transformers/all-MiniLM-L6-v2`. Its parameters are frozen and only the cached 384-dimensional embeddings are passed to logistic regression. Train/test splitting is done by `paper_id`, not by individual span. Run `python src/cross_validate.py` to regenerate all reported results in one command.

The main report includes length-distribution plots exported from `test_length.ipynb`.
