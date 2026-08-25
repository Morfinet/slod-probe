# SLoD probe

This repository checks whether a frozen text embedding contains enough information for a linear classifier to distinguish three levels of detail in scientific papers:

- `macro`: titles, abstracts, introduction and conclusion paragraphs;
- `meso`: the first sentence of a regular section;
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

The other conditions are:

```bash
python src/probe.py --train --eval --condition cross_domain
python src/probe.py --train --eval --condition length_controlled
```

Each command writes `metrics.json`, a confusion matrix, predictions and qualitative examples to `results/<condition>/`.

## Rebuilding the data

The source is one validation shard of `allenai/peS2o`, which is derived from S2ORC. The download is about 493 MB.

```bash
python src/dataset.py --download
python src/embed.py --condition both
python src/probe.py --condition all
```

`dataset.py` scans the shard, assigns a domain using title keywords, creates structural labels and samples 250 spans for every domain/class combination. The final data contain 1,500 spans from 590 papers, with 500 spans per label.

I used `sentence-transformers/all-MiniLM-L6-v2`. Its parameters are frozen and only the cached 384-dimensional embeddings are passed to logistic regression. Train/test splitting is done by `paper_id`, not by individual span.

For the length control, every retained span is cut to exactly 24 model tokens and embedded again. I used 24 rather than 100-150 tokens because titles and many section-leading sentences are shorter than 100 tokens. A larger lower bound would remove most of these examples and would change the dataset at the same time as controlling length.
