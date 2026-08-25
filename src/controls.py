from collections import defaultdict

import numpy as np

from utils import LABELS, make_id


CONTROL_TOKENS = 24


def make_length_controlled(records, tokenizer, target=CONTROL_TOKENS):
    """Truncate every retained span to the same number of model tokens."""
    result = []
    for record in records:
        ids = tokenizer.encode(record["text"], add_special_tokens=False, truncation=False)
        if len(ids) < target:
            continue

        row = dict(record)
        row["original_span_id"] = record["span_id"]
        row["span_id"] = make_id(record["span_id"], "length_controlled", str(target))
        row["text"] = tokenizer.decode(ids[:target], skip_special_tokens=True).strip()
        row["original_token_count"] = record["token_count"]
        row["token_count"] = target
        row["model_token_count"] = target
        result.append(row)
    return result


def balanced_indices(labels, seed):
    """Downsample classes to the size of the smallest class."""
    groups = defaultdict(list)
    for i, label in enumerate(labels):
        groups[label].append(i)

    if any(label not in groups for label in LABELS):
        raise ValueError("One of the classes is missing")

    rng = np.random.default_rng(seed)
    size = min(len(groups[label]) for label in LABELS)
    selected = []
    for label in LABELS:
        selected.extend(rng.choice(groups[label], size, replace=False))
    return np.asarray(sorted(selected), dtype=int)
