import hashlib
import json
import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("macro", "meso", "micro")
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_id(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
