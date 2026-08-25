"""Extract and cache frozen MiniLM embeddings."""

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from controls import CONTROL_TOKENS, make_length_controlled
from utils import ROOT, read_jsonl, write_json, write_jsonl


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(name):
    model = SentenceTransformer(name)
    model.max_seq_length = 256
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def save_embeddings(model, records, prefix, model_name, batch_size):
    array_path = prefix.with_suffix(".npy")
    metadata_path = prefix.with_suffix(".json")
    span_ids = [row["span_id"] for row in records]

    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("span_ids") == span_ids:
            return

    embeddings = model.encode(
        [row["text"] for row in records],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, embeddings)
    write_json(
        metadata_path,
        {
            "model": model_name,
            "shape": list(embeddings.shape),
            "weights_frozen": all(not p.requires_grad for p in model.parameters()),
            "span_ids": span_ids,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spans", type=Path, default=ROOT / "data" / "spans" / "spans.jsonl")
    parser.add_argument("--condition", choices=("full", "controlled", "both"), default="both")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = read_jsonl(args.spans)
    model = load_model(args.model)

    if args.condition in ("full", "both"):
        save_embeddings(model, records, ROOT / "embeddings" / "full_embeddings", args.model, args.batch_size)

    if args.condition in ("controlled", "both"):
        controlled = make_length_controlled(records, model.tokenizer, CONTROL_TOKENS)
        write_jsonl(args.spans.with_name("controlled_spans.jsonl"), controlled)
        save_embeddings(
            model,
            controlled,
            ROOT / "embeddings" / "controlled_embeddings",
            args.model,
            args.batch_size,
        )


if __name__ == "__main__":
    main()
