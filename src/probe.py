"""Train and evaluate the linear probe."""

import argparse
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from controls import balanced_indices
from utils import LABELS, ROOT, SEED, read_jsonl, set_seed, write_json


def load_embeddings(prefix, records):
    metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["span_ids"] != [row["span_id"] for row in records]:
        raise ValueError("Embedding rows do not match the span file")
    return np.load(prefix.with_suffix(".npy"))


def nlp_split(records):
    indices = np.array([i for i, row in enumerate(records) if row["domain"] == "nlp"])
    labels = np.array([records[i]["label"] for i in indices])
    papers = np.array([records[i]["paper_id"] for i in indices])
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(split.split(np.zeros(len(indices)), labels, papers))
    return indices[train], indices[test]


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=SEED),
    )


def save_confusion(matrix, output, title):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(3):
        for col in range(3):
            ax.text(col, row, matrix[row, col], ha="center", va="center")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def pick_examples(rows):
    examples = {}
    for label in LABELS:
        correct = sorted(
            [row for row in rows if row["true_label"] == label and row["correct"]],
            key=lambda row: row["confidence"],
            reverse=True,
        )[:3]
        failed = sorted(
            [row for row in rows if row["true_label"] == label and not row["correct"]],
            key=lambda row: row["confidence"],
            reverse=True,
        )[:3]
        examples[label] = {"correct": correct, "failed": failed}
    return examples


def evaluate(condition, records, embeddings, train_idx, test_idx):
    train_rows = [records[i] for i in train_idx]
    test_rows = [records[i] for i in test_idx]
    y_train = np.array([row["label"] for row in train_rows])
    y_test = np.array([row["label"] for row in test_rows])

    model = make_model()
    model.fit(embeddings[train_idx], y_train)
    predicted = model.predict(embeddings[test_idx])
    probabilities = model.predict_proba(embeddings[test_idx])

    rows = []
    for source, label, probs in zip(test_rows, predicted, probabilities):
        rows.append(
            {
                "span_id": source["span_id"],
                "paper_id": source["paper_id"],
                "section_name": source["section_name"],
                "true_label": source["label"],
                "predicted_label": str(label),
                "confidence": float(probs.max()),
                "correct": bool(label == source["label"]),
                "text": source["text"],
            }
        )

    output = ROOT / "results" / condition
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "qualitative_examples.json", pick_examples(rows))

    matrix = confusion_matrix(y_test, predicted, labels=LABELS)
    np.savetxt(
        output / "confusion_matrix.csv",
        matrix,
        fmt="%d",
        delimiter=",",
        header=",".join(LABELS),
        comments="",
    )
    save_confusion(matrix, output / "confusion_matrix.png", condition.replace("_", " ").title())

    report = classification_report(y_test, predicted, labels=LABELS, output_dict=True, zero_division=0)
    majority = Counter(y_train).most_common(1)[0][0]
    majority_prediction = np.full(len(y_test), majority)
    metrics = {
        "condition": condition,
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, labels=LABELS, average="macro")),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "majority_macro_f1": float(
            f1_score(y_test, majority_prediction, labels=LABELS, average="macro", zero_division=0)
        ),
        "train_spans": len(train_idx),
        "test_spans": len(test_idx),
        "paper_overlap": len(
            {row["paper_id"] for row in train_rows} & {row["paper_id"] for row in test_rows}
        ),
    }
    write_json(output / "metrics.json", metrics)
    return metrics


def run(condition):
    set_seed()
    full_rows = read_jsonl(ROOT / "data" / "spans" / "spans.jsonl")
    full_embeddings = load_embeddings(ROOT / "embeddings" / "full_embeddings", full_rows)
    full_train, full_test = nlp_split(full_rows)
    train_papers = {full_rows[i]["paper_id"] for i in full_train}
    test_papers = {full_rows[i]["paper_id"] for i in full_test}

    if condition == "in_domain":
        return evaluate(condition, full_rows, full_embeddings, full_train, full_test)

    if condition == "cross_domain":
        train = np.array([i for i, row in enumerate(full_rows) if row["domain"] == "nlp"])
        test = np.array([i for i, row in enumerate(full_rows) if row["domain"] == "cv"])
        return evaluate(condition, full_rows, full_embeddings, train, test)

    rows = read_jsonl(ROOT / "data" / "spans" / "controlled_spans.jsonl")
    embeddings = load_embeddings(ROOT / "embeddings" / "controlled_embeddings", rows)
    train_pool = [
        i for i, row in enumerate(rows) if row["domain"] == "nlp" and row["paper_id"] in train_papers
    ]
    test_pool = [
        i for i, row in enumerate(rows) if row["domain"] == "nlp" and row["paper_id"] in test_papers
    ]
    train_local = balanced_indices([rows[i]["label"] for i in train_pool], SEED)
    test_local = balanced_indices([rows[i]["label"] for i in test_pool], SEED + 1)
    train = np.array([train_pool[i] for i in train_local])
    test = np.array([test_pool[i] for i in test_local])
    return evaluate(condition, rows, embeddings, train, test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--condition",
        choices=("in_domain", "cross_domain", "length_controlled", "all"),
        default="all",
    )
    args = parser.parse_args()

    conditions = (
        ("in_domain", "cross_domain", "length_controlled")
        if args.condition == "all"
        else (args.condition,)
    )
    for condition in conditions:
        print(json.dumps(run(condition), indent=2))


if __name__ == "__main__":
    main()
