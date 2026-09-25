"""Paper-grouped cross-validation and paper bootstrap confidence intervals."""

import csv
import json
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from controls import balanced_indices
from probe import load_embeddings, make_model, pick_examples, run as run_original, save_confusion
from utils import LABELS, ROOT, SEED, read_jsonl, write_json


N_FOLDS = 5
N_REPEATS = 5
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026


def grouped_folds(records):
    indices = np.array([i for i, row in enumerate(records) if row["domain"] == "nlp"])
    labels = [records[i]["label"] for i in indices]
    papers = [records[i]["paper_id"] for i in indices]
    folds = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + repeat)
        repeat_folds = []
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(indices)), labels, papers)):
            train_papers = {papers[i] for i in train}
            test_papers = {papers[i] for i in test}
            if train_papers & test_papers:
                raise ValueError("Paper leakage")
            repeat_folds.append((repeat, fold, train_papers, test_papers))
        if Counter(paper for _, _, _, test in repeat_folds for paper in test) != Counter(set(papers)):
            raise ValueError("Each paper must be tested once per repeat")
        folds.extend(repeat_folds)
    return folds


def indices_for_papers(records, paper_ids):
    return np.array([i for i, row in enumerate(records) if row["domain"] == "nlp" and row["paper_id"] in paper_ids])


def macro_f1(rows, weights=None):
    return float(f1_score(
        [row["true_label"] for row in rows],
        [row["predicted_label"] for row in rows],
        labels=LABELS, average="macro", sample_weight=weights, zero_division=0,
    ))


def repeat_score(rows, weights=None):
    values = []
    for repeat in range(N_REPEATS):
        indices = [i for i, row in enumerate(rows) if row["repeat"] == repeat]
        subset = [rows[i] for i in indices]
        local_weights = None if weights is None else weights[indices]
        values.append(macro_f1(subset, local_weights))
    return float(np.mean(values)), values


def bootstrap(rows, sampled_papers=None):
    papers = np.array(sorted({row["paper_id"] for row in rows}))
    if sampled_papers is None:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        sampled_papers = rng.choice(papers, size=(BOOTSTRAP_SAMPLES, len(papers)), replace=True)
    paper_position = {paper: i for i, paper in enumerate(papers)}
    label_position = {label: i for i, label in enumerate(LABELS)}
    matrices = np.zeros((N_REPEATS, len(papers), len(LABELS), len(LABELS)), dtype=np.int32)
    for row in rows:
        matrices[row["repeat"], paper_position[row["paper_id"]],
                 label_position[row["true_label"]], label_position[row["predicted_label"]]] += 1
    scores = np.empty(len(sampled_papers))
    for i, sample in enumerate(sampled_papers):
        weights = np.bincount([paper_position[paper] for paper in sample], minlength=len(papers))
        confusion = np.einsum("p,rpij->rij", weights, matrices)
        diagonal = np.diagonal(confusion, axis1=1, axis2=2)
        denominator = confusion.sum(axis=1) + confusion.sum(axis=2)
        per_class = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0)
        scores[i] = per_class.mean(axis=1).mean()
    return scores, sampled_papers


def evaluate(condition, records, embeddings, folds, controlled=False):
    predictions = []
    fold_metrics = []
    tested = []
    for repeat, fold, train_papers, test_papers in folds:
        train_idx = indices_for_papers(records, train_papers)
        test_idx = indices_for_papers(records, test_papers)
        if controlled:
            train_idx = train_idx[balanced_indices([records[i]["label"] for i in train_idx], SEED)]
            test_idx = test_idx[balanced_indices([records[i]["label"] for i in test_idx], SEED + 1)]
        if not len(train_idx) or not len(test_idx):
            raise ValueError(f"Empty train or test in fold {fold}")
        y_train = [records[i]["label"] for i in train_idx]
        y_test = [records[i]["label"] for i in test_idx]
        model = make_model()
        model.fit(embeddings[train_idx], y_train)
        predicted = model.predict(embeddings[test_idx])
        probabilities = model.predict_proba(embeddings[test_idx])
        majority = Counter(y_train).most_common(1)[0][0]
        fold_metrics.append({
            "repeat": repeat,
            "split_seed": SEED + repeat,
            "fold": fold,
            "train_papers": len({records[i]["paper_id"] for i in train_idx}),
            "test_papers": len({records[i]["paper_id"] for i in test_idx}),
            "train_spans": len(train_idx),
            "test_spans": len(test_idx),
            "macro_f1": float(f1_score(y_test, predicted, labels=LABELS, average="macro", zero_division=0)),
        })
        for i, label, probs in zip(test_idx, predicted, probabilities):
            row = records[i]
            predictions.append({
                "span_id": row["span_id"],
                "paper_id": row["paper_id"],
                "repeat": repeat,
                "fold": fold,
                "section_name": row["section_name"],
                "true_label": row["label"],
                "predicted_label": str(label),
                "majority_prediction": majority,
                "confidence": float(probs.max()),
                "correct": bool(label == row["label"]),
                "text": row["text"],
            })
            tested.append(i)
    if not controlled:
        expected = {i for i, row in enumerate(records) if row["domain"] == "nlp"}
        for repeat in range(N_REPEATS):
            repeat_tested = [i for i, row in zip(tested, predictions) if row["repeat"] == repeat]
            if len(repeat_tested) != len(set(repeat_tested)) or set(repeat_tested) != expected:
                raise ValueError("Each NLP span must be tested once per repeat")

    output = ROOT / "results" / condition
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    write_json(output / "qualitative_examples.json", pick_examples(predictions))
    write_json(output / "fold_metrics.json", fold_metrics)
    truth = [row["true_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    majority = [row["majority_prediction"] for row in predictions]
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    np.savetxt(output / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",", header=",".join(LABELS), comments="")
    save_confusion(matrix, output / "confusion_matrix.png", condition.replace("_", " ").title())
    report = classification_report(truth, predicted, labels=LABELS, output_dict=True, zero_division=0)
    scores, _ = bootstrap(predictions)
    mean_f1, repeat_f1 = repeat_score(predictions)
    metrics = {
        "condition": condition,
        "evaluation": "NLP paper-grouped repeated 5-fold out-of-fold",
        "split_seeds": [SEED + i for i in range(N_REPEATS)],
        "folds": N_FOLDS,
        "repeats": N_REPEATS,
        "accuracy": float(np.mean([accuracy_score(
            [row["true_label"] for row in predictions if row["repeat"] == repeat],
            [row["predicted_label"] for row in predictions if row["repeat"] == repeat]
        ) for repeat in range(N_REPEATS)])),
        "macro_f1": mean_f1,
        "macro_f1_per_repeat": repeat_f1,
        "macro_f1_ci95_paper_bootstrap": [float(x) for x in np.quantile(scores, [0.025, 0.975])],
        "bootstrap_replicates": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "per_class": {
            label: {"precision": report[label]["precision"], "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"], "support": int(report[label]["support"])}
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "majority_macro_f1": float(np.mean([f1_score(
            [row["true_label"] for row in predictions if row["repeat"] == repeat],
            [row["majority_prediction"] for row in predictions if row["repeat"] == repeat],
            labels=LABELS, average="macro", zero_division=0,
        ) for repeat in range(N_REPEATS)])),
        "test_papers": len({row["paper_id"] for row in predictions}),
        "test_spans": len(predictions),
        "unique_test_spans": len({row["span_id"] for row in predictions}),
        "test_spans_per_repeat": [sum(row["repeat"] == repeat for row in predictions) for repeat in range(N_REPEATS)],
        "paper_overlap": 0,
    }
    write_json(output / "metrics.json", metrics)
    return metrics, predictions


def update_cross_domain_interval():
    output = ROOT / "results" / "cross_domain"
    with (output / "predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    papers = np.array(sorted({row["paper_id"] for row in rows}))
    positions = {paper: i for i, paper in enumerate(papers)}
    labels = {label: i for i, label in enumerate(LABELS)}
    matrices = np.zeros((len(papers), 3, 3), dtype=np.int32)
    for row in rows:
        matrices[positions[row["paper_id"]], labels[row["true_label"]], labels[row["predicted_label"]]] += 1
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scores = np.empty(BOOTSTRAP_SAMPLES)
    for i in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(len(papers), size=len(papers))
        matrix = matrices[selected].sum(axis=0)
        diagonal = np.diag(matrix)
        denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
        scores[i] = np.divide(2 * diagonal, denominator, out=np.zeros(3), where=denominator != 0).mean()
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["macro_f1_ci95_paper_bootstrap"] = [float(x) for x in np.quantile(scores, [0.025, 0.975])]
    metrics["bootstrap_replicates"] = BOOTSTRAP_SAMPLES
    metrics["bootstrap_seed"] = BOOTSTRAP_SEED
    metrics["test_papers"] = len(papers)
    write_json(metrics_path, metrics)
    return metrics


def run():
    full_rows = read_jsonl(ROOT / "data" / "spans" / "spans.jsonl")
    controlled_rows = read_jsonl(ROOT / "data" / "spans" / "controlled_spans.jsonl")
    full_embeddings = load_embeddings(ROOT / "embeddings" / "full_embeddings", full_rows)
    controlled_embeddings = load_embeddings(ROOT / "embeddings" / "controlled_embeddings", controlled_rows)
    folds = grouped_folds(full_rows)
    write_json(ROOT / "results" / "paper_folds.json", [
        {"repeat": repeat, "split_seed": SEED + repeat, "fold": fold,
         "train_papers": sorted(train), "test_papers": sorted(test)}
        for repeat, fold, train, test in folds
    ])
    results = {}
    results["in_domain"], _ = evaluate("in_domain", full_rows, full_embeddings, folds)
    results["length_controlled"], _ = evaluate(
        "length_controlled", controlled_rows, controlled_embeddings, folds, controlled=True
    )
    run_original("cross_domain")
    results["cross_domain"] = update_cross_domain_interval()
    return results


def main():
    print(json.dumps({"conditions": run()}, indent=2))


if __name__ == "__main__":
    main()
