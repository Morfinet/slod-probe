"""Build structural SLoD labels from an S2ORC-derived peS2o shard."""

import argparse
import gzip
import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from utils import LABELS, ROOT, SEED, make_id, write_json, write_jsonl


DATA_URL = (
    "https://huggingface.co/datasets/allenai/peS2o/resolve/main/"
    "data/v1/validation-00001-of-00002.json.gz"
)

DOMAIN_PATTERNS = {
    "nlp": re.compile(
        r"\b(natural language processing|language models?|language modeling|machine translation|"
        r"named entity|text classification|question answering|dialog(?:ue| system)|sentiment analysis|"
        r"semantic parsing|information extraction|text generation|summari[sz]ation|speech translation|"
        r"document retrieval)\b",
        re.I,
    ),
    "cv": re.compile(
        r"\b(computer vision|image classification|object detection|image segmentation|semantic segmentation|"
        r"visual recognition|image recognition|video recognition|image retrieval|pose estimation|"
        r"visual tracking|action recognition|scene understanding|image captioning|depth estimation)\b",
        re.I,
    ),
}

INTRO = re.compile(r"\bintroduction\b", re.I)
CONCLUSION = re.compile(r"\b(conclusion|conclusions|concluding remarks|summary and discussion)\b", re.I)
EXCLUDED = re.compile(
    r"\b(reference|bibliograph|acknowledg|appendix|supplement\w*|author contribution|front matter)\b",
    re.I,
)
DETAIL = re.compile(
    r"\b(method|methodology|material|experiment|evaluation|result|analysis|implementation|"
    r"architecture|model|approach|training|setup|dataset|data collection|ablation|performance)\b",
    re.I,
)
SECTION_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+", re.I)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def is_header(text):
    words = text.split()
    return (
        1 <= len(words) <= 14
        and len(text) <= 120
        and text[-1] not in ".,;!?"
        and bool(re.search(r"[A-Za-z]", text))
        and (text[0].isupper() or text[0].isdigit())
        and not re.match(r"^(figure|table|equation|copyright)\b", text, re.I)
    )


def parse_document(text):
    blocks = [clean(line) for line in text.splitlines() if clean(line)]
    if len(blocks) < 4:
        return "", "", []

    title, abstract = blocks[:2]
    sections = []
    current = {"name": "Front matter", "paragraphs": []}
    for block in blocks[2:]:
        if is_header(block):
            if current["paragraphs"]:
                sections.append(current)
            name = SECTION_NUMBER.sub("", block).strip(" :-")
            current = {"name": name or "Unlabeled", "paragraphs": []}
        else:
            current["paragraphs"].append(block)
    if current["paragraphs"]:
        sections.append(current)
    return title, abstract, sections


def domain_from_title(title):
    matches = [name for name, pattern in DOMAIN_PATTERNS.items() if pattern.search(title)]
    return matches[0] if len(matches) == 1 else None


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text, maxsplit=1)[0]


def make_span(record, domain, section, label, source, text):
    text = clean(text)
    minimum = {
        "title": 4,
        "abstract": 40,
        "introduction": 30,
        "conclusion": 30,
        "section_lead": 10,
        "detail": 30,
    }[source]
    if len(text.split()) < minimum:
        return None

    paper_id = str(record["id"])
    return {
        "span_id": make_id(paper_id, domain, section, label, source, text),
        "paper_id": paper_id,
        "section_name": section,
        "label": label,
        "text": text,
        "token_count": len(text.split()),
        "domain": domain,
        "source_kind": source,
    }


def label_paper(record, domain):
    title, abstract, sections = parse_document(record.get("text", ""))
    if not title or not abstract or not sections:
        return []

    spans = []

    def add(section, label, source, text):
        span = make_span(record, domain, section, label, source, text)
        if span is not None:
            spans.append(span)

    add("Title", "macro", "title", title)
    add("Abstract", "macro", "abstract", abstract)

    for section in sections:
        name = section["name"]
        paragraphs = section["paragraphs"]
        if not paragraphs or EXCLUDED.search(name):
            continue
        if INTRO.search(name):
            for paragraph in paragraphs[:2]:
                add(name, "macro", "introduction", paragraph)
        elif CONCLUSION.search(name):
            for paragraph in paragraphs[:2]:
                add(name, "macro", "conclusion", paragraph)
        else:
            add(name, "meso", "section_lead", first_sentence(paragraphs[0]))
            if DETAIL.search(name):
                for paragraph in paragraphs[1:]:
                    add(name, "micro", "detail", paragraph)

    return list({span["span_id"]: span for span in spans}.values())


def sample_balanced(candidates, target=250, max_per_paper=4, seed=SEED):
    rng = random.Random(seed)
    selected = []

    for domain in ("nlp", "cv"):
        for label in LABELS:
            papers = defaultdict(list)
            for row in candidates:
                if row["domain"] == domain and row["label"] == label:
                    papers[row["paper_id"]].append(row)

            paper_ids = sorted(papers)
            rng.shuffle(paper_ids)
            for paper_id in paper_ids:
                rng.shuffle(papers[paper_id])
                papers[paper_id] = papers[paper_id][:max_per_paper]

            chosen = []
            level = 0
            while len(chosen) < target:
                before = len(chosen)
                for paper_id in paper_ids:
                    if level < len(papers[paper_id]):
                        chosen.append(papers[paper_id][level])
                        if len(chosen) == target:
                            break
                if len(chosen) == before:
                    raise ValueError(f"Not enough examples for {domain}/{label}")
                level += 1
            selected.extend(chosen)

    return sorted(selected, key=lambda row: (row["domain"], row["label"], row["paper_id"], row["span_id"]))


def build(raw_path, output_path):
    candidates = []
    scanned = 0
    with gzip.open(raw_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            scanned += 1
            if not str(record.get("source", "")).startswith("s2orc"):
                continue
            text = record.get("text", "")
            title = clean(text.split("\n", 1)[0]) if text else ""
            domain = domain_from_title(title)
            if domain:
                candidates.extend(label_paper(record, domain))

    selected = sample_balanced(candidates)
    write_jsonl(output_path, selected)
    counts = Counter((row["domain"], row["label"]) for row in selected)
    write_json(
        output_path.with_name("dataset_manifest.json"),
        {
            "source": "allenai/peS2o validation shard (S2ORC-derived)",
            "source_url": DATA_URL,
            "seed": SEED,
            "records_scanned": scanned,
            "spans": len(selected),
            "papers": len({row["paper_id"] for row in selected}),
            "counts": {f"{d}/{l}": counts[d, l] for d in ("nlp", "cv") for l in LABELS},
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "pes2o.json.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "spans" / "spans.jsonl")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if args.download and not args.raw.exists():
        args.raw.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, args.raw)
    if not args.raw.exists():
        raise FileNotFoundError("Raw data not found. Run this script with --download.")
    build(args.raw, args.output)


if __name__ == "__main__":
    main()
