
import csv
import re
import unicodedata

import numpy as np
import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "models/mmbert-hindi-full/best"
MAX_LENGTH = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

raw = load_from_disk("data/processed/multiwiki_hindi")["validation"]
saved = load_from_disk("data/processed/mmbert_tokenized")["validation"]


def merge_spans(spans):
    spans = sorted(spans)
    merged = []

    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def is_word_char(char):
    # L = letters, M = combining marks, N = numbers.
    # Includes Devanagari vowel signs and other combining marks.
    return unicodedata.category(char)[0] in ("L", "M", "N")



def refine_spans(answer, spans):
    refined = []

    for start, end in spans:
        # Remove whitespace at the edges.
        while start < end and answer[start].isspace():
            start += 1

        while end > start and answer[end - 1].isspace():
            end -= 1

        if start >= end:
            continue

        # Expand only if the span begins inside a word.
        if is_word_char(answer[start]):
            while start > 0 and is_word_char(answer[start - 1]):
                start -= 1

        # Expand only if the span ends inside a word.
        if is_word_char(answer[end - 1]):
            while end < len(answer) and is_word_char(answer[end]):
                end += 1

        refined.append((start, end))

    return merge_spans(refined)


def confidence_refine(answer, spans, confidences, threshold):
    result = []

    for span, confidence in zip(spans, confidences):
        if confidence >= threshold:
            # High-confidence hallucination:
            # expand to Hindi word boundaries.
            result.extend(refine_spans(answer, [span]))
        else:
            # Low-confidence hallucination:
            # preserve the original predicted boundaries.
            result.append(span)

    return merge_spans(result)

def prepare_input(row):
    question = row["question"]
    answer = row["answer"]
    context = "\n".join(row["context"])

    prefix = "Question: " + question + "\nContext: "
    answer_text = "Answer: " + answer

    answer_tokens = tokenizer(
        answer_text, add_special_tokens=False
    )["input_ids"]

    prefix_tokens = tokenizer(
        prefix, add_special_tokens=False
    )["input_ids"]

    available = MAX_LENGTH - len(answer_tokens) - 3

    if available < len(prefix_tokens):
        raise ValueError("Question or answer too long")

    context_tokens = tokenizer(
        context, add_special_tokens=False
    )["input_ids"]

    context_tokens = context_tokens[
        :available - len(prefix_tokens)
    ]

    truncated_context = tokenizer.decode(
        context_tokens, skip_special_tokens=True
    )

    return tokenizer(
        prefix + truncated_context,
        answer_text,
        max_length=MAX_LENGTH,
        truncation="only_first",
        return_offsets_mapping=True,
        return_tensors="pt",
    )



def predict_spans(row, index):
    encoded = prepare_input(row)

    assert encoded["input_ids"][0].tolist() == saved[index]["input_ids"], (
        f"Tokenization mismatch at validation example {index}"
    )

    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded["offset_mapping"][0].tolist()

    inputs = {
        k: v.to(device)
        for k, v in encoded.items()
        if k != "offset_mapping"
    }

    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        predictions = logits.argmax(-1).tolist()
        hallucination_probs = probabilities[:, 1].tolist()

    prefix_length = len("Answer: ")
    token_spans = []

    for label, prob, sequence_id, (start, end) in zip(
        predictions, hallucination_probs, sequence_ids, offsets
    ):
        if sequence_id != 1 or label != 1:
            continue

        if end <= prefix_length:
            continue

        start = max(0, start - prefix_length)
        end = min(len(row["answer"]), end - prefix_length)

        if start < end:
            token_spans.append((start, end, prob))

    merged = merge_spans([
        (start, end) for start, end, _ in token_spans
    ])

    # Mean positive-token probability for each merged span.
    confidences = []

    for start, end in merged:
        probs = [
            prob
            for a, b, prob in token_spans
            if a < end and b > start
        ]
        confidences.append(float(np.mean(probs)))

    return merged, confidences

def get_true_spans(row):
    return merge_spans([
        (item["start"], item["end"])
        for item in row["hallucinated_labels"]
    ])


def calculate_iou(answer, true_spans, pred_spans):
    length = len(answer)

    true_mask = np.zeros(length, dtype=bool)
    pred_mask = np.zeros(length, dtype=bool)

    for start, end in true_spans:
        true_mask[start:end] = True

    for start, end in pred_spans:
        pred_mask[start:end] = True

    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()

    return float(intersection / union) if union else 1.0


def highlight(answer, spans):
    result = ""
    previous = 0

    for start, end in spans:
        result += answer[previous:start]
        result += "【" + answer[start:end] + "】"
        previous = end

    return result + answer[previous:]


print("Evaluating baseline vs Hindi word-boundary refinement...")


THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

print("Running V3 confidence-aware refinement...")

results = []
skipped = 0

for i, row in enumerate(raw):
    if row["hallucination"] and not row["hallucinated_labels"]:
        skipped += 1
        continue

    answer = row["answer"]
    true_spans = get_true_spans(row)

    baseline_spans, confidences = predict_spans(row, i)
    v2_spans = refine_spans(answer, baseline_spans)

    record = {
        "index": i,
        "hallucination": int(row["hallucination"]),
        "question": row["question"],
        "answer": answer,
        "ground_truth": highlight(answer, true_spans),
        "baseline": highlight(answer, baseline_spans),
        "v2": highlight(answer, v2_spans),
        "baseline_iou": calculate_iou(
            answer, true_spans, baseline_spans
        ),
        "v2_iou": calculate_iou(
            answer, true_spans, v2_spans
        ),
    }

    for threshold in THRESHOLDS:
        refined = confidence_refine(
            answer, baseline_spans, confidences, threshold
        )

        key = f"v3_{threshold:.2f}"
        record[key + "_iou"] = calculate_iou(
            answer, true_spans, refined
        )
        record[key + "_spans"] = highlight(answer, refined)

    results.append(record)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(raw)}")


positive = [r for r in results if r["hallucination"] == 1]

print("\n========== VALIDATION RESULTS ==========")
print("Evaluated:", len(results))
print("Missing annotations skipped:", skipped)

methods = [
    ("Original", "baseline_iou"),
    ("V2", "v2_iou"),
]

for threshold in THRESHOLDS:
    methods.append((
        f"V3 threshold {threshold:.2f}",
        f"v3_{threshold:.2f}_iou"
    ))

print("\nMethod                     Positive mean IoU   All mean IoU")
print("-" * 65)

for name, key in methods:
    pos_mean = np.mean([r[key] for r in positive])
    all_mean = np.mean([r[key] for r in results])

    print(f"{name:27s} {pos_mean:.4f}             {all_mean:.4f}")


print("\n========== V3 VS V2 ==========")

for threshold in THRESHOLDS:
    key = f"v3_{threshold:.2f}_iou"

    improved = sum(
        r[key] > r["v2_iou"] + 1e-9 for r in results
    )
    worsened = sum(
        r[key] < r["v2_iou"] - 1e-9 for r in results
    )
    unchanged = len(results) - improved - worsened

    print(
        f"Threshold {threshold:.2f}: "
        f"Improved={improved}, "
        f"Worsened={worsened}, "
        f"Unchanged={unchanged}"
    )


output_path = "results/span_refinement_v3.csv"

with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", output_path)
print("Done!")
