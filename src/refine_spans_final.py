
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

raw = load_from_disk("data/processed/multiwiki_hindi")["test"]
saved = load_from_disk("data/processed/mmbert_tokenized")["test"]

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

    # Verify exact consistency with the original tokenized data.
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
        predictions = model(**inputs).logits.argmax(-1)[0].tolist()

    spans = []
    prefix_length = len("Answer: ")

    for label, sequence_id, (start, end) in zip(
        predictions, sequence_ids, offsets
    ):
        if sequence_id != 1 or label != 1:
            continue

        if end <= prefix_length:
            continue

        start = max(0, start - prefix_length)
        end = min(len(row["answer"]), end - prefix_length)

        if start < end:
            spans.append((start, end))

    return merge_spans(spans)


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

results = []
skipped = 0

for i, row in enumerate(raw):
    if row["hallucination"] and not row["hallucinated_labels"]:
        skipped += 1
        continue

    answer = row["answer"]
    true_spans = get_true_spans(row)

    baseline_spans = predict_spans(row, i)
    refined_spans = refine_spans(answer, baseline_spans)

    baseline_iou = calculate_iou(
        answer, true_spans, baseline_spans
    )

    refined_iou = calculate_iou(
        answer, true_spans, refined_spans
    )

    results.append({
        "index": i,
        "hallucination": int(row["hallucination"]),
        "question": row["question"],
        "answer": answer,
        "baseline_iou": baseline_iou,
        "refined_iou": refined_iou,
        "change": refined_iou - baseline_iou,
        "ground_truth": highlight(answer, true_spans),
        "baseline": highlight(answer, baseline_spans),
        "refined": highlight(answer, refined_spans),
    })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(raw)}")


positive = [r for r in results if r["hallucination"] == 1]

print("\n========== VALIDATION RESULTS ==========")
print("Evaluated examples:", len(results))
print("Skipped missing annotations:", skipped)

for name, rows in [
    ("All examples", results),
    ("Hallucinated examples", positive),
]:
    baseline = np.mean([r["baseline_iou"] for r in rows])
    refined = np.mean([r["refined_iou"] for r in rows])

    print(f"\n{name}:")
    print("Baseline mean IoU:", round(float(baseline), 4))
    print("Refined mean IoU:", round(float(refined), 4))
    print("Difference:", round(float(refined - baseline), 4))


improved = sum(r["change"] > 1e-9 for r in results)
worsened = sum(r["change"] < -1e-9 for r in results)
unchanged = len(results) - improved - worsened

print("\nImproved:", improved)
print("Worsened:", worsened)
print("Unchanged:", unchanged)

output_path = "results/span_refinement_test.csv"

with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n========== TOP IMPROVEMENTS ==========")

for r in sorted(results, key=lambda x: x["change"], reverse=True)[:5]:
    print("\nQuestion:", r["question"])
    print("Ground truth:", r["ground_truth"])
    print("Baseline:", r["baseline"])
    print("Refined:", r["refined"])
    print("IoU change:", round(r["change"], 4))

print("\n========== TOP REGRESSIONS ==========")

for r in sorted(results, key=lambda x: x["change"])[:5]:
    print("\nQuestion:", r["question"])
    print("Ground truth:", r["ground_truth"])
    print("Baseline:", r["baseline"])
    print("Refined:", r["refined"])
    print("IoU change:", round(r["change"], 4))

print("\nSaved:", output_path)

print("\n========== FINAL TEST COMPARISON ==========")

positive = [
    r for r in results
    if r["hallucination"] == 1
]

for name, rows in [
    ("All test examples", results),
    ("Hallucinated test examples", positive),
]:
    baseline = np.mean([
        r["baseline_iou"] for r in rows
    ])

    refined = np.mean([
        r["refined_iou"] for r in rows
    ])

    print(f"\n{name}")
    print(f"Original mean IoU: {baseline:.4f}")
    print(f"V2 mean IoU:       {refined:.4f}")
    print(f"Improvement:       {refined - baseline:+.4f}")


improved = sum(r["change"] > 1e-9 for r in results)
worsened = sum(r["change"] < -1e-9 for r in results)
unchanged = len(results) - improved - worsened

print("\nImproved examples:", improved)
print("Worsened examples:", worsened)
print("Unchanged examples:", unchanged)


def micro_iou(rows, column):
    # Compute aggregate character-level IoU.
    # Each row must contain the raw spans or masks;
    # mean of individual IoUs is not micro IoU.
    raise NotImplementedError(
        "Micro IoU requires aggregate character intersections "
        "and unions. Do not calculate it by averaging IoUs."
    )

print("Done!")
