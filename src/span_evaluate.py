
import numpy as np
import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_PATH = "models/mmbert-hindi-full/best"
DATA_PATH = "data/processed/multiwiki_hindi"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

data = load_from_disk(DATA_PATH)
test_data = data["test"]

tokenized_test = load_from_disk(
    "data/processed/mmbert_tokenized"
)["test"]

def get_first_text(row):
    question = row["question"]
    answer = row["answer"]
    context = "\n".join(row["context"])

    prefix = "Question: " + question + "\nContext: "
    answer_text = "Answer: " + answer

    answer_tokens = tokenizer(
        answer_text,
        add_special_tokens=False
    )["input_ids"]

    prefix_tokens = tokenizer(
        prefix,
        add_special_tokens=False
    )["input_ids"]

    available = 1024 - len(answer_tokens) - 3

    if available < len(prefix_tokens):
        raise ValueError("Question or answer too long")

    context_tokens = tokenizer(
        context,
        add_special_tokens=False
    )["input_ids"]

    context_tokens = context_tokens[
        :available - len(prefix_tokens)
    ]

    truncated_context = tokenizer.decode(
        context_tokens,
        skip_special_tokens=True
    )

    return prefix + truncated_context


def predict_spans(row):
    answer = row["answer"]
    answer_prefix = "Answer: "
    answer_text = answer_prefix + answer

    first_text = get_first_text(row)

    # Same paired-input strategy as training.
    encoded = tokenizer(
        first_text,
        answer_text,
        max_length=1024,
        truncation="only_first",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    saved_ids = tokenized_test[row["_index"]]["input_ids"]
    assert encoded["input_ids"][0].tolist() == saved_ids, (
        f"Tokenization mismatch at index {row['_index']}"
    )

    sequence_ids = encoded.sequence_ids(0)
    offsets = encoded["offset_mapping"][0].tolist()

    inputs = {
        k: v.to(device)
        for k, v in encoded.items()
        if k != "offset_mapping"
    }

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = logits.argmax(dim=-1)[0].cpu().tolist()

    predicted_spans = []
    prefix_length = len(answer_prefix)

    for label, sequence_id, (start, end) in zip(
        predictions, sequence_ids, offsets
    ):
        if sequence_id != 1 or end <= prefix_length or label != 1:
            continue

        # Convert offsets from prefixed answer to original answer.
        start = max(0, start - prefix_length)
        end = min(len(answer), end - prefix_length)

        if start < end:
            predicted_spans.append((start, end))

    return merge_spans(predicted_spans)


def merge_spans(spans):
    if not spans:
        return []

    spans.sort()
    merged = [list(spans[0])]

    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def get_true_spans(row):
    spans = []

    for item in row["hallucinated_labels"]:
        start = item["start"]
        end = item["end"]
        spans.append((start, end))

    return merge_spans(spans)


def spans_to_mask(spans, length):
    mask = np.zeros(length, dtype=bool)

    for start, end in spans:
        mask[start:end] = True

    return mask


def calculate_iou(true_spans, pred_spans, answer_length):
    true_mask = spans_to_mask(true_spans, answer_length)
    pred_mask = spans_to_mask(pred_spans, answer_length)

    intersection = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()

    if union == 0:
        return 1.0

    return intersection / union


def highlight_answer(answer, spans):
    if not spans:
        return answer

    result = ""
    previous = 0

    for start, end in spans:
        result += answer[previous:start]
        result += "【" + answer[start:end] + "】"
        previous = end

    result += answer[previous:]
    return result


print("\nEvaluating character spans...")

ious = []
positive_ious = []
total_intersection = 0
total_union = 0
skipped = 0

examples = []

for i, row in enumerate(test_data):
    # Positive examples without span annotations cannot be scored.
    if row["hallucination"] and not row["hallucinated_labels"]:
        skipped += 1
        continue

    true_spans = get_true_spans(row)
    row["_index"] = i
    pred_spans = predict_spans(row)

    answer = row["answer"]

    iou = calculate_iou(
        true_spans,
        pred_spans,
        len(answer)
    )

    true_mask = spans_to_mask(true_spans, len(answer))
    pred_mask = spans_to_mask(pred_spans, len(answer))

    total_intersection += np.logical_and(
        true_mask, pred_mask
    ).sum()

    total_union += np.logical_or(
        true_mask, pred_mask
    ).sum()

    ious.append(iou)

    if row["hallucination"]:
        positive_ious.append(iou)

    if len(examples) < 5 and row["hallucination"]:
        examples.append((row, true_spans, pred_spans, iou))

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(test_data)}")

print("\n========== SPAN RESULTS ==========")
print("Evaluated examples:", len(ious))
print("Skipped examples:", skipped)
print("Mean IoU (all):", round(float(np.mean(ious)), 4))
print("Mean IoU (hallucinated):", round(float(np.mean(positive_ious)), 4))

micro_iou = (
    total_intersection / total_union
    if total_union else 1.0
)

print("Micro character IoU:", round(float(micro_iou), 4))

print("\n========== EXAMPLES ==========")

for row, true_spans, pred_spans, iou in examples:
    print("\nQuestion:", row["question"])
    print("Original answer:", row["answer"])
    print("Ground truth:", highlight_answer(row["answer"], true_spans))
    print("Prediction:", highlight_answer(row["answer"], pred_spans))
    print("IoU:", round(float(iou), 4))

print("\nSpan evaluation completed!")
