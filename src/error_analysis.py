
import csv
import numpy as np

from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "models/mmbert-hindi-full/best"

raw = load_from_disk(
    "data/processed/multiwiki_hindi"
)["validation"]

tokenized = load_from_disk(
    "data/processed/mmbert_tokenized"
)["validation"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

args = TrainingArguments(
    output_dir="results/error_analysis_tmp",
    per_device_eval_batch_size=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
)

print("Running validation predictions...")
output = trainer.predict(tokenized)

predictions = np.argmax(output.predictions, axis=-1)
labels = output.label_ids

results = []

for i, row in enumerate(raw):
    valid = labels[i] != -100

    predicted_tokens = predictions[i][valid]
    true_tokens = labels[i][valid]

    actual = int(row["hallucination"])
    predicted = int(np.any(predicted_tokens == 1))

    missing_spans = (
        actual == 1 and not row["hallucinated_labels"]
    )

    # Token-level overlap, not character-level span IoU.
    true_mask = true_tokens == 1
    pred_mask = predicted_tokens == 1

    intersection = np.logical_and(
        true_mask, pred_mask
    ).sum()

    union = np.logical_or(
        true_mask, pred_mask
    ).sum()

    token_iou = (
        float(intersection / union)
        if union > 0 else 1.0
    )

    if actual == 0 and predicted == 1:
        error_type = "False Positive"

    elif actual == 1 and predicted == 0:
        error_type = "False Negative"

    elif missing_spans:
        error_type = "Missing Span Annotation"

    elif actual == 1 and predicted == 1 and token_iou < 0.5:
        error_type = "Poor Localization"

    else:
        error_type = "Correct"

    results.append({
        "index": i,
        "error_type": error_type,
        "actual": actual,
        "predicted": predicted,
        "token_iou": round(token_iou, 4),
        "question": row["question"],
        "context": "\n".join(row["context"]),
        "answer": row["answer"],
        "true_spans": str(row["hallucinated_labels"]),
    })

results.sort(
    key=lambda x: (
        x["error_type"] == "Correct",
        x["token_iou"]
    )
)

output_path = "results/mmbert_error_analysis.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

print("\nERROR ANALYSIS SUMMARY")

for error_type in [
    "False Positive",
    "False Negative",
    "Poor Localization",
    "Missing Span Annotation",
    "Correct",
]:
    count = sum(
        x["error_type"] == error_type
        for x in results
    )
    print(f"{error_type}: {count}")

print("\n10 LOWEST TOKEN IoU EXAMPLES")

poor = [
    x for x in results
    if x["actual"] == 1
    and x["error_type"] != "Missing Span Annotation"
]

poor.sort(key=lambda x: x["token_iou"])

for item in poor[:10]:
    print("\nIndex:", item["index"])
    print("Error:", item["error_type"])
    print("Token IoU:", item["token_iou"])
    print("Question:", item["question"])
    print("Answer:", item["answer"])
    print("True spans:", item["true_spans"])

print("\nSaved:", output_path)
