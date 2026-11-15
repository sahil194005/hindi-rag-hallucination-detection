
import numpy as np

from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_PATH = "models/mmbert-hindi-full/best"

raw = load_from_disk("data/processed/multiwiki_hindi")["test"]
tokenized = load_from_disk("data/processed/mmbert_tokenized")["test"]

assert len(raw) == len(tokenized)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

args = TrainingArguments(
    output_dir="results/evaluation_tmp",
    per_device_eval_batch_size=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
)

print("Evaluating on test dataset...")
output = trainer.predict(tokenized)

predictions = np.argmax(output.predictions, axis=-1)
labels = output.label_ids

# Identify hallucinated examples with no span annotations.
missing = [
    i for i, row in enumerate(raw)
    if row["hallucination"] and not row["hallucinated_labels"]
]

print("\nExamples with missing spans:", len(missing))

# TOKEN-LEVEL METRICS
# Ignore examples without valid positive span annotations.
valid_rows = [
    i for i in range(len(raw))
    if i not in set(missing)
]

true_tokens = []
pred_tokens = []

for i in valid_rows:
    mask = labels[i] != -100
    true_tokens.extend(labels[i][mask])
    pred_tokens.extend(predictions[i][mask])

p, r, f1, _ = precision_recall_fscore_support(
    true_tokens,
    pred_tokens,
    average="binary",
    zero_division=0,
)

print("\nTOKEN-LEVEL RESULTS")
print("Precision:", round(p, 4))
print("Recall:", round(r, 4))
print("F1:", round(f1, 4))

# EXAMPLE-LEVEL METRICS
# Use original binary labels, including missing-span examples.
example_true = [
    int(row["hallucination"])
    for row in raw
]

example_pred = []

for i in range(len(raw)):
    mask = labels[i] != -100
    predicted_hallucination = np.any(
        predictions[i][mask] == 1
    )
    example_pred.append(int(predicted_hallucination))

p, r, f1, _ = precision_recall_fscore_support(
    example_true,
    example_pred,
    average="binary",
    zero_division=0,
)

print("\nEXAMPLE-LEVEL RESULTS")
print("Accuracy:", round(
    accuracy_score(example_true, example_pred), 4
))
print("Precision:", round(p, 4))
print("Recall:", round(r, 4))
print("F1:", round(f1, 4))

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    example_true,
    example_pred,
    labels=[0, 1],
))

print("\nEvaluation completed!")
