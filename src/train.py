
import os
import numpy as np
import torch

from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "jhu-clsp/mmBERT-small"
DATA_PATH = "data/processed/mmbert_tokenized"
OUTPUT_DIR = "models/mmbert-hindi-full"

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected. Training stopped.")

print("GPU:", torch.cuda.get_device_name(0))

# Load our already-tokenized dataset.
dataset = load_from_disk(DATA_PATH)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))

# Load the pretrained model with a new 2-class classification head.
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={0: "SUPPORTED", 1: "HALLUCINATED"},
    label2id={"SUPPORTED": 0, "HALLUCINATED": 1},
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Reduce GPU memory use during training.
model.gradient_checkpointing_enable()

# Pad each batch only to its longest sequence.
collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predicted_labels = np.argmax(predictions, axis=-1)

    # Ignore tokens labeled -100.
    mask = labels != -100

    y_true = labels[mask]
    y_pred = predicted_labels[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,

    # Small physical batch for 8 GB VRAM.
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    # Effective training batch size = 1 x 8.
    gradient_accumulation_steps=8,

    fp16=True,
    gradient_checkpointing=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    warmup_steps=240,

    logging_steps=50,
    save_total_limit=2,

    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
    compute_metrics=compute_metrics,
)


print("\nStarting training...")

trainer.train()

print("\nFinal validation metrics:")
metrics = trainer.evaluate()
print(metrics)

print("\nSaving best model...")
trainer.save_model(OUTPUT_DIR + "/best")
tokenizer.save_pretrained(OUTPUT_DIR + "/best")

print("\nTraining completed!")
