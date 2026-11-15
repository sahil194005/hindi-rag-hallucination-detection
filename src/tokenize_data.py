
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from collections import Counter

MODEL_NAME = "jhu-clsp/mmBERT-small"
MAX_LENGTH = 1024

print("Loading dataset...")
data = load_from_disk("data/processed/multiwiki_hindi")

print("Loading mmBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


def tokenize_example(row):
    question = row["question"]
    answer = row["answer"]
    context = "\n".join(row["context"])

    prefix = "Question: " + question + "\nContext: "
    answer_text = "Answer: " + answer
    answer_prefix_length = len("Answer: ")

    # Tokenize the answer without special tokens to measure its length.
    answer_tokens = tokenizer(
        answer_text,
        add_special_tokens=False
    )["input_ids"]

    # Reserve space for the complete question and answer.
    prefix_tokens = tokenizer(
        prefix,
        add_special_tokens=False
    )["input_ids"]

    available = MAX_LENGTH - len(answer_tokens) - 3

    if available < len(prefix_tokens):
        raise ValueError("Question or answer too long")

    # Truncate the evidence using its token IDs.
    context_tokens = tokenizer(
        context,
        add_special_tokens=False
    )["input_ids"]

    context_tokens = context_tokens[:available - len(prefix_tokens)]

    truncated_context = tokenizer.decode(
        context_tokens,
        skip_special_tokens=True
    )

    first_text = prefix + truncated_context

    # Use the standard paired-text tokenizer API.
    encoding = tokenizer(
        first_text,
        answer_text,
        max_length=MAX_LENGTH,
        truncation="only_first",
        padding=False,
        return_offsets_mapping=True
    )

    labels = []

    # sequence_ids: 0 = question/context, 1 = answer, None = special
    sequence_ids = encoding.sequence_ids()

    for (start, end), sequence_id in zip(
        encoding["offset_mapping"],
        sequence_ids
    ):
        if sequence_id != 1:
            labels.append(-100)
            continue

        # Ignore the "Answer: " prefix.
        if end <= answer_prefix_length:
            labels.append(-100)
            continue

        # Convert offsets to positions within the original answer.
        start -= answer_prefix_length
        end -= answer_prefix_length

        label = 0

        for span in row["hallucinated_labels"]:
            if start < span["end"] and end > span["start"]:
                label = 1
                break

        labels.append(label)

    assert len(encoding["input_ids"]) == len(labels)
    assert len(encoding["input_ids"]) <= MAX_LENGTH

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels
    }

print("Tokenizing dataset...")

processed = DatasetDict()

for split in ["train", "validation", "test"]:
    processed[split] = data[split].map(
        tokenize_example,
        remove_columns=data[split].column_names,
        desc=f"Tokenizing {split}"
    )

    lengths = [len(x) for x in processed[split]["input_ids"]]
    counts = Counter(
        label
        for row in processed[split]["labels"]
        for label in row
    )

    print(f"\n{split}:")
    print("Examples:", len(processed[split]))
    print("Longest sequence:", max(lengths))
    print("Average length:", round(sum(lengths) / len(lengths), 1))
    print("Token labels:", dict(counts))

print("\nSaving tokenized dataset...")
processed.save_to_disk("data/processed/mmbert_tokenized")

print("\nDONE!")
