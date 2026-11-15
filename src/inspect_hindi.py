
from datasets import load_dataset
from collections import Counter

DATASET_ID = "alexandrainst/multi-wiki-qa-synthetic-hallucinations"

print("Loading Hindi dataset...")

data = load_dataset(DATASET_ID, "hi", split="train")

print("\nTotal rows:", len(data))
print("Columns:", data.column_names)

print("\nHallucination distribution:")
print(Counter(data["hallucination"]))

print("\nFirst 3 examples:")

for i in range(min(3, len(data))):
    row = data[i]

    print("\n" + "=" * 70)
    print("Example:", i)
    print("Question:", row["question"])
    print("Answer:", row["answer"])
    print("Hallucination:", row["hallucination"])
    print("Hallucinated parts:", row["hallucinated_parts"])
    print("Span labels:", row["hallucinated_labels"])

    context = row["context"]
    print("Context preview:", str(context)[:400])

    # Check whether the annotated character spans match the answer.
    for span in row["hallucinated_labels"]:
        start = span["start"]
        end = span["end"]
        print(
            "Annotated text:",
            repr(row["answer"][start:end])
        )

print("\nSaving a local copy...")
data.save_to_disk("data/raw/multiwiki_hindi")

print("Done!")
