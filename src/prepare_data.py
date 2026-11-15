
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import GroupShuffleSplit

INPUT = "data/raw/multiwiki_hindi"
OUTPUT = "data/processed/multiwiki_hindi"

data = load_from_disk(INPUT)

def group_id(row):
    # Keep examples with the same evidence and question together.
    content = json.dumps(
        {
            "question": row["question"],
            "context": row["context"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

groups = [group_id(row) for row in data]
indices = np.arange(len(data))

# First: 80% train, 20% temporary holdout.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, temp_idx = next(
    splitter.split(indices, groups=groups)
)

# Second: split the holdout into validation and test.
temp_groups = [groups[i] for i in temp_idx]

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)

dev_local, test_local = next(
    splitter2.split(temp_idx, groups=temp_groups)
)

dev_idx = temp_idx[dev_local]
test_idx = temp_idx[test_local]

splits = DatasetDict({
    "train": data.select(train_idx.tolist()),
    "validation": data.select(dev_idx.tolist()),
    "test": data.select(test_idx.tolist()),
})

# Verify no evidence/question group occurs in multiple splits.
group_sets = {
    name: {group_id(row) for row in ds}
    for name, ds in splits.items()
}

assert group_sets["train"].isdisjoint(group_sets["validation"])
assert group_sets["train"].isdisjoint(group_sets["test"])
assert group_sets["validation"].isdisjoint(group_sets["test"])

for name, ds in splits.items():
    counts = Counter(ds["hallucination"])
    print(f"{name}: {len(ds)} examples, labels={dict(counts)}")

splits.save_to_disk(OUTPUT)
print(f"\nSaved to {OUTPUT}")
