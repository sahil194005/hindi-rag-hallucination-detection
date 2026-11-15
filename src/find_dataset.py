
from huggingface_hub import HfApi

api = HfApi()

search_terms = [
    "MultiWikiQHalluA",
    "multi-wiki-hallu",
    "multiwiki"
]

for term in search_terms:
    print(f"\nSearching for: {term}")
    print("-" * 50)

    datasets = list(api.list_datasets(search=term, limit=20))

    if not datasets:
        print("No datasets found.")
        continue

    for dataset in datasets:
        print(dataset.id)
