
import csv
import os

OUTPUT = "data/challenge/hindi_rag_pilot.csv"

fields = [
    "id",
    "question",
    "context",
    "answer",
    "hallucination",
    "hallucinated_text",
    "start",
    "end",
    "category",
    "source",
    "reviewed",
]

categories = [
    "date",
    "number",
    "person",
    "location",
    "entity",
    "unsupported_detail",
    "paraphrase",
    "multiple_passages",
    "hindi_numerals",
    "other",
]

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

print("Created:", OUTPUT)
print("\nSuggested categories:")
for category in categories:
    print("-", category)

print("\nAdd 20 independently reviewed examples.")
