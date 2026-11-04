
from pathlib import Path
import csv
import json

ROOT = Path(r"C:\Users\User\Downloads\movies_script")

MOVIE_ID = "1179933"

# 1. Inspect metadata CSV files
print("\n========== METADATA ==========")

metadata_dir = ROOT / "movie_metadata"

for csv_file in metadata_dir.glob("*.csv"):
    print(f"\nFILE: {csv_file.name}")

    try:
        with csv_file.open(
            "r", encoding="utf-8-sig", errors="replace",
            newline=""
        ) as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                print(row[:12])
                if i == 2:
                    break
    except Exception as e:
        print("Error:", e)


# 2. Find files belonging to the same movie
print("\n========== MOVIE FILES ==========")

for path in ROOT.rglob(f"*{MOVIE_ID}*"):
    if path.is_file():
        print(path.relative_to(ROOT))


# 3. Inspect one annotated text file
print("\n========== ANNOTATION SAMPLE ==========")

annotation_dirs = [
    ROOT / "screenplay_data/data/manual_annotations",
    ROOT / "screenplay_data/data/BERT_annotations",
]

for directory in annotation_dirs:
    matches = list(directory.rglob(f"*{MOVIE_ID}*"))

    if matches:
        file = matches[0]
        print("\nFILE:", file.name)

        with file.open(
            "r", encoding="utf-8", errors="replace"
        ) as f:
            for i, line in enumerate(f):
                if i == 25:
                    break
                print(line, end="")

    else:
        print("\nNo matching file in:", directory)


# 4. Inspect the rule-based JSON structure
print("\n========== RULE-BASED JSON ==========")

json_dir = ROOT / "screenplay_data/data/rule_based_annotations"

matches = list(json_dir.rglob(f"*{MOVIE_ID}*.json"))

if matches:
    file = matches[0]
    print("FILE:", file.name)

    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print("Top-level type:", type(data).__name__)
    print("Number of segments:", len(data))

    for segment_index, segment in enumerate(data[:2]):
        print(f"\nSEGMENT {segment_index}")

        for item in segment[:3]:
            print(item)

else:
    print("No matching JSON file found.")
