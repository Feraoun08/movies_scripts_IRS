from pathlib import Path

root = Path(
    r"C:\Users\User\Downloads\movies_script"
    r"\screenplay_data\data\raw_texts"
)

files = list(root.rglob("*"))

files = [f for f in files if f.is_file()]

print("Number of screenplay files:", len(files))

if files:
    sample = files[0]

    print("\nSample file:", sample)
    print("\nFirst 40 lines:\n")

    with sample.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 40:
                break
            print(line, end="")