"""Quick end-to-end test on the real dataset.

    python try_pipeline.py "C:\\path\\to\\movies_script" --movies 30 --query "detective killer snow"
"""
import argparse
import random
import statistics
import sys
from collections import Counter

from irsearch.chunker import chunk_scene
from irsearch.config import Config, ParserConfig
from irsearch.dataset import load_dataset, read_text_file
from irsearch.parser import parse_screenplay
from irsearch.pipeline import build, load
from irsearch.preprocess import Preprocessor

ap = argparse.ArgumentParser()
ap.add_argument("data_dir")
ap.add_argument("--movies", type=int, default=30, help="how many movies to test")
ap.add_argument("--out", default="out_test")
ap.add_argument("--query", default="detective killer snow")
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

cfg = Config()
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# PHASE 0 - dataset
print("\n== PHASE 0: dataset ==")
data = load_dataset(a.data_dir)
ids = data.with_raw_text()
print(f"  screenplays: {len(ids)}   metadata rows: {len(data.metadata)}")
check("screenplays found", len(ids) > 0)
if not ids:
    sys.exit("No screenplay found: check the folder path (raw_texts/...).")
matched = sum(i in data.metadata for i in ids)
check("metadata matched by IMDb id", matched / len(ids) >= 0.8, f"{matched}/{len(ids)}")
sample = random.Random(a.seed).sample(ids, min(a.movies, len(ids)))

# PHASE 1 - parsing
print("\n== PHASE 1: parsing ==")
lossless = ParserConfig(drop_page_artifacts=False)
parsed = {}
lost, no_boundary, no_dialogue = [], [], []
kinds = Counter()
for i in sample:
    text = read_text_file(data.bundles[i].files["raw_text"])
    scenes = parse_screenplay(text, cfg.parser)
    parsed[i] = scenes
    words = [w for s in parse_screenplay(text, lossless) for l in s.lines() for w in l.text.split()]
    if words != text.split():
        lost.append(data.title(i))
    if not scenes or not scenes[0].has_boundary:
        no_boundary.append(data.title(i))
    k = Counter(e.kind for s in scenes for e in s.elements)
    kinds.update(k)
    if not k["speaker"]:
        no_dialogue.append(data.title(i))
n_scenes = [len(s) for s in parsed.values()]
print(f"  scenes per movie: median {statistics.median(n_scenes):.0f} (min {min(n_scenes)}, max {max(n_scenes)})")
print(f"  elements: {dict(kinds)}")
check("no text lost or reordered", not lost, f"{len(lost)} movies differ {lost[:3]}")
check("reliable scene boundaries", len(no_boundary) <= 0.1 * len(sample), f"{len(no_boundary)} without {no_boundary[:3]}")
check("dialogue detected", len(no_dialogue) <= 0.1 * len(sample), f"{len(no_dialogue)} without {no_dialogue[:3]}")
first = next((s for sc in parsed.values() for s in sc if s.int_ext), None)
if first:
    print(f"  example heading: {first.heading!r} -> {first.int_ext} | {first.locations} | {first.time_of_day}")

# PHASE 2 - preprocessing
print("\n== PHASE 2: preprocessing ==")
pre = Preprocessor(cfg.preprocess)
demo = next((s for sc in parsed.values() for s in sc if s.n_words > 40), None)
if demo:
    print("  original :", " ".join(demo.text.split()[:25]))
    print("  tokens   :", pre.process(" ".join(demo.text.split()[:25])))

# PHASE 3 - chunking
print("\n== PHASE 3: chunking ==")
sizes, split_sizes, bad_rebuild, too_big = [], [], 0, 0
c = cfg.chunk
for scenes in parsed.values():
    for sc in scenes:
        chunks = chunk_scene(sc, "m", cfg)
        words = sc.text.split()
        if not chunks:
            continue
        rebuilt = chunks[0].text.split()
        for prev, nxt in zip(chunks, chunks[1:]):
            rebuilt += nxt.text.split()[c.overlap_words:]
        bad_rebuild += rebuilt != words
        for k in chunks:
            sizes.append(k.n_words)
            if len(chunks) > 1:
                split_sizes.append(k.n_words)
            too_big += k.n_words > (c.max_words if len(chunks) > 1 else c.single_chunk_max_words)
print(f"  chunks: {len(sizes)}   median {statistics.median(sizes):.0f} words (min {min(sizes)}, max {max(sizes)})")
if split_sizes:
    print(f"  split chunks: median {statistics.median(split_sizes):.0f}, min {min(split_sizes)}, max {max(split_sizes)}")
check("scene rebuilt exactly from its chunks", bad_rebuild == 0, f"{bad_rebuild} scenes differ")
check("chunk size within limits", too_big == 0, f"{too_big} chunks too big")

# PHASE 4 - build + inverted index
print("\n== PHASE 4: index ==")
report = build(a.data_dir, a.out, cfg, limit=a.movies, log=lambda m: None)
index, store, pre, _ = load(a.out)
st = index.stats()
print(f"  movies {report['movies']}, scenes {report['scenes']}, chunks {report['chunks']}")
print(f"  vocabulary {st['vocabulary_size']:,}, postings {st['total_postings']:,}, avg length {st['avg_doc_length']}")
check("doc table matches readable store", index.N == store.counts()["chunks"] == report["chunks"])
top = sorted(index.postings, key=index.df, reverse=True)[:12]
print("  most frequent terms (df):", [(t, index.df(t)) for t in top])
check("no screenplay noise in the index", not ({"int", "ext", "contd"} & set(index.postings)))

terms = pre.process(a.query)
print(f"\n  query {a.query!r} -> terms {terms}")
for t in terms:
    print(f"    df('{t}') = {index.df(t)}")
if terms and all(index.df(t) for t in terms):
    sets = [{d for d, _ in index.iter_postings(t)} for t in terms]
    hits = set.intersection(*sets)
    scored = sorted(hits, key=lambda d: -sum(dict(index.iter_postings(t)).get(d, 0) for t in terms))[:3]
    print(f"  chunks containing all terms: {len(hits)}")
    for row in store.get_chunks(scored):
        print(f"   - {row['title']} ({row['year']}) scene {row['scene_idx']} part {row['part_idx'] + 1}/{row['n_parts']}")
        print("     " + row["text"].replace("\n", " ")[:160] + "...")
else:
    print("  (some query terms are absent from this small index: try --query with other words or more --movies)")

print(f"\n== RESULT: {sum(results)}/{len(results)} checks passed ==")
sys.exit(0 if all(results) else 1)