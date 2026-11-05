"""Dataset access for the Kaggle Movie Scripts Corpus.

Pipeline phases:
    1. Configuration   constants and column mappings
    2. Value parsing   IMDb ids, years, numbers, genres, titles
    3. Discovery       scan the folders, group files per movie
    4. Metadata        movie_meta_data.csv -> per-movie fields
    5. Loading         build the Dataset, select the text source, iterate movies
    6. Annotations     light readers for BERT / rule-based JSON (not yet in the pipeline)
    7. Inspection      human-readable report on the corpus

Rules:
    - Raw screenplay TXT files are the primary text source.
    - Files and metadata are matched by IMDb id (int), never by title.
    - Annotations are discovered and readable, but not merged into the text.
    - movie_meta_data.csv has no genre column, so `genres` stays empty.
"""
from __future__ import annotations

import csv
import io
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import ParserConfig
from .models import MovieRecord
from .parser import _indent, cue_name, is_heading, parse_screenplay


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
COMPONENT_DIRS = {
    "raw_texts": "raw_text",
    "raw_text_lemmas": "lemmas",
    "bert_annotations": "bert",
    "rule_based_annotations": "rule_json",
    "manual_annotations": "manual",
    "movie_character_texts": "characters",
}

COMPONENT_LABELS = {
    "raw_text": "raw_texts",
    "lemmas": "raw_text_lemmas",
    "bert": "BERT_annotations",
    "rule_json": "rule_based_annotations",
    "manual": "manual_annotations",
    "characters": "movie_character_texts",
}

META_FILES = {
    "movie_meta_data.csv": "movies",
    "metacritic_reviews_cut_versions.csv": "reviews",
    "screenplay_awards.csv": "awards",
}

STEM_RE = re.compile(r"^(?P<title>.+?)_(?P<id>\d{4,8})(?:_lemmas|_anno)?$")

JSON_PARSER_AVAILABLE = False


@dataclass
class DatasetConfig:
    id_keys: tuple[str, ...] = ("imdbid", "imdb_id", "tconst")
    title_keys: tuple[str, ...] = ("title", "movie_title", "name")
    year_keys: tuple[str, ...] = ("year", "release_year")
    genre_keys: tuple[str, ...] = ("genre", "genres")
    imdb_rating_keys: tuple[str, ...] = ("imdb_user_rating", "imdb_rating", "imdbrating", "rating")
    metascore_keys: tuple[str, ...] = ("metascore", "metacritic_score")
    votes_keys: tuple[str, ...] = ("number_of_imdb_user_votes", "imdb_votes", "votes", "number_of_votes")


# =============================================================================
# 2. VALUE PARSING
# =============================================================================
def read_text_file(path: Path) -> str:
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def read_csv(path: Path) -> list[dict]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    return list(csv.DictReader(io.StringIO(read_text_file(path))))


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")


def _pick(row: dict, keys: tuple[str, ...]):
    """First non-empty value among the candidate column names."""
    normalized = {_norm_key(k): v for k, v in row.items() if k is not None}
    for key in keys:
        value = normalized.get(key)
        if value is not None and str(value).strip() not in ("", "nan", "NaN", "N/A", "n/a", "None"):
            return value
    return None


def parse_imdb_id(value) -> int | None:
    """'0038190' / '38190' / 'tt0038190' -> 38190"""
    if value is None:
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def format_imdb_id(imdb_id: int) -> str:
    return f"tt{imdb_id:07d}"


def parse_year(value) -> int | None:
    if value is None:
        return None
    match = re.search(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", str(value))
    return int(match.group(1)) if match else None


def parse_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ".").split("/")[0].strip())
    except ValueError:
        return None


def parse_int(value) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def parse_genres(value) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if text.startswith("["):
        try:
            return [str(g).strip() for g in json.loads(text.replace("'", '"')) if str(g).strip()]
        except json.JSONDecodeError:
            text = text.strip("[]")
    return [g.strip(" '\"") for g in re.split(r"[|,;/]", text) if g.strip(" '\"")]


def clean_title(stem_title: str) -> str:
    """'Winter s Bone' -> "Winter's Bone" """
    title = re.sub(r"\s+", " ", stem_title.replace("_", " ")).strip()
    return re.sub(r"(?<=\w) s\b", "'s", title)


# =============================================================================
# 3. DISCOVERY
# =============================================================================
@dataclass
class ScriptFiles:
    """Every corpus file belonging to one movie."""
    imdb_id: int
    title_from_file: str
    files: dict[str, Path] = field(default_factory=dict)

    def get(self, component: str) -> Path | None:
        return self.files.get(component)


@dataclass
class Dataset:
    root: Path
    bundles: dict[int, ScriptFiles]
    metadata: dict[int, dict]
    meta_files: dict[str, Path]
    unparsed: list[Path] = field(default_factory=list)
    duplicates: list[Path] = field(default_factory=list)

    def with_raw_text(self) -> list[int]:
        ids = [i for i, bundle in self.bundles.items() if bundle.get("raw_text")]
        return sorted(ids, key=lambda i: (self.title(i).lower(), i))

    def title(self, imdb_id: int) -> str:
        meta = self.metadata.get(imdb_id) or {}
        return meta.get("title") or self.bundles[imdb_id].title_from_file

    def record(self, imdb_id: int) -> MovieRecord:
        bundle = self.bundles[imdb_id]
        meta = self.metadata.get(imdb_id) or {}
        raw = bundle.get("raw_text")
        return MovieRecord(
            movie_id=format_imdb_id(imdb_id),
            title=self.title(imdb_id),
            year=meta.get("year"),
            genres=list(meta.get("genres") or []),
            imdb_rating=meta.get("imdb_rating"),
            metascore=meta.get("metascore"),
            imdb_votes=meta.get("imdb_votes"),
            source=str(raw) if raw else "",
        )


def _component_of(path: Path, root: Path) -> str | None:
    """Component key from the nearest ancestor folder with a known name."""
    for part in reversed(path.relative_to(root).parts[:-1]):
        component = COMPONENT_DIRS.get(part.lower())
        if component:
            return component
    return None


def _discover_files(root: Path):
    """Scan the tree -> (bundles, meta_files, unparsed, duplicates)."""
    bundles: dict[int, ScriptFiles] = {}
    meta_files: dict[str, Path] = {}
    unparsed: list[Path] = []
    duplicates: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        meta_kind = META_FILES.get(path.name.lower())
        if meta_kind:
            meta_files.setdefault(meta_kind, path)
            continue

        component = _component_of(path, root)
        if component is None:
            continue

        match = STEM_RE.match(path.stem)
        if not match:
            unparsed.append(path)
            continue

        imdb_id = int(match.group("id"))
        bundle = bundles.setdefault(imdb_id, ScriptFiles(imdb_id, clean_title(match.group("title"))))
        if component in bundle.files:
            duplicates.append(path)
        else:
            bundle.files[component] = path

    return bundles, meta_files, unparsed, duplicates


# =============================================================================
# 4. METADATA
# =============================================================================
def load_movie_metadata(path: Path, ds: DatasetConfig | None = None) -> dict[int, dict]:
    """movie_meta_data.csv -> {imdb_id: {title, year, imdb_rating, metascore, imdb_votes, genres}}"""
    ds = ds or DatasetConfig()
    metadata: dict[int, dict] = {}

    for row in read_csv(path):
        imdb_id = parse_imdb_id(_pick(row, ds.id_keys))
        if imdb_id is None or imdb_id in metadata:
            continue
        title = _pick(row, ds.title_keys)
        metadata[imdb_id] = {
            "title": str(title).strip() if title else None,
            "year": parse_year(_pick(row, ds.year_keys)),
            "imdb_rating": parse_float(_pick(row, ds.imdb_rating_keys)),
            "metascore": parse_float(_pick(row, ds.metascore_keys)),
            "imdb_votes": parse_int(_pick(row, ds.votes_keys)),
            "genres": parse_genres(_pick(row, ds.genre_keys)),
        }
    return metadata


# =============================================================================
# 5. LOADING
# =============================================================================
def load_dataset(data_dir: str | Path, ds: DatasetConfig | None = None) -> Dataset:
    ds = ds or DatasetConfig()
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"dataset directory not found: {root}")

    bundles, meta_files, unparsed, duplicates = _discover_files(root)
    metadata = load_movie_metadata(meta_files["movies"], ds) if "movies" in meta_files else {}
    return Dataset(root, bundles, metadata, meta_files, unparsed, duplicates)


def resolve_source(
    files: ScriptFiles,
    preference: str = "auto",
    fallback_to_txt: bool = True,
) -> tuple[str, Path | None]:
    """Pick the screenplay text source. Raw TXT unless the JSON parser is integrated."""
    raw = files.get("raw_text")

    if preference in ("auto", "txt"):
        return "txt", raw

    if preference == "json":
        if JSON_PARSER_AVAILABLE and files.get("rule_json"):
            return "json", files.get("rule_json")
        if fallback_to_txt:
            return "txt", raw
        raise NotImplementedError(
            "rule-based JSON parser is not integrated yet "
            "(use source_preference='txt' or fallback_to_txt=True)"
        )

    raise ValueError(f"unknown source preference: {preference!r}")


def iter_movies(
    data_dir: str | Path,
    ds: DatasetConfig | None = None,
    limit: int | None = None,
    source_preference: str = "auto",
    fallback_to_txt: bool = True,
) -> Iterator[tuple[MovieRecord, str]]:
    """Yield (MovieRecord, screenplay text) for every movie with a raw text file."""
    data = load_dataset(data_dir, ds)
    ids = data.with_raw_text()
    if not ids:
        raise FileNotFoundError(
            f"no raw screenplay found under {data.root} (expected a 'raw_texts' folder with "
            "'<Title>_<imdbid>.txt' files). Run: python -m irsearch inspect <dir>"
        )

    count = 0
    for imdb_id in ids:
        _, path = resolve_source(data.bundles[imdb_id], source_preference, fallback_to_txt)
        if path is None:
            continue
        yield data.record(imdb_id), read_text_file(path)
        count += 1
        if limit and count >= limit:
            return


# =============================================================================
# 6. ANNOTATION READERS  (not used by the pipeline yet)
# =============================================================================
def read_bert_annotations(path: Path) -> list[tuple[str, str]]:
    """'label: text' lines -> [(label, text)]. Labels are noisy: reference only."""
    pairs = []
    for line in read_text_file(path).splitlines():
        label, sep, text = line.partition(": ")
        if sep:
            pairs.append((label.strip(), text.strip()))
    return pairs


def read_rule_json(path: Path) -> list:
    return json.loads(read_text_file(path))


def iter_rule_items(data) -> Iterator[tuple[int, dict]]:
    """Flatten the JSON into (segment_index, item)."""
    for seg_idx, segment in enumerate(data):
        for item in (segment if isinstance(segment, list) else [segment]):
            if isinstance(item, dict):
                yield seg_idx, item


def rule_item_kind(item: dict) -> str:
    """'character' if subj is set, 'scene_heading' if location fields are set, else 'other'."""
    head = item.get("head_text") or {}
    if not isinstance(head, dict):
        return "other"
    if head.get("subj"):
        return "character"
    if any(head.get(k) for k in ("terior", "location", "ToD", "shot type")):
        return "scene_heading"
    return "other"


# =============================================================================
# 7. INSPECTION REPORT
# =============================================================================
def inspect_dataset(data_dir: str | Path, sample: int = 40, ds: DatasetConfig | None = None) -> str:
    data = load_dataset(data_dir, ds or DatasetConfig())

    out = _report_components(data)
    out += _report_metadata_files(data)

    ids = data.with_raw_text()
    out.append(f"\nMovies with an original screenplay: {len(ids)}")
    if not ids:
        out.append("Nothing to load: check the folder names (raw_texts, ...) and the '<Title>_<imdbid>.txt' naming.")
        return "\n".join(out)

    out += _report_matching(data, ids)
    out += _report_fill_rate(data, ids)

    step = max(1, len(ids) // sample)
    picked = ids[::step][:sample]
    texts = [(i, read_text_file(data.bundles[i].files["raw_text"])) for i in picked]

    out += _report_text_stats(texts)
    out += _report_annotations(data, picked)
    out += _report_preview(data, picked[0], texts[0][1])
    return "\n".join(out)


def _report_components(data: Dataset) -> list[str]:
    counts: Counter = Counter()
    for bundle in data.bundles.values():
        counts.update(bundle.files.keys())

    out = [f"Dataset root: {data.root.resolve()}", "", "Components (files found / distinct movies):"]
    out += [f"  {label:24s} {counts.get(key, 0):6d}" for key, label in COMPONENT_LABELS.items()]
    out.append(f"  unparsed file names     : {len(data.unparsed)}  {[p.name for p in data.unparsed[:3]]}")
    out.append(f"  duplicate files ignored : {len(data.duplicates)}")
    return out


def _report_metadata_files(data: Dataset) -> list[str]:
    listing = ", ".join(f"{k}={p.name}" for k, p in data.meta_files.items()) or "NONE FOUND"
    out = ["\nMetadata files: " + listing]
    for path in data.meta_files.values():
        rows = read_csv(path)
        columns = list(rows[0].keys()) if rows else []
        out.append(f"  [{path.name}] {len(rows)} rows; columns: {columns}")
    return out


def _report_matching(data: Dataset, ids: list[int]) -> list[str]:
    matched = [i for i in ids if i in data.metadata]
    without_meta = [data.bundles[i].files["raw_text"].name for i in ids if i not in data.metadata][:5]
    without_script = [
        i for i in data.metadata
        if i not in data.bundles or not data.bundles[i].get("raw_text")
    ]

    out = [f"  matched to movie_meta_data.csv by IMDb id: {len(matched)}/{len(ids)}"]
    if without_meta:
        out.append(f"  e.g. without metadata: {without_meta}")
    out.append(f"  metadata rows without a screenplay file: {len(without_script)}")

    for key in ("lemmas", "bert", "rule_json", "manual", "characters"):
        have = sum(1 for i in ids if data.bundles[i].get(key))
        out.append(f"  also has {COMPONENT_LABELS[key]:24s}: {have}/{len(ids)}")
    return out


def _report_fill_rate(data: Dataset, ids: list[int]) -> list[str]:
    records = [data.record(i) for i in ids]
    fields = ("year", "imdb_rating", "metascore", "imdb_votes", "genres")
    filled = {f: sum(1 for r in records if getattr(r, f) not in (None, [], "")) for f in fields}

    out = ["\nMetadata fill rate:"]
    out += [f"  {f:12s} {filled[f]:6d}/{len(records)}" for f in fields]
    if filled["genres"] == 0:
        out.append("  -> no genre information in this dataset: genre filtering needs an external source.")
    return out


def _report_text_stats(texts: list[tuple[int, str]]) -> list[str]:
    cfg = ParserConfig()
    headings, words, cues, scene_counts = [], [], [], []
    indented = no_boundary = 0

    for _, text in texts:
        lines = text.splitlines()
        scenes = parse_screenplay(text, cfg)

        headings.append(sum(1 for line in lines if is_heading(line)))
        words.append(len(text.split()))
        cues.append(sum(1 for k, line in enumerate(lines[:-1]) if cue_name(line, cfg) and lines[k + 1].strip()))
        scene_counts.append(len(scenes))
        indented += any(_indent(line.replace("\t", "    ")) >= 8 for line in lines)
        no_boundary += len(scenes) == 1 and not scenes[0].has_boundary

    return [
        f"\nRaw text format statistics on {len(texts)} sampled screenplays:",
        f"  words/script        : median {statistics.median(words):,.0f}  (min {min(words):,}, max {max(words):,})",
        f"  INT./EXT. headings  : median {statistics.median(headings):.0f}; scripts with 0 headings: {sum(1 for h in headings if h == 0)}",
        f"  scenes after parsing: median {statistics.median(scene_counts):.0f}; scripts with NO reliable scene boundary: {no_boundary}",
        f"  cue-like lines      : median {statistics.median(cues):.0f}",
        f"  keep indentation    : {indented}/{len(texts)} scripts",
    ]


def _report_annotations(data: Dataset, picked: list[int]) -> list[str]:
    out: list[str] = []

    kinds: Counter = Counter()
    head_types: Counter = Counter()
    n_json = 0
    for imdb_id in picked:
        path = data.bundles[imdb_id].get("rule_json")
        if not path:
            continue
        n_json += 1
        for _, item in iter_rule_items(read_rule_json(path)):
            kinds[rule_item_kind(item)] += 1
            head_types[item.get("head_type")] += 1
    if n_json:
        out.append(f"\nRule-based JSON ({n_json} sampled files): head_type values = {dict(head_types)}")
        out.append(f"  item kinds from head_text fields: {dict(kinds)}")

    labels: Counter = Counter()
    for imdb_id in picked:
        path = data.bundles[imdb_id].get("bert")
        if path:
            labels.update(label for label, _ in read_bert_annotations(path))
    if labels:
        out.append(f"BERT annotation labels (sample): {dict(labels)}")

    return out


def _report_preview(data: Dataset, imdb_id: int, text: str) -> list[str]:
    out = [f"\nFirst 25 lines of '{data.title(imdb_id)}' (indentation visible):"]
    out += ["  |" + line[:110] for line in text.splitlines()[:25]]
    return out