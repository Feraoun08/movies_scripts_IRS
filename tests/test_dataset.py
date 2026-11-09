
import csv
import json
from .dataset import (
    load_dataset,
    iter_movies,
    resolve_source,
    rule_item_kind,
    parse_imdb_id,
    format_imdb_id,
)


def create_fake_dataset(tmp_path):
    """Create a miniature version of our actual dataset."""

    root = tmp_path / "movies_script"

    # Original screenplay
    raw_dir = (
        root / "screenplay_data" / "data"
        / "raw_texts" / "raw_texts"
    )

    raw_dir.mkdir(parents=True)

    screenplay = raw_dir / "Test Movie_0038190.txt"

    screenplay.write_text(
        "INT. POLICE STATION - NIGHT\n"
        "JOHN\n"
        "Where is the suspect?\n"
        "SARAH\n"
        "He escaped through the back door.\n",
        encoding="utf-8",
    )

    # Lemmatized version
    lemma_dir = (
        root / "screenplay_data" / "data"
        / "raw_text_lemmas" / "raw_text_lemmas"
    )

    lemma_dir.mkdir(parents=True)

    (lemma_dir / "Test Movie_0038190_lemmas.txt").write_text(
        "police station night john suspect",
        encoding="utf-8",
    )

    # BERT annotations
    bert_dir = (
        root / "screenplay_data" / "data"
        / "BERT_annotations" / "BERT_annotations"
    )

    bert_dir.mkdir(parents=True)

    (bert_dir / "Test Movie_0038190_anno.txt").write_text(
        "scene_heading: INT. POLICE STATION - NIGHT\n"
        "speaker_heading: JOHN\n"
        "dialog: Where is the suspect?\n",
        encoding="utf-8",
    )

    # Rule-based JSON annotations
    json_dir = (
        root / "screenplay_data" / "data"
        / "rule_based_annotations" / "rule_based_annotations"
    )

    json_dir.mkdir(parents=True)

    annotation_data = [
        [
            {
                "head_type": "heading",
                "head_text": {
                    "terior": "INT.",
                    "location": ["POLICE STATION"],
                    "ToD": "NIGHT",
                    "subj": None,
                },
                "text": "A police station at night.",
            },
            {
                "head_type": "heading",
                "head_text": {
                    "subj": "JOHN",
                    "location": None,
                    "terior": None,
                    "ToD": None,
                },
                "text": "Where is the suspect?",
            },
        ]
    ]

    (json_dir / "Test Movie_0038190.json").write_text(
        json.dumps(annotation_data),
        encoding="utf-8",
    )

    # Metadata CSV
    metadata_dir = root / "movie_metadata"
    metadata_dir.mkdir(parents=True)

    metadata_file = metadata_dir / "movie_meta_data.csv"

    with metadata_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "imdbid",
                "title",
                "year",
                "metascore",
                "imdb user rating",
                "number of imdb user votes",
            ],
        )

        writer.writeheader()

        writer.writerow({
            "imdbid": 38190,
            "title": "Test Movie",
            "year": 1998,
            "metascore": 85,
            "imdb user rating": 7.5,
            "number of imdb user votes": 12000,
        })

    return root


def test_imdb_id_helpers():
    assert parse_imdb_id("tt0038190") == 38190
    assert parse_imdb_id("0038190") == 38190
    assert format_imdb_id(38190) == "tt0038190"


def test_dataset_grouping_and_metadata(tmp_path):
    root = create_fake_dataset(tmp_path)

    dataset = load_dataset(root)

    # Exactly one movie should be discovered.
    assert len(dataset.bundles) == 1

    movie_id = 38190

    bundle = dataset.bundles[movie_id]

    # All components should belong to the same movie.
    assert bundle.get("raw_text") is not None
    assert bundle.get("lemmas") is not None
    assert bundle.get("bert") is not None
    assert bundle.get("rule_json") is not None

    # Metadata should match using IMDb ID.
    record = dataset.record(movie_id)

    assert record.movie_id == "tt0038190"
    assert record.title == "Test Movie"
    assert record.year == 1998
    assert record.imdb_rating == 7.5
    assert record.metascore == 85.0
    assert record.imdb_votes == 12000

    # No genre column was provided.
    assert record.genres == []


def test_iter_movies_returns_original_text(tmp_path):
    root = create_fake_dataset(tmp_path)

    movies = list(iter_movies(root))

    assert len(movies) == 1

    record, text = movies[0]

    assert record.title == "Test Movie"
    assert "Where is the suspect?" in text
    assert "He escaped through the back door." in text

    # It must return the original TXT, not the annotation.
    assert "speaker_heading:" not in text


def test_source_selection(tmp_path):
    root = create_fake_dataset(tmp_path)

    dataset = load_dataset(root)

    bundle = dataset.bundles[38190]

    # Auto mode uses original TXT.
    kind, path = resolve_source(bundle, "auto")

    assert kind == "txt"
    assert path == bundle.get("raw_text")

    # JSON mode falls back to TXT because its parser is not integrated.
    kind, path = resolve_source(
        bundle,
        "json",
        fallback_to_txt=True,
    )

    assert kind == "txt"
    assert path == bundle.get("raw_text")


def test_rule_item_classification():
    scene_heading = {
        "head_type": "heading",
        "head_text": {
            "terior": "INT.",
            "location": ["KITCHEN"],
            "ToD": "NIGHT",
        },
        "text": "A dark kitchen.",
    }

    character = {
        "head_type": "heading",
        "head_text": {
            "subj": "JOHN",
        },
        "text": "Hello!",
    }

    other = {
        "head_type": "heading",
        "head_text": {},
        "text": "Some text.",
    }

    assert rule_item_kind(scene_heading) == "scene_heading"
    assert rule_item_kind(character) == "character"
    assert rule_item_kind(other) == "other"
