
from pathlib import Path
from collections import Counter
import random

from irsearch.config import Config
from irsearch.dataset import DatasetConfig, iter_movies
from irsearch.parser import parse_screenplay


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_ROOT = Path(r"C:\Users\User\Downloads\movies_script")

NUMBER_OF_MOVIES = 10
RANDOM_SAMPLE = False

# Display only the first N scenes of each screenplay
SCENES_TO_DISPLAY = 3

# Maximum number of characters displayed per element
MAX_TEXT_LENGTH = 250


# =========================================================
# INITIALIZATION
# =========================================================

dataset_config = DatasetConfig()

parser_config = Config().parser


# =========================================================
# PARSER TEST FUNCTION
# =========================================================

def test_movie(movie, screenplay_text):

    scenes = parse_screenplay(screenplay_text, parser_config)

    elements_count = Counter()
    total_words = 0

    for scene in scenes:
        total_words += scene.n_words

        for element in scene.elements:
            elements_count[element.kind] += 1

    headings = sum(
        1 for scene in scenes
        if scene.heading
    )

    speakers = set(
        element.speaker
        for scene in scenes
        for element in scene.elements
        if element.speaker
    )

    dialogue_words = sum(
        len(element.text.split())
        for scene in scenes
        for element in scene.elements
        if element.kind == "dialogue"
    )

    action_words = sum(
        len(element.text.split())
        for scene in scenes
        for element in scene.elements
        if element.kind == "action"
    )

    print("\n" + "=" * 75)
    print(f"MOVIE: {movie.title}")
    print(f"IMDb ID: {movie.movie_id}")
    print("=" * 75)

    print(f"Original words:       {len(screenplay_text.split())}")
    print(f"Parsed words:         {total_words}")
    print(f"Detected scenes:      {len(scenes)}")
    print(f"Scenes with headings: {headings}")
    print(f"Detected speakers:    {len(speakers)}")
    print(f"Dialogue words:       {dialogue_words}")
    print(f"Action words:         {action_words}")

    print("\nElement distribution:")

    for kind, count in elements_count.most_common():
        print(f"  {kind:<15}: {count}")

    print("\nDetected speakers:")

    for speaker in sorted(speakers):
        print(f"  - {speaker}")

    # Display sample scenes
    print("\nSAMPLE SCENES")

    for scene in scenes[:SCENES_TO_DISPLAY]:

        print("\n" + "-" * 60)
        print(f"Scene {scene.idx}")
        print(f"Heading: {scene.heading}")
        print(f"Location: {scene.location}")
        print(f"Time: {scene.time_of_day}")
        print("-" * 60)

        for element in scene.elements[:10]:

            text = element.text.replace("\n", " ")

            if len(text) > MAX_TEXT_LENGTH:
                text = text[:MAX_TEXT_LENGTH] + "..."

            print(
                f"[{element.kind}] "
                f"{element.speaker or ''} "
                f"{text}"
            )

    # Basic warnings
    warnings = []

    if not screenplay_text.strip():
        warnings.append("Empty screenplay")

    if not scenes:
        warnings.append("No scenes returned")

    if len(scenes) == 1 and not scenes[0].has_boundary:
        warnings.append("Only one pseudo-scene detected")

    if headings == 0:
        warnings.append("No scene headings detected")

    if not speakers:
        warnings.append("No speakers detected")

    if total_words < len(screenplay_text.split()) * 0.5:
        warnings.append("Less than 50% of original words represented")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  [!] {warning}")
    else:
        print("\nNo basic validation warnings.")


# =========================================================
# LOAD AND TEST REAL SCREENPLAYS
# =========================================================

def main():

    print("Loading screenplay dataset...")

    movies = list(
        iter_movies(
            data_dir=DATASET_ROOT,
            ds=dataset_config,
            limit=NUMBER_OF_MOVIES
        )
    )

    if not movies:
        print("No movies found. Check DATASET_ROOT.")
        return

    print(f"Loaded {len(movies)} movies.")

    if RANDOM_SAMPLE:
        random.seed(42)
        movies = random.sample(
            movies,
            min(NUMBER_OF_MOVIES, len(movies))
        )

    for movie, screenplay_text in movies:
        test_movie(movie, screenplay_text)


if __name__ == "__main__":
    main()

