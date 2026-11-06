import pytest

from irsearch.config import ParserConfig
from irsearch.models import ACTION, DIALOGUE, HEADING, PAREN, SPEAKER, TRANSITION
from irsearch.parser import (
    cue_name, is_heading, is_transition, match_implicit_heading, parse_heading_line,
    parse_screenplay, split_cue,
)


def kinds(scene):
    return [e.kind for e in scene.elements]


def words(scenes):
    """All words of the parsed result, in order (heading first in each scene)."""
    return " ".join(l.text for s in scenes for l in s.lines()).split()


# ------------------------------------------------------------------ scene headings
@pytest.mark.parametrize("line", [
    "INT. POLICE STATION - NIGHT", "EXT. DESERT - DAY", "INT./EXT. CAR - MOVING - DAY",
    "EXT./INT. HOUSE - DAY", "I/E. TRUCK - NIGHT", "INT KITCHEN - DAY", "INT.KITCHEN - DAY",
    "  12   INT. KITCHEN - DAY   12", "A12 EXT. ROOF - DAWN", "Int. Kitchen - Day",
    "INTERIOR - KITCHEN - DAY", "int. kitchen - day", "INT. BIG ROOM -- NIGHT", "INT. HOUSE \u2013 DAY",
])
def test_heading_variants_detected(line):
    assert is_heading(line)


@pytest.mark.parametrize("line", [
    "INTERROGATION begins slowly.", "INTERIOR DESIGNER", "EXTRA 1", "EXTERIOR shots only.",
    "Int. means interior, and it was used all through the old scripts of the time.",
    "He went into the kitchen.", "INT", "",
])
def test_non_headings_rejected(line):
    assert not is_heading(line)


def test_structured_scene_metadata():
    h = parse_heading_line("12 INT. HOUSE PARTY - BASEMENT - NIGHT 12")
    assert h.text == "INT. HOUSE PARTY - BASEMENT - NIGHT"
    assert (h.int_ext, h.locations, h.time_of_day, h.shot_type, h.scene_number) == (
        "INT", ["HOUSE PARTY", "BASEMENT"], "NIGHT", None, "12")
    h = parse_heading_line("INT./EXT. CAR - MOVING - CLOSE ON DRIVER - DAY (FLASHBACK)")
    assert h.int_ext == "INT/EXT" and h.locations == ["CAR", "MOVING"]
    assert h.shot_type == "CLOSE ON" and h.time_of_day == "DAY"
    h = parse_heading_line("EXT. DESERT")
    assert (h.locations, h.time_of_day, h.shot_type) == (["DESERT"], "", None)   # nothing invented
    h = parse_heading_line("INT. APARTMENT 4B")
    assert h.locations == ["APARTMENT 4B"] and h.scene_number is None             # location digits kept
    assert parse_heading_line("INT. KITCHEN, NIGHT").time_of_day == "NIGHT"
    assert parse_heading_line("INT. WAREHOUSE - SOMETHING ODD").time_of_day == ""


def test_scene_numbers_kept_when_artifacts_are_not_dropped():
    sc = parse_screenplay("12 INT. A - DAY\nx\n13 INT. B - DAY\ny\n14 INT. C - DAY\nz\n",
                          ParserConfig(drop_page_artifacts=False))
    assert sc[0].heading == "12 INT. A - DAY"


def test_wrapped_heading_is_joined():
    text = "INT. A VERY LONG LOCATION NAME THAT WRAPS -\nNIGHT\nSome action here.\nINT. B - DAY\nx\nINT. C - DAY\ny\n"
    sc = parse_screenplay(text)
    assert sc[0].heading == "INT. A VERY LONG LOCATION NAME THAT WRAPS - NIGHT"
    assert sc[0].time_of_day == "NIGHT" and len(sc) == 3


def test_scene_fields_flow_to_scene_and_location_property():
    text = "INT. KITCHEN - NIGHT\nTom cooks.\nEXT. GARDEN - DAY\nRoses.\nINT. HALL - DAY\nx\n"
    sc = parse_screenplay(text)
    assert (sc[0].int_ext, sc[0].locations, sc[0].time_of_day) == ("INT", ["KITCHEN"], "NIGHT")
    assert sc[1].location == ("EXT", "GARDEN", "DAY")


def test_adjacent_scenes_are_never_merged_even_without_content():
    sc = parse_screenplay("INT. A - DAY\nINT. B - DAY\nINT. C - DAY\nText.\n")
    assert [s.heading for s in sc] == ["INT. A - DAY", "INT. B - DAY", "INT. C - DAY"]


def test_implicit_headings_only_when_script_has_no_int_ext():
    body = "".join(f"THE KITCHEN - NIGHT\nTom cooks pasta number {i}.\nTHE GARDEN - DAY\nRoses.\n" for i in range(3))
    sc = parse_screenplay(body)
    assert len(sc) == 6 and all(s.has_boundary for s in sc)
    assert sc[0].int_ext == "" and sc[0].locations == ["THE KITCHEN"] and sc[0].time_of_day == "NIGHT"
    # with a real INT./EXT. structure the same line stays plain action text (no false split)
    real = "INT. A - DAY\nTHE KITCHEN - NIGHT\nx\nINT. B - DAY\ny\nINT. C - DAY\nz\n"
    assert len(parse_screenplay(real)) == 3
    assert match_implicit_heading("LATER - NIGHT") is None
    assert match_implicit_heading("HE RUNS - NIGHT FALLS") is None


# ------------------------------------------------------------------ cues, dialogue, parentheticals
def test_cue_shape_and_config_limits():
    cfg = ParserConfig()
    assert split_cue("CHIEF MORGAN (V.O.) (CONT'D)", cfg) == ("CHIEF MORGAN", ["CONT'D", "V.O."])
    assert split_cue("JOHN CONT'D", cfg) == ("JOHN", ["CONT'D"])
    assert cue_name("DR. SMITH", cfg) == "DR. SMITH" and cue_name("COP #2", cfg) == "COP #2"
    assert cue_name("MARY/JOHN", cfg) == "MARY/JOHN"
    assert cue_name("CUT TO:", cfg) is None and cue_name("ANGLE ON THE DOOR", cfg) is None
    assert cue_name("A VERY LONG UPPERCASE ACTION LINE HERE", cfg) is None       # > max_cue_words
    assert cue_name("SIX WORDS IN THIS NAME HERE", ParserConfig(max_cue_words=6)) is not None
    assert cue_name("SIX WORDS IN THIS NAME HERE", ParserConfig(max_cue_words=5)) is None
    assert cue_name("ALEXANDER", ParserConfig(max_cue_chars=5)) is None


def test_original_cue_text_and_order_preserved():
    sc = parse_screenplay("INT. A - DAY\n\n          CHIEF MORGAN (V.O.)\n     (quietly)\n     Hello there.\n"
                          "\nINT. B - DAY\nx\nINT. C - DAY\ny\n")
    e = sc[0].elements
    assert [x.kind for x in e] == [SPEAKER, PAREN, DIALOGUE]
    assert e[0].text == "CHIEF MORGAN (V.O.)" and e[0].speaker == "CHIEF MORGAN"
    assert e[1].text == "(quietly)" and e[2].text == "Hello there."


def test_multiline_parenthetical_and_inline_paren_with_text():
    text = ("INT. A - DAY\n\n          SARAH\n     (trying to\n     stay calm)\n     I know.\n"
            "          TOM\n     (beat) Fine.\nINT. B - DAY\nx\nINT. C - DAY\ny\n")
    e = parse_screenplay(text)[0].elements
    paren = [x.text for x in e if x.kind == PAREN]
    assert paren == ["(trying to stay calm)", "(beat)"]
    assert [x.text for x in e if x.kind == DIALOGUE] == ["I know.", "Fine."]


def test_cue_directly_after_dialogue_without_blank_line_starts_a_new_speaker():
    text = "INT. A - DAY\n          ANNA\n     Hi.\n          BOB\n     Hello.\n          ANNA\n     Bye.\nINT. B - DAY\nx\nINT. C - DAY\ny\n"
    e = parse_screenplay(text)[0].elements
    assert [(x.kind, x.speaker) for x in e if x.kind == SPEAKER] == [(SPEAKER, "ANNA"), (SPEAKER, "BOB"), (SPEAKER, "ANNA")]


def test_blank_line_inside_a_speech_in_indented_file():
    block = "          ANNA\n     First paragraph of the speech.\n\n     Second paragraph of the speech.\n\nAction line.\n\n"
    text = "INT. A - DAY\n\n" + block * 6 + "INT. B - DAY\nx\nINT. C - DAY\ny\n"
    e = parse_screenplay(text)[0].elements
    d = [x.text for x in e if x.kind == DIALOGUE][0]
    assert d == "First paragraph of the speech. Second paragraph of the speech."   # one speech, one element
    assert [x.kind for x in e][:3] == [SPEAKER, DIALOGUE, ACTION]


def test_caps_action_is_not_a_cue_in_an_indented_file():
    lines = []
    for i in range(6):
        lines += ["          ANNA", "     Line %d of her speech." % i, "", "Action number %d." % i, ""]
    lines += ["THE DOOR OPENS", "He steps inside.", ""]
    text = "INT. A - DAY\n\n" + "\n".join(lines) + "\nINT. B - DAY\nx\nINT. C - DAY\ny\n"
    sc = parse_screenplay(text)
    assert "THE DOOR OPENS" not in [e.speaker for e in sc[0].elements]
    assert any(e.kind == ACTION and e.text.startswith("THE DOOR OPENS") for e in sc[0].elements)


def test_inline_dialogue_mixed_case_and_continuation():
    text = "INT. A - DAY\n" + "".join(f"Bob: line number {i} here\nEve: reply {i}\n" for i in range(25))
    text += "Note: this is not a speaker\nINT. B - DAY\nx\nINT. C - DAY\ny\n"
    e = parse_screenplay(text)[0].elements
    assert {x.speaker for x in e if x.kind == SPEAKER} == {"BOB", "EVE"}
    assert not any(x.speaker == "NOTE" for x in e)
    assert [x.text for x in e if x.kind == SPEAKER][:2] == ["Bob:", "Eve:"]


# ------------------------------------------------------------------ transitions / artifacts
@pytest.mark.parametrize("line", ["CUT TO:", "SMASH CUT TO:", "FADE IN:", "FADE OUT.", "FADE TO BLACK.",
                                  "DISSOLVE TO:", "MATCH CUT TO:", "THE END", "CUT TO BLACK.", "BACK TO SCENE TO:"])
def test_transitions(line):
    assert is_transition(line)


@pytest.mark.parametrize("line", ["CUT THE ENGINE", "FADE AWAY", "He cuts to the chase.", "THE ENDLESS ROAD AHEAD OF US"])
def test_not_transitions(line):
    assert not is_transition(line)


def test_page_artifacts_dropped_but_counting_dialogue_kept():
    text = ("INT. A - DAY\n\n          ANNA\n     One,\n     2\n     3\n     four.\n\n12.\n\n(CONTINUED)\n\n"
            "          ANNA (CONT'D)\n     More.\n          (MORE)\nINT. B - DAY\nx\nINT. C - DAY\ny\n")
    sc = parse_screenplay(text)
    ws = words(sc)
    assert "12." not in ws and "(CONTINUED)" not in ws and "(MORE)" not in ws
    assert [x.text for x in sc[0].elements if x.kind == DIALOGUE][0] == "One, 2 3 four."
    kept = parse_screenplay(text, ParserConfig(drop_page_artifacts=False))
    assert "(CONTINUED)" in words(kept) and "12." in words(kept)


def test_text_order_and_content_preserved():
    text = ("FADE IN:\n\nINT. A - DAY\n\nTom walks in.\n\n          TOM\n     Hello.\n\nCUT TO:\n\n"
            "EXT. B - DAY\n\nRain.\n\nINT. C - DAY\n\nEnd.\n")
    sc = parse_screenplay(text)
    assert words(sc) == text.split()      # nothing lost, nothing reordered, nothing invented
    assert sc[0].heading == "" and kinds(sc[0]) == [TRANSITION]


# ------------------------------------------------------------------ safety / fallbacks
@pytest.mark.parametrize("bad", ["", "   \n\n\t\n", None, "\ufeff\x00\x01", b"", 12345 - 12345 or ""])
def test_empty_and_malformed_inputs(bad):
    assert parse_screenplay(bad) == []


def test_bytes_and_binary_noise_do_not_crash():
    sc = parse_screenplay(b"INT. A - DAY\nHello \xff\xfe world.\n")
    assert len(words(sc)) >= 4
    parse_screenplay("\x00\x01INT. A - DAY\n" + "x" * 50000)


def test_no_headings_gives_one_pseudo_scene_without_claims():
    sc = parse_screenplay("He walked.\n\n          ANNA\n     Hi.\n" * 5)
    assert len(sc) == 1 and sc[0].has_boundary is False
    assert (sc[0].int_ext, sc[0].locations, sc[0].time_of_day, sc[0].shot_type) == ("", [], "", None)


def test_too_few_headings_merge_into_pseudo_scene_but_keep_them_as_heading_elements():
    text = "INT. A - DAY\nOne.\nEXT. B - DAY\nTwo.\n"
    sc = parse_screenplay(text)                              # 2 headings < min_scene_headings (3)
    assert len(sc) == 1 and not sc[0].has_boundary
    assert [e.kind for e in sc[0].elements] == [HEADING, ACTION, HEADING, ACTION]
    assert words(sc) == text.split()
    assert len(parse_screenplay(text, ParserConfig(min_scene_headings=2))) == 2
    assert len(parse_screenplay(text, ParserConfig(min_scene_headings=0))) == 2


def test_script_flattened_onto_one_line_is_recovered():
    flat = " ".join(f"INT. ROOM {i} - DAY Tom paces around the room and thinks about things number {i}. " * 3 for i in range(40))
    assert len(flat) > 2000 and "\n" not in flat
    assert len(parse_screenplay(flat)) >= 40


# ------------------------------------------------------------------ annotation data policy
def test_annotation_data_is_txt_provenance_only(classic):
    sc = parse_screenplay(classic)
    police = sc[1]
    assert police.annotation_data["txt"]["heading_line"] > 0
    cue = [e for e in police.elements if e.kind == SPEAKER and e.speaker == "CHIEF MORGAN"][0]
    assert cue.annotation_data["txt"]["extensions"] == ["V.O."]
    assert all(e.annotation_type is None for s in sc for e in s.elements)      # never invented
    assert all(set(e.annotation_data) <= {"txt"} for s in sc for e in s.elements)


def test_preserve_annotation_data_false_leaves_everything_empty(classic):
    sc = parse_screenplay(classic, ParserConfig(preserve_annotation_data=False))
    assert all(s.annotation_data == {} for s in sc)
    assert all(e.annotation_data == {} for s in sc for e in s.elements)
    assert [s.heading for s in sc] == [s.heading for s in parse_screenplay(classic)]   # same parse