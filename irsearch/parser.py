"""Screenplay parser: original TXT screenplay -> list[Scene] of typed Elements.

Scope: structure only. Nothing here chunks, preprocesses or indexes text.
The TXT file is the source of truth: element text is the original wording, in the original
order. The only text ever dropped is (a) blank lines and (b) page artifacts (page numbers,
"(CONTINUED)", "(MORE)") when ParserConfig.drop_page_artifacts is True.

Pipeline
    1. _prepare_lines  : normalise whitespace, drop page artifacts, keep original line numbers
    2. _build_profile  : one pass over the file to learn its conventions (how often each
                         character cue occurs, whether it uses indentation, whether dialogue is
                         written inline as "NAME: text", whether headings lack INT./EXT.)
    3. _ScreenplayParser.run : second pass that classifies each line

Element kinds (models.py): heading, action, speaker, paren, dialogue, transition.

Structured scene metadata (Scene.int_ext / locations / time_of_day / shot_type) is filled ONLY
from what the heading line literally says, using small closed vocabularies; anything not
recognised stays empty. `annotation_type` is never set: it is reserved for labels coming from
the dataset's own annotation files (BERT / rule-based JSON). What the parser can say about a
TXT-derived element (source line numbers, cue extensions such as "V.O.", the raw heading, the
scene number) is stored under annotation_data["txt"] when ParserConfig.preserve_annotation_data
is True, so it can never be mistaken for dataset annotation data.

ParserConfig settings used here: max_cue_words, max_cue_chars, min_scene_headings,
drop_page_artifacts, preserve_annotation_data. (source_preference / fallback_to_txt belong to
dataset.py.)
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .config import ParserConfig
from .models import ACTION, DIALOGUE, HEADING, PAREN, SPEAKER, TRANSITION, Element, Scene

__all__ = [
    "HeadingInfo", "parse_screenplay", "parse_heading_line", "match_implicit_heading",
    "is_heading", "is_transition", "split_cue", "cue_name",
]

# vocabularies
_TOD_CORE = frozenset("""
    DAY DAYTIME NIGHT NIGHTTIME MORNING AFTERNOON EVENING DAWN DUSK SUNRISE SUNSET TWILIGHT
    MIDNIGHT NOON MIDDAY CONTINUOUS CONTINUING LATER SAME MOMENTS MOMENT MINUTES HOURS DAYS
    WEEKS MONTHS YEARS EARLIER
""".split())
_TOD_VOCAB = _TOD_CORE | frozenset("EARLY LATE NEXT THAT THE FOLLOWING A FEW OF AND TIME".split())

_SHOT_RE = re.compile(
    r"^(?:EXTREME\s+CLOSE[\s-]*UP|CLOSE[\s-]*UP|CLOSE\s+ON|CLOSE\s+SHOT|CLOSE|ANGLE\s+ON|"
    r"REVERSE\s+ANGLE|HIGH\s+ANGLE|LOW\s+ANGLE|ANGLE|WIDE\s+SHOT|WIDE\s+ON|WIDE|"
    r"ESTABLISHING(?:\s+SHOT)?|MEDIUM\s+SHOT|MEDIUM\s+ON|TWO[\s-]*SHOT|POV|P\.O\.V\.?|INSERT|"
    r"AERIAL(?:\s+SHOT|\s+VIEW)?|TRACKING(?:\s+SHOT)?|OVER\s+(?:THE\s+)?SHOULDER|"
    r"BIRD'?S[\s-]*EYE(?:\s+VIEW)?|SLOW\s+MOTION)\b",
    re.I,
)

# Uppercase lines that look like names but are not characters.
_NON_CUE_START = re.compile(
    r"^(?:ANGLE|CLOSE|WIDE|INSERT|POV|P\.O\.V|FLASHBACK|FLASH|MONTAGE|INTERCUT|SUPER|TITLE|"
    r"CHYRON|CAPTION|SUBTITLE|BACK\s+TO|END\s+OF|THE\s+END|OVER\s+BLACK|SERIES\s+OF|LATER|"
    r"MEANWHILE|CONTINUOUS|RESUME|OMITTED|CREDITS?|OPENING|CLOSING|TIME\s+CUT|SMASH|MATCH|"
    r"JUMP|DISSOLVE|FADE|CUT|WIPE|IRIS)\b"
)
# Single words that are far more often sound effects / stage words than character names;
# accepted as a cue only when they repeat (>= 3 times) in the same file.
_NON_NAME_WORDS = frozenset(
    "SILENCE DARKNESS BLACK BEAT PAUSE SUDDENLY NOTHING BANG BOOM CRASH THUD SCREAM CLICK BEEP RING".split()
)
_NON_INLINE = frozenset(
    "NOTE TITLE SUPER CAPTION INSERT CHYRON SUBTITLE TIME SIGN TEXT SCREEN READS".split()
)
_NAME_TITLES = ("MR.", "MRS.", "MS.", "DR.", "ST.", "JR.", "SR.", "MME.", "CAPT.", "LT.", "SGT.", "COL.", "GEN.")
_LOWER = re.compile(r"[a-z]")

# scene headings
_NUM = r"[A-Z]?\d{1,4}[A-Z]{0,2}"
_MARK_DOTTED = r"(?:INT\.?\s*/\s*EXT\.?|EXT\.?\s*/\s*INT\.?|I\s*/\s*E\.|INT\.|EXT\.)"
_MARK_PLAIN = (
    r"(?:(?:INT\s*/\s*EXT|EXT\s*/\s*INT|I/E|INT|EXT)(?=[\s\-\u2013\u2014:]|$)"
    r"|(?:INTERIOR|EXTERIOR)(?=\s*[\-\u2013\u2014:]))"
)
_HEAD_DOTTED = re.compile(rf"^(?:(?P<num>{_NUM})[.):]?\s+)?(?P<marker>{_MARK_DOTTED})(?P<rest>.*)$", re.I)
_HEAD_PLAIN = re.compile(rf"^(?:(?P<num>{_NUM})[.):]?\s+)?(?P<marker>{_MARK_PLAIN})(?P<rest>.*)$")
_NUM_SUFFIX = re.compile(rf"\s+(?P<num>{_NUM})\s*$")
_SEGMENT_SPLIT = re.compile(r"\s+-\s+|\s*(?:--+|[\u2013\u2014]+)\s*")
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


@dataclass
class HeadingInfo:
    text: str                 # heading as it will be shown (scene numbers removed if configured)
    raw: str                  # the original line, whitespace-trimmed
    int_ext: str = ""         # "INT" | "EXT" | "INT/EXT" | "EXT/INT" | "" (implicit headings)
    locations: list[str] = field(default_factory=list)
    time_of_day: str = ""
    shot_type: str | None = None
    scene_number: str | None = None
    explicit: bool = True     # False: no INT./EXT. marker (implicit "PLACE - NIGHT" heading)


def _is_tod(segment: str) -> bool:
    words = [w for part in segment.upper().split("/") for w in re.findall(r"[A-Z']+", part)]
    return bool(words) and all(w in _TOD_VOCAB for w in words) and any(w in _TOD_CORE for w in words)


def _shot(segment: str) -> str | None:
    m = _SHOT_RE.match(segment.strip())
    return re.sub(r"\s+", " ", m.group(0)) if m else None


def _split_segments(rest: str) -> list[str]:
    segs = [s.strip(" .-\u2013\u2014\t") for s in _SEGMENT_SPLIT.split(rest.strip())]
    segs = [s for s in segs if s]
    if len(segs) == 1 and "," in segs[0]:               # "KITCHEN, NIGHT"
        head, _, tail = segs[0].rpartition(",")
        if head.strip() and _is_tod(_TRAILING_PAREN.sub("", tail).strip()):
            segs = [head.strip(), tail.strip()]
    return segs


def _classify_segments(segs: list[str]) -> tuple[list[str], str, str | None]:
    """Split heading segments into (locations, time_of_day, shot_type); unknown -> location."""
    locations: list[str] = []
    tod = ""
    shot: str | None = None
    for idx, seg in enumerate(segs):
        core = _TRAILING_PAREN.sub("", seg).strip()
        if idx > 0 and _is_tod(core):
            tod = f"{tod} - {core}" if tod else core
        elif idx > 0 and shot is None and _shot(core):
            shot = _shot(core)
        else:
            locations.append(seg)
    return locations, tod, shot


def _heading_plausible(marker: str, rest: str, has_tod: bool) -> bool:
    """Reject prose that merely starts with 'Int.' / 'Ext.'."""
    body = rest.strip(" .:-\u2013\u2014\t")
    words = body.split()
    if not words or len(words) > 14:
        return False
    letters = [c for c in body if c.isalpha()]
    lower_ratio = (sum(c.islower() for c in letters) / len(letters)) if letters else 0.0
    if rest.rstrip().endswith(".") and lower_ratio > 0.5 and len(words) > 6:
        return False
    if marker == marker.upper() or marker[:1].isupper():
        return True
    return has_tod or lower_ratio < 0.5     # all-lowercase marker: needs more evidence


def parse_heading_line(line: str, strip_numbers: bool = True) -> HeadingInfo | None:
    """Parse an INT./EXT. scene heading; None when the line is not one."""
    s = line.strip()
    if not s or len(s) > 200:
        return None
    m = _HEAD_DOTTED.match(s) or _HEAD_PLAIN.match(s)
    if not m:
        return None
    marker, rest, num = m.group("marker"), m.group("rest"), m.group("num")

    suffix_num = None
    sm = _NUM_SUFFIX.search(rest)
    if sm:
        before = rest[: sm.start()].rstrip()
        segs_before = _split_segments(before)
        if (num and sm.group("num").upper() == num.upper()) or (
            len(segs_before) > 1 and _is_tod(_TRAILING_PAREN.sub("", segs_before[-1]))
        ):
            suffix_num, rest = sm.group("num"), before

    segs = _split_segments(rest)
    locations, tod, shot = _classify_segments(segs)
    if not _heading_plausible(marker, rest, bool(tod)):
        return None

    mk = re.sub(r"[\s.]", "", marker.upper())
    int_ext = {"I/E": "INT/EXT", "INTERIOR": "INT", "EXTERIOR": "EXT"}.get(mk, mk)

    text = s[m.start("marker"):] if strip_numbers else s
    if strip_numbers and suffix_num:
        text = re.sub(r"\s+" + re.escape(suffix_num) + r"\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return HeadingInfo(
        text=text, raw=re.sub(r"\s+", " ", s), int_ext=int_ext, locations=locations,
        time_of_day=tod, shot_type=shot, scene_number=num or suffix_num, explicit=True,
    )


def match_implicit_heading(line: str) -> HeadingInfo | None:
    """'THE KITCHEN - NIGHT' (no INT./EXT.). Deliberately strict: ALL CAPS, short, a location
    followed by a recognised time of day. Only used for scripts that lack INT./EXT. headings."""
    s = line.strip()
    if not s or len(s) > 80 or _LOWER.search(s) or s.endswith(":") or "!" in s or "?" in s:
        return None
    segs = _split_segments(s)
    if len(segs) < 2 or len(s.split()) > 12:
        return None
    first = _TRAILING_PAREN.sub("", segs[0]).strip()
    if not any(c.isalpha() for c in first) or _is_tod(first) or _shot(first):
        return None
    if not _is_tod(_TRAILING_PAREN.sub("", segs[-1]).strip()):
        return None
    locations, tod, shot = _classify_segments(segs)
    text = re.sub(r"\s+", " ", s)
    return HeadingInfo(text=text, raw=text, int_ext="", locations=locations, time_of_day=tod,
                       shot_type=shot, explicit=False)


def is_heading(line: str) -> bool:
    """True for an explicit INT./EXT. heading line (indentation and scene numbers allowed)."""
    return parse_heading_line(line) is not None


#  transitions
_TRANSITION = re.compile(
    r"""^(?:
        (?:SMASH|MATCH|JUMP|FLASH|QUICK)\s+CUT(?:\s+(?:TO|BACK\s+TO)\b.*)?[:.]?
      | CUT(?:\s+(?:TO|BACK\s+TO)\b[^!?]*|\s+BACK|\s+AWAY)?[:.]?
      | (?:LAP\s+)?DISSOLVE(?:\s+TO\b.*)?[:.]?
      | FADE(?:\s+(?:IN|OUT|UP|DOWN|TO\s+[A-Z]+))?\s*[:.]?
      | WIPE(?:\s+TO\b.*)?[:.]?
      | IRIS\s+(?:IN|OUT)[:.]?
      | THE\s+END[.:]?
      | .*\bTO:
    )$""",
    re.X,
)


def is_transition(line: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) <= 40 and s == s.upper() and bool(_TRANSITION.match(s))


# character cues
_CUE_PUNCT = set(" .'\u2019-#&/")
_BARE_CONTD = re.compile(r"\s+CONT['\u2019]?D\.?$")


def split_cue(line: str, cfg: ParserConfig | None = None) -> tuple[str, list[str]] | None:
    """If the line has the *shape* of a character cue return (name, extensions).

    "CHIEF MORGAN (V.O.) (CONT'D)" -> ("CHIEF MORGAN", ["CONT'D", "V.O."]).
    Shape only: whether it IS a cue is decided by the parser from the surrounding lines.
    """
    cfg = cfg or ParserConfig()
    s = line.strip()
    if not s or len(s) > cfg.max_cue_chars + 40:
        return None
    name, exts = s, []
    while True:
        m = _TRAILING_PAREN.search(name)
        if not m:
            break
        exts.append(m.group(0).strip()[1:-1].strip())
        name = name[: m.start()]
    name = name.strip().rstrip(":").strip()
    bm = _BARE_CONTD.search(name)
    if bm:
        exts.append("CONT'D")
        name = name[: bm.start()].rstrip()
    name = re.sub(r"\s+", " ", name)
    if not name or len(name) > cfg.max_cue_chars or len(name.split()) > cfg.max_cue_words:
        return None
    if not name[0].isalnum() or not any(c.isalpha() for c in name):
        return None
    if any(c.islower() for c in name) or not all(c.isalnum() or c in _CUE_PUNCT for c in name):
        return None
    if name.endswith(".") and not name.endswith(_NAME_TITLES):
        return None                              # a sentence, not a name
    if _NON_CUE_START.match(name) or is_transition(name):
        return None
    return name, exts


def cue_name(line: str, cfg: ParserConfig | None = None) -> str | None:
    sc = split_cue(line, cfg)
    return sc[0] if sc else None


_INLINE_UPPER = re.compile(
    r"^(?P<name>[A-Z][A-Z0-9 .'\u2019\-#&/]{0,38}?)(?P<ext>(?:\s*\([^()]*\))*)\s*:\s+(?P<text>\S.*)$"
)
_INLINE_MIXED = re.compile(r"^(?P<name>[A-Z][A-Za-z0-9 .'\u2019\-#&/]{0,38}?)\s*:\s+(?P<text>\S.*)$")


def _inline_match(text: str, cfg: ParserConfig):
    """'NAME: spoken text' -> (name, speaker_prefix, dialogue) or None."""
    m = _INLINE_UPPER.match(text) or _INLINE_MIXED.match(text)
    if not m:
        return None
    name = re.sub(r"\s+", " ", m.group("name")).strip()
    if (not name or len(name) > cfg.max_cue_chars or len(name.split()) > cfg.max_cue_words
            or name.upper() in _NON_INLINE or _NON_CUE_START.match(name.upper())):
        return None
    return name, text[: m.start("text")].rstrip(), m.group("text").strip()


def _speaker_id(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().upper()


#  line preparation
@dataclass
class _Line:
    no: int          # 1-based line number in the original file
    text: str        # stripped text ("" for blank lines)
    indent: int      # leading spaces (tabs expanded to 4)


def _indent(line: str) -> int:
    s = line.expandtabs(4)
    return len(s) - len(s.lstrip(" "))


_CONTINUED = re.compile(
    r"^\(?\s*(?:CONTINUED|CONT['\u2019]?D|CONTINUING)\s*(?:[:.]|\))?\s*\(?\d{0,3}[A-Z]?\)?\s*:?$", re.I
)
_MORE = re.compile(r"^\(\s*MORE\s*\)$", re.I)
_PAGE_NUM = re.compile(r"^[-\u2013\u2014]?\s*\d{1,3}[A-Z]?\.?\s*[-\u2013\u2014]?$")
_PAGE_WORD = re.compile(r"^PAGE\s+\d{1,3}(?:\s+OF\s+\d{1,3})?\.?$", re.I)


def _is_page_artifact(text: str, isolated: bool) -> bool:
    if _CONTINUED.match(text) or _MORE.match(text) or _PAGE_WORD.match(text):
        return True
    # a bare number is a page number only when it stands alone (not "1 / 2 / 3" inside dialogue)
    return isolated and bool(_PAGE_NUM.match(text))


_CTRL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\u200b\ufeff]")
_FLAT_BREAK = re.compile(r"(?:\s+|(?<=[.!?]))(?=(?:INT|EXT)\.(?:/(?:INT|EXT)\.)?\s+[A-Z])")
_TOD_ALT = "|".join(sorted(_TOD_CORE, key=len, reverse=True))
_FLAT_AFTER_HEADING = re.compile(
    rf"^((?:INT|EXT)\.(?:/(?:INT|EXT)\.)?\s+[^\n]*?\s[-\u2013\u2014]+\s(?:{_TOD_ALT})(?:\s+(?:{_TOD_ALT}))*)[ ]+(?=\S)",
    re.M,
)


def _normalize_text(text) -> str:
    if text is None:
        return ""
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", errors="replace")
    text = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n").replace("\u00a0", " ")
    text = _CTRL.sub("", text).expandtabs(4)
    # A whole script flattened onto very few lines: restore line breaks before INT./EXT. markers.
    if len(text) > 2000 and sum(1 for l in text.split("\n") if l.strip()) <= 3:
        text = _FLAT_AFTER_HEADING.sub(r"\1\n", _FLAT_BREAK.sub("\n", text))
    return text


def _prepare_lines(text, cfg: ParserConfig) -> list[_Line]:
    raw = _normalize_text(text).split("\n")
    lines = [_Line(i + 1, r.strip(), _indent(r)) for i, r in enumerate(raw)]
    if cfg.drop_page_artifacts:
        n = len(lines)
        keep = []
        for i, ln in enumerate(lines):
            if ln.text:
                isolated = i == 0 or not lines[i - 1].text or i + 1 >= n or not lines[i + 1].text
                if _is_page_artifact(ln.text, isolated):
                    continue
            keep.append(ln)
        lines = keep
    return lines


#  file profile (pass 1)
@dataclass
class _Profile:
    cue_counts: Counter
    inline_names: frozenset
    cue_indent: int | None        # typical cue indentation, None if the file is not indentation-aware
    dialogue_indent: int | None
    implicit_headings: bool       # use "PLACE - NIGHT" headings (script has no INT./EXT. convention)


def _build_profile(lines: list[_Line], cfg: ParserConfig) -> _Profile:
    n = len(lines)
    cues: Counter = Counter()
    cue_ind: dict[str, list[int]] = defaultdict(list)
    after_ind: dict[str, list[int]] = defaultdict(list)
    inline: Counter = Counter()
    explicit = implicit = 0
    for k, ln in enumerate(lines):
        if not ln.text:
            continue
        if is_heading(ln.text):
            explicit += 1
            continue
        if is_transition(ln.text):
            continue
        nxt = lines[k + 1] if k + 1 < n else None
        has_next = bool(nxt and nxt.text and not is_heading(nxt.text) and not is_transition(nxt.text))
        sc = split_cue(ln.text, cfg)
        if sc and has_next:
            cues[sc[0]] += 1
            cue_ind[sc[0]].append(ln.indent)
            after_ind[sc[0]].append(nxt.indent)
        elif (im := _inline_match(ln.text, cfg)) is not None:
            inline[_speaker_id(im[0])] += 1
        if match_implicit_heading(ln.text):
            implicit += 1

    repeated = [i for name, c in cues.items() if c >= 2 for i in cue_ind[name]]
    cue_indent = dlg_indent = None
    if len(repeated) >= 5 and statistics.median(repeated) >= 6:
        cue_indent = int(statistics.median(repeated))
        after = [i for name, c in cues.items() if c >= 2 for i in after_ind[name]]
        dlg_indent = int(statistics.median(after)) if after else None

    total_inline = sum(inline.values())
    inline_mode = total_inline >= 20 and total_inline > 2 * sum(cues.values())
    inline_names = frozenset(k for k, c in inline.items() if c >= (2 if inline_mode else 3))
    return _Profile(
        cue_counts=cues, inline_names=inline_names, cue_indent=cue_indent, dialogue_indent=dlg_indent,
        implicit_headings=(explicit < cfg.min_scene_headings <= implicit),
    )


#  pass 2: the parser
_PAREN_LEAD = re.compile(r"^(\([^()]*\))\s*(.*)$")


class _ScreenplayParser:
    def __init__(self, lines: list[_Line], profile: _Profile, cfg: ParserConfig) -> None:
        self.L = lines
        self.p = profile
        self.cfg = cfg
        self.scenes: list[Scene] = []
        self.n_headings = 0
        self.action: list[_Line] = []
        self.cur = self._blank_scene()

    #  bookkeeping
    def _prov(self, start: int, end: int, **extra) -> dict:
        if not self.cfg.preserve_annotation_data:
            return {}
        d = {"line_start": start, "line_end": end}
        d.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        return {"txt": d}

    def _blank_scene(self) -> Scene:
        return Scene(idx=0, heading="", elements=[], has_boundary=True,
                     annotation_data={"txt": {"preamble": True}} if self.cfg.preserve_annotation_data else {})

    def _add(self, kind: str, text: str, speaker: str | None, start: int, end: int, **extra) -> None:
        self.cur.elements.append(Element(kind=kind, text=text, speaker=speaker,
                                         annotation_data=self._prov(start, end, **extra)))
        txt = self.cur.annotation_data.get("txt")
        if txt is not None:
            txt["line_end"] = end

    def _flush_action(self) -> None:
        if self.action:
            text = " ".join(l.text for l in self.action)
            self._add(ACTION, text, None, self.action[0].no, self.action[-1].no)
            self.action = []

    def _new_scene(self, info: HeadingInfo, start: int, end: int) -> None:
        if self.cur.heading or self.cur.elements:
            self.scenes.append(self.cur)
        self.n_headings += 1
        scene = Scene(
            idx=0, heading=info.text, elements=[], has_boundary=True,
            int_ext=info.int_ext, locations=list(info.locations),
            time_of_day=info.time_of_day, shot_type=info.shot_type,
        )
        if self.cfg.preserve_annotation_data:
            scene.annotation_data = {"txt": {
                "heading_line": start, "line_start": start, "line_end": end,
                "raw_heading": info.raw if info.raw != info.text else None,
                "scene_number": info.scene_number,
                "heading_kind": "explicit" if info.explicit else "implicit",
            }}
            scene.annotation_data["txt"] = {k: v for k, v in scene.annotation_data["txt"].items() if v is not None}
        self.cur = scene

    #  headings
    def _heading_at(self, i: int) -> tuple[HeadingInfo, int] | None:
        ln = self.L[i]
        strip = self.cfg.drop_page_artifacts
        info = parse_heading_line(ln.text, strip_numbers=strip)
        if info is None and self.p.implicit_headings:
            info = match_implicit_heading(ln.text)
        if info is None:
            return None
        used = 1
        if info.explicit and ln.text.rstrip().endswith(("-", "\u2013", "\u2014", ",")) and i + 1 < len(self.L):
            nxt = self.L[i + 1]
            words = nxt.text.split()
            if (nxt.text and nxt.text == nxt.text.upper() and 0 < len(words) <= 6
                    and not is_heading(nxt.text) and not is_transition(nxt.text) and not any(c in nxt.text for c in "!?:")):
                joined = parse_heading_line(ln.text + " " + nxt.text, strip_numbers=strip)
                if joined is not None:
                    return joined, 2
        return info, used

    #dialogue
    def _accept_cue(self, k: int) -> tuple[str, list[str]] | None:
        L, p = self.L, self.p
        if k + 1 >= len(L):
            return None
        ln = L[k]
        sc = split_cue(ln.text, self.cfg)
        if sc is None:
            return None
        nxt = L[k + 1]
        if not nxt.text or is_heading(nxt.text) or is_transition(nxt.text):
            return None
        name = sc[0]
        count = p.cue_counts.get(name, 0)
        if sum(c.isalpha() for c in name) < 2 and count < 3:
            return None
        if name in _NON_NAME_WORDS and count < 3:
            return None
        if p.cue_indent is not None:               
            return sc if ln.indent >= p.cue_indent - 6 else None
        if count >= 2:
            return sc
        if len(name.split()) <= 3 and _LOWER.search(nxt.text):
            return sc
        return None

    def _try_dialogue(self, i: int) -> int | None:
        ln = self.L[i]
        if self.p.inline_names:
            im = _inline_match(ln.text, self.cfg)
            if im and _speaker_id(im[0]) in self.p.inline_names:
                return self._read_inline(i, im)
        cue = self._accept_cue(i)
        return self._read_block(i, cue) if cue else None

    def _read_inline(self, i: int, im) -> int:
        name, prefix, spoken = im
        L, n = self.L, len(self.L)
        ln = L[i]
        speaker = _speaker_id(name)
        self._flush_action()
        self._add(SPEAKER, prefix, speaker, ln.no, ln.no, layout="inline")
        parts, end, j = [spoken], ln.no, i + 1
        while j < n and L[j].text and L[j].indent > ln.indent:         
            t = L[j].text
            if is_heading(t) or is_transition(t) or self._accept_cue(j):
                break
            parts.append(t)
            end = L[j].no
            j += 1
        self._add(DIALOGUE, " ".join(parts), speaker, ln.no, end, layout="inline")
        return j

    def _blank_continues(self, j: int, d_indent: int) -> bool:
        """Indentation-aware files only: one blank line inside a speech does not end it."""
        L, n = self.L, len(self.L)
        if self.p.dialogue_indent is None or d_indent == 0 or j + 1 >= n:
            return False
        nxt = L[j + 1]
        if not nxt.text or is_heading(nxt.text) or is_transition(nxt.text) or self._accept_cue(j + 1):
            return False
        return abs(nxt.indent - d_indent) <= 3

    def _read_block(self, i: int, cue: tuple[str, list[str]]) -> int:
        L, n = self.L, len(self.L)
        name, exts = cue
        speaker = _speaker_id(name)
        ln = L[i]
        self._flush_action()
        self._add(SPEAKER, ln.text, speaker, ln.no, ln.no, extensions=exts)   
        j = first = i + 1
        d_indent = L[j].indent
        floor = max(1, d_indent - 3) if d_indent > 0 else 0
        buf: list[_Line] = []

        def flush() -> None:
            if buf:
                self._add(DIALOGUE, " ".join(b.text for b in buf), speaker, buf[0].no, buf[-1].no)
                buf.clear()

        while j < n:
            cur = L[j]
            if not cur.text:
                if self._blank_continues(j, d_indent):
                    j += 1
                    continue
                break
            if is_heading(cur.text) or is_transition(cur.text):
                break
            if j > first and self._accept_cue(j):
                break                                   
            if floor and cur.indent < floor:
                break                                   
            t = cur.text
            if t.startswith("("):
                k, joined = j, t
                while not _PAREN_LEAD.match(joined) and k + 1 < n and k - j < 2 and L[k + 1].text:
                    k += 1
                    joined += " " + L[k].text
                m = _PAREN_LEAD.match(joined)
                if m:                                   
                    flush()
                    self._add(PAREN, m.group(1), speaker, cur.no, L[k].no)
                    if m.group(2):
                        buf.append(_Line(L[k].no, m.group(2), cur.indent))
                    j = k + 1
                    continue
            buf.append(cur)
            j += 1
        flush()
        return j

    # main loop
    def run(self) -> list[Scene]:
        L, n = self.L, len(self.L)
        i = 0
        while i < n:
            ln = L[i]
            if not ln.text:
                self._flush_action()
                i += 1
                continue
            h = self._heading_at(i)
            if h:
                info, used = h
                self._flush_action()
                self._new_scene(info, ln.no, L[i + used - 1].no)
                i += used
                continue
            if is_transition(ln.text):
                self._flush_action()
                self._add(TRANSITION, ln.text, None, ln.no, ln.no)
                i += 1
                continue
            nxt = self._try_dialogue(i)
            if nxt is not None:
                i = nxt
                continue
            self.action.append(ln)
            i += 1
        self._flush_action()
        if self.cur.heading or self.cur.elements:
            self.scenes.append(self.cur)
        return self._finalize()

    def _finalize(self) -> list[Scene]:
        scenes = [s for s in self.scenes if s.heading or s.elements]
        if not scenes:
            return []
        if self.n_headings == 0 or self.n_headings < self.cfg.min_scene_headings:
           
            merged: list[Element] = []
            for s in scenes:
                if s.heading:
                    merged.append(Element(kind=HEADING, text=s.heading, annotation_data=dict(s.annotation_data)))
                merged.extend(s.elements)
            data = {}
            if self.cfg.preserve_annotation_data:
                data = {"txt": {"pseudo_scene": True, "headings_found": self.n_headings}}
            return [Scene(idx=0, heading="", elements=merged, has_boundary=False, annotation_data=data)]
        for k, s in enumerate(scenes):
            s.idx = k
        return scenes


# public entry point
def parse_screenplay(text, cfg: ParserConfig | None = None) -> list[Scene]:
    """Parse an original screenplay (str, or bytes decoded as UTF-8) into scenes.

    Always returns a list: [] for empty / whitespace-only / non-text input, a single pseudo-scene
    (has_boundary=False) when fewer than cfg.min_scene_headings scene headings are found.
    """
    cfg = cfg or ParserConfig()
    lines = _prepare_lines(text, cfg)
    if not any(l.text for l in lines):
        return []
    return _ScreenplayParser(lines, _build_profile(lines, cfg), cfg).run()