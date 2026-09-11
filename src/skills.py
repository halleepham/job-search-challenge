"""
REQ-2: skill vocabulary and extraction (FROZEN v1.0).

Matching is **word-boundary regex over normalized text, never substring**
(AC-2.2). This is the exact defect in the Stage 2 AI code, where bidirectional
substring matching on strings of length >= 3 made "Java" match "JavaScript" and
"Go" match "MongoDB" - silently inflating the largest score component.

Extraction reads the description first (AC-2.3 v1.1): `skills_desc` is 98.0%
null, so it can only supplement, never gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

VOCAB_PATH = Path("data/vocabulary/skills_vocabulary.json")

#: AC-2.4 section headings. A section runs until the next recognized heading.
_REQUIRED_HEADING = re.compile(
    r"^[^\S\n]*(?:required|requirements|qualifications|must[- ]have|"
    r"minimum(?:\s+qualifications)?|basic\s+qualifications|what\s+you'?ll\s+need|"
    r"who\s+you\s+are)\b[^\n]*$",
    re.I | re.M,
)
_PREFERRED_HEADING = re.compile(
    r"^[^\S\n]*(?:preferred(?:\s+qualifications)?|nice[- ]to[- ]have|bonus|"
    r"a\s+plus|desired|additional\s+qualifications|pluses)\b[^\n]*$",
    re.I | re.M,
)
#: Any other heading also terminates a section, so a section cannot run into
#: Benefits, Compensation, or EEO boilerplate.
_OTHER_HEADING = re.compile(
    r"^[^\S\n]*(?:benefits|compensation|salary|about\s+us|about\s+the|"
    r"responsibilities|what\s+you'?ll\s+do|duties|eeo|equal\s+opportunity|"
    r"perks|why\s+join)\b[^\n]*$",
    re.I | re.M,
)

#: Boundary guards. `&` is excluded on both sides so "R&D" does not yield "R";
#: `+`, `#` and `.` are excluded so "C++", "C#" and ".net" match as whole terms
#: rather than as prefixes of longer tokens.
_LEFT = r"(?<![\w+#.&/*])"
_RIGHT = r"(?![\w+#&/*])"

#: AC-2.2 v1.1: a single character in prose is not evidence. These count only
#: when another recognised skill sits within CO_OCCURRENCE_WINDOW characters, so
#: "experience with R and SAS" counts while "P/R organization" does not.
CO_OCCURRENCE_WINDOW = 40


@dataclass(frozen=True)
class Vocabulary:
    canonical: tuple[str, ...]
    alias_to_canonical: dict[str, str]
    pattern: re.Pattern
    surface_to_canonical: dict[str, str]


@lru_cache(maxsize=1)
def load_vocabulary(path: Path | str = VOCAB_PATH) -> Vocabulary:
    """Load the committed gazetteer and compile one combined matcher."""
    raw = json.loads(Path(path).read_text())["skills"]

    surface: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}
    for canon, aliases in raw.items():
        surface[canon] = canon
        for alias in aliases:
            surface[alias] = canon
            alias_to_canonical[alias] = canon

    # Longest surface form first: `finditer` does not overlap, so "power bi"
    # consumes the span before "bi" can match inside it.
    ordered = sorted(surface, key=len, reverse=True)
    pattern = re.compile(
        _LEFT + "(?:" + "|".join(re.escape(t) for t in ordered) + ")" + _RIGHT, re.I
    )
    return Vocabulary(tuple(raw), alias_to_canonical, pattern, surface)


def match_skills(text: str | None, vocab: Vocabulary | None = None) -> set[str]:
    """
    AC-2.2: canonical skills present in *text*, by word-boundary match.

    Never substring: "JavaScript" does not yield "java", "MongoDB" does not
    yield "go", and "R&D" does not yield "r".
    """
    if not text:
        return set()
    vocab = vocab or load_vocabulary()
    hits = [(m.start(), m.end(), vocab.surface_to_canonical[m.group(0).lower()])
            for m in vocab.pattern.finditer(text)]

    multi = [(a, b) for a, b, name in hits if len(name) > 1]
    found = {name for _, _, name in hits if len(name) > 1}

    # A single-letter skill needs a neighbour: measured on the corpus, 645
    # postings matched `r` and 83 had it as their only skill - from "P/R
    # organization", "mission r equirements" and "Project Man******r".
    for start, end, name in hits:
        if len(name) == 1 and any(
            start - CO_OCCURRENCE_WINDOW <= other_end
            and other_start <= end + CO_OCCURRENCE_WINDOW
            for other_start, other_end in multi
        ):
            found.add(name)
    return found


def split_sections(description: str | None) -> dict[str, str]:
    """
    AC-2.4: split a description into ``required`` / ``preferred`` / ``other``.

    A section opens at a recognized heading and runs to the next heading of any
    recognized kind, so a preferred block cannot absorb the Benefits section.
    """
    out = {"required": "", "preferred": "", "other": ""}
    if not description:
        return out

    marks: list[tuple[int, str]] = []
    for rx, kind in ((_REQUIRED_HEADING, "required"),
                     (_PREFERRED_HEADING, "preferred"),
                     (_OTHER_HEADING, "other")):
        marks += [(m.start(), kind) for m in rx.finditer(description)]
    if not marks:
        return out

    marks.sort()
    for i, (start, kind) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(description)
        out[kind] += description[start:end] + "\n"
    return out


def extract_skills(
    description: str | None,
    skills_desc: str | None = None,
    vocab: Vocabulary | None = None,
) -> tuple[set[str], set[str]]:
    """
    AC-2.3 / AC-2.5: ``(required_skills, preferred_skills)``.

    Description section-parsing is the primary path. When no section headings
    parse, every skill found anywhere in the description is treated as required
    and preferred stays **empty** - never a copy of required, never a guess
    (AC-2.5). `skills_desc`, present on only 2% of postings, adds to required
    and can never remove or gate a skill.
    """
    vocab = vocab or load_vocabulary()
    sections = split_sections(description)

    if sections["required"] or sections["preferred"]:
        required = match_skills(sections["required"], vocab)
        preferred = match_skills(sections["preferred"], vocab)
    else:
        required = match_skills(description, vocab)
        preferred = set()

    required |= match_skills(skills_desc, vocab)
    return required, preferred - required
