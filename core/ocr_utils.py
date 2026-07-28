"""
core/ocr_utils.py — Canonical OCR label parsing helpers.

Single source of truth for all numeric-label detection and area/length
parsing.  Replaces five scattered implementations that disagreed on which
unit suffixes, decimal separators, and integer forms to accept.

Public API
----------
is_numeric_ocr_label(text)  -> bool
parse_area_m2(text)         -> Optional[float]
parse_room_area_m2(text)    -> Optional[float]
is_apartment_total_m2(area) -> bool
looks_like_room_name(text)  -> bool
parse_length_m(text)        -> Optional[float]
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Area unit suffixes accepted (case-insensitive)
_UNIT_SUFFIX = (
    r"(?:\s*(?:м²|м2|кв\.?\s*м\.?|m²|m2|sq\.?\s*m\.?))"
)

# Numeric label: integer or decimal (dot or comma), optional area unit
_NUMERIC_LABEL_RE = re.compile(
    r"^\s*[+\-]?\d+([.,]\d+)?" + _UNIT_SUFFIX + r"?\s*$",
    re.IGNORECASE,
)

# Area value: decimal number (mandatory) + optional unit
_AREA_PATTERN = re.compile(
    r"^\s*(\d{1,4}[.,]\d{1,2})\s*"
    r"(?:м²|м2|кв\.?\s*м\.?|m²|m2|sq\.?\s*m\.?)?\s*$",
    re.IGNORECASE,
)

# Wall-length value: decimal number only, no unit
_LENGTH_PATTERN = re.compile(
    r"^\s*(\d{1,3}[.,]\d{1,2})\s*$",
)

# Default ceiling for a single room's printed area label, in m².  Mirrors
# PipelineConfig.max_room_label_m2 — callers holding a config should pass its
# value through rather than relying on this default.
MAX_ROOM_LABEL_M2 = 25.0

_CYR_VOWELS = frozenset("аеёиоуыэюя")
_LAT_VOWELS = frozenset("aeiouy")

# Wall-length in millimetres: bare 4-digit integer (BTI dimension style,
# e.g. "2420", "5950"). 1–3 digit integers are excluded — they collide with
# room numbers and apartment numbers.
_LENGTH_MM_PATTERN = re.compile(
    r"^\s*([1-9]\d{3})\s*$",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_numeric_ocr_label(text: str) -> bool:
    """
    Return True when *text* looks like a numeric area / measurement label.

    Accepted:  "12.5", "12,5", "8", "12.50 м²", "15.3м2", "8.4 кв.м",
               "9.0 m²", "12.5M2"
    Rejected:  "Кухня", "WC", "Ванная", "Коридор", ""

    The check strips common area-unit suffixes then verifies the remainder
    is an integer or decimal number (dot or comma as separator).
    """
    if not text or not text.strip():
        return False
    return bool(_NUMERIC_LABEL_RE.match(text))


def parse_area_m2(text: str) -> Optional[float]:
    """
    Parse *text* as a room area in square metres.

    Accepts: '12.4', '12,4', '12.4 м²', '52.0 m2', '8,5 кв.м' etc.
    Returns ``None`` if the text does not match.
    """
    m = _AREA_PATTERN.match(text.strip())
    if m:
        try:
            return float(m.group(1).replace(",", "."))
        except ValueError:
            pass
    return None


def is_apartment_total_m2(area_m2: Optional[float],
                          max_room_m2: float = MAX_ROOM_LABEL_M2) -> bool:
    """True when *area_m2* is too large to be a single room's printed area.

    Floor plans carry two kinds of numeric area label and OCR cannot tell them
    apart: the per-room area written inside each space, and the apartment total
    printed in the header/stamp.  Adopting a total as a room area is what put
    45–150 m² "rooms" in the export and skewed every scale calibrator that
    paired it with a polygon (audit 28.07.2026: 152/695 labels = 22%).

    ``max_room_m2`` is a *label* ceiling, not a room-size ceiling: a genuine
    open-plan living space can exceed it, but its printed label almost never
    does on the residential plans this pipeline handles, whereas totals almost
    always do.
    """
    return area_m2 is not None and area_m2 > max_room_m2


def parse_room_area_m2(text: str,
                       max_room_m2: float = MAX_ROOM_LABEL_M2) -> Optional[float]:
    """Parse *text* as a single room's area, rejecting apartment totals.

    Same grammar as :func:`parse_area_m2`; returns ``None`` for values above
    ``max_room_m2`` so callers cannot accidentally size or name a room from
    the apartment total.  Use :func:`parse_area_m2` when the total is wanted
    (e.g. an apartment-level area sum).
    """
    area = parse_area_m2(text)
    if area is None or is_apartment_total_m2(area, max_room_m2):
        return None
    return area


def looks_like_room_name(text: str) -> bool:
    """True when *text* plausibly reads as a room name rather than OCR noise.

    "≥3 letters" alone let watermark and hatch-pattern shrapnel through as room
    names ("AAAA", "UUV", "Jl Ill"): 124 of 819 exported names (15%) were junk
    in the 28.07.2026 audit.  A real room name — Russian or Latin — has vowels
    and does not repeat one letter forever, so require:

    * at least 3 letters, all from one alphabet;
    * at least one vowel in that alphabet;
    * no letter taking more than half of the string;
    * no run of 3+ identical letters.
    """
    if not text:
        return False
    s = text.strip()
    letters = [ch for ch in s if ch.isalpha()]
    if len(letters) < 3:
        return False

    low = [ch.lower() for ch in letters]
    cyr = sum(1 for ch in low if 'а' <= ch <= 'я' or ch == 'ё')
    lat = sum(1 for ch in low if 'a' <= ch <= 'z')
    if cyr and lat:
        return False                      # mixed alphabets — OCR confusion
    vowels = _CYR_VOWELS if cyr else _LAT_VOWELS
    if not any(ch in vowels for ch in low):
        return False
    if max(low.count(ch) for ch in set(low)) > len(low) / 2.0:
        return False
    for i in range(len(low) - 2):
        if low[i] == low[i + 1] == low[i + 2]:
            return False
    return True


def parse_length_m(text: str) -> Optional[float]:
    """
    Parse *text* as a wall length in metres (e.g. '3.92', '2,55').

    Only accepts a bare decimal; unit-suffixed strings are rejected so that
    area labels do not accidentally match.
    Returns the value in *metres* (no conversion — the caller's responsibility).
    """
    m = _LENGTH_PATTERN.match(text.strip())
    if m:
        try:
            val = float(m.group(1).replace(",", "."))
            # Sanity: wall lengths on floor plans are 0.1 m – 50 m
            if 0.1 <= val <= 50.0:
                return val
        except ValueError:
            pass
    # BTI-style millimetre dimension: "2420" -> 2.42 m
    m = _LENGTH_MM_PATTERN.match(text.strip())
    if m:
        try:
            return float(m.group(1)) / 1000.0
        except ValueError:
            pass
    return None
