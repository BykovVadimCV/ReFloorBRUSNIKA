"""
core/ocr_utils.py — Canonical OCR label parsing helpers.

Single source of truth for all numeric-label detection and area/length
parsing.  Replaces five scattered implementations that disagreed on which
unit suffixes, decimal separators, and integer forms to accept.

Public API
----------
is_numeric_ocr_label(text)  -> bool
parse_area_m2(text)         -> Optional[float]
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
    return None
