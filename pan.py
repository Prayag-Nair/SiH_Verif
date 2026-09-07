"""
PAN structural validation.

A PAN is 10 characters: AAAAA9999A
    positions 1-3  arbitrary letters (sequence assigned by the department)
    position  4    holder category, a fixed alphabet
    position  5    first letter of surname (individual) or entity name
    positions 6-9  numeric sequence
    position 10    alphabetic check character

Be honest about what this module is. There is no public check-digit algorithm
for PAN and no offline cryptographic material, so this is format validation and
nothing more. The only authoritative PAN check is the Income Tax
Department's own verification API, which needs registration. Say exactly that
when judge question #6 comes up rather than letting the dashboard imply a
lookup happened.
"""

from __future__ import annotations

import re

PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# Fourth character encodes the holder type.
HOLDER_TYPES = {
    "P": "Individual",
    "C": "Company",
    "H": "Hindu Undivided Family",
    "A": "Association of Persons",
    "B": "Body of Individuals",
    "G": "Government Agency",
    "J": "Artificial Juridical Person",
    "L": "Local Authority",
    "F": "Firm / LLP",
    "T": "Trust",
}


def validate_pan(candidate: str) -> dict:
    """Returns a structural report. `verified` is intentionally absent."""
    value = re.sub(r"[^A-Za-z0-9]", "", candidate or "").upper()
    reasons: list[str] = []

    if len(value) != 10:
        return {
            "input_length": len(value),
            "format_ok": False,
            "holder_type": None,
            "reasons": ["PAN_LENGTH_INVALID"],
        }

    format_ok = bool(PAN_PATTERN.match(value))
    if not format_ok:
        reasons.append("PAN_PATTERN_INVALID")

    holder_code = value[3]
    holder_type = HOLDER_TYPES.get(holder_code)
    if holder_type is None:
        format_ok = False
        reasons.append("PAN_HOLDER_TYPE_UNKNOWN")

    if format_ok:
        reasons.append("PAN_FORMAT_OK")

    return {
        "masked": value[:2] + "*" * 6 + value[-2:],
        "format_ok": format_ok,
        "holder_type": holder_type,
        "surname_initial": value[4] if format_ok else None,
        "reasons": reasons,
    }


def cross_check_name(pan_value: str, ocr_name: str) -> dict:
    """
    Weak consistency signal: position 5 of a PAN should be the first letter of
    the holder's surname. If OCR pulled a name off the card, a mismatch is
    mildly interesting. Do not flag on this alone. Surname ordering on Indian
    documents is inconsistent, and OCR misreads are common, so this is a hint
    for a human reviewer, not a verdict input.
    """
    value = re.sub(r"[^A-Za-z0-9]", "", pan_value or "").upper()
    if not PAN_PATTERN.match(value) or not ocr_name:
        return {"checked": False, "match": None, "reasons": []}

    tokens = [token for token in re.split(r"\s+", ocr_name.strip().upper()) if token]
    if not tokens:
        return {"checked": False, "match": None, "reasons": []}

    expected = value[4]
    initials = {token[0] for token in tokens}
    match = expected in initials
    return {
        "checked": True,
        "match": match,
        "reasons": [] if match else ["PAN_SURNAME_INITIAL_MISMATCH"],
    }
