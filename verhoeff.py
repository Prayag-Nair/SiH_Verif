"""
Verhoeff checksum (dihedral group D5).

Aadhaar numbers are 12 digits where the 12th digit is a Verhoeff check digit
computed over the first 11. Verhoeff catches all single-digit errors and all
adjacent transpositions, which is why UIDAI chose it over a plain mod-10 sum.

This is a *mathematical* check on the number itself. It proves nothing about
whether the number was ever issued to anyone. Any forger can generate a
Verhoeff-valid 12-digit number in a loop. Treat a pass as "not obviously
garbage", never as "real".
"""

# Multiplication table for the dihedral group D5.
_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# Permutation table, applied cyclically by digit position.
_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# Multiplicative inverse within D5.
_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _digits_reversed(number: str) -> list[int]:
    cleaned = "".join(ch for ch in number if ch.isdigit())
    return [int(ch) for ch in reversed(cleaned)]


def validate(number: str) -> bool:
    """True if the trailing digit is a correct Verhoeff check digit."""
    digits = _digits_reversed(number)
    if not digits:
        return False
    checksum = 0
    for position, digit in enumerate(digits):
        checksum = _D[checksum][_P[position % 8][digit]]
    return checksum == 0


def check_digit(partial_number: str) -> int:
    """Compute the check digit that should be appended to `partial_number`."""
    digits = _digits_reversed(partial_number)
    checksum = 0
    for position, digit in enumerate(digits):
        checksum = _D[checksum][_P[(position + 1) % 8][digit]]
    return _INV[checksum]


def validate_aadhaar_number(number: str) -> tuple[bool, str]:
    """
    Aadhaar-specific wrapper: enforces length and the leading-digit rule
    (UIDAI does not issue numbers starting with 0 or 1) before the checksum.
    Returns (ok, reason_code).
    """
    cleaned = "".join(ch for ch in number if ch.isdigit())
    if len(cleaned) != 12:
        return False, "AADHAAR_LENGTH_INVALID"
    if cleaned[0] in "01":
        return False, "AADHAAR_LEADING_DIGIT_INVALID"
    if not validate(cleaned):
        return False, "AADHAAR_VERHOEFF_FAILED"
    return True, "AADHAAR_VERHOEFF_OK"
