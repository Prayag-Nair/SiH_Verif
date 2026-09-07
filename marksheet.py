"""
Marksheet verification.

The user selects exactly five subjects before verification.

The system then:
1. Finds those subjects in the OCR text.
2. Extracts the marks belonging to those subjects.
3. Ignores additional subjects.
4. Calculates the selected-subject total.
5. Calculates percentage from the selected subjects.
6. Checks that selected marks are within their maximum.
7. Does not trust unrelated OCR rows.

This is an internal-consistency check, not official board verification.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Reason codes
# ---------------------------------------------------------------------------

SELECTION_COUNT_INVALID = "MARKSHEET_SELECTION_COUNT_INVALID"
SELECTED_SUBJECT_NOT_FOUND = "SELECTED_SUBJECT_NOT_FOUND"
MARKSHEET_SELECTION_OK = "MARKSHEET_SELECTION_OK"

MARKS_EXCEED_MAXIMUM = "MARKS_EXCEED_MAXIMUM"
TOTAL_CONSISTENT = "TOTAL_CONSISTENT"
TOTAL_MISMATCH = "TOTAL_MISMATCH"
PERCENTAGE_MISMATCH = "PERCENTAGE_MISMATCH"
PERCENTAGE_IMPOSSIBLE = "PERCENTAGE_IMPOSSIBLE"

BASELINE_IRREGULARITY_ADVISORY = "BASELINE_IRREGULARITY_ADVISORY"
MARKSHEET_NO_INTERNAL_INCONSISTENCY = "MARKSHEET_NO_INTERNAL_INCONSISTENCY"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MarksheetResult:
    subjects: list[dict] = field(default_factory=list)

    selected_subjects: list[str] = field(default_factory=list)
    selected_total: int | None = None
    selected_maximum: int | None = None
    selected_percentage: float | None = None

    # Kept for compatibility with the existing API.
    computed_total: int | None = None
    printed_total: int | None = None
    printed_percentage: float | None = None

    baseline_outliers: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalise_subject(value: str) -> str:
    """
    Normalize a subject name so OCR errors, punctuation and repeated spaces
    don't prevent matching.
    """

    value = value.upper()

    # Common OCR substitutions.
    value = value.replace("&", " AND ")

    value = re.sub(r"[^A-Z0-9 ]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        _normalise_subject(left),
        _normalise_subject(right),
    ).ratio()


# ---------------------------------------------------------------------------
# Subject matching
# ---------------------------------------------------------------------------

def _find_best_subject_line(
    selected_subject: str,
    lines: list[str],
) -> tuple[str | None, float]:
    """
    Find the OCR line that most closely matches the selected subject.

    A fuzzy match is intentional because Tesseract may produce:
        ENGUSH LNG & LIT.
    instead of:
        ENGLISH LNG & LIT.
    """

    wanted = _normalise_subject(selected_subject)

    best_line = None
    best_score = 0.0

    for line in lines:
        clean_line = _normalise_subject(line)

        if not clean_line:
            continue

        # Strong match when the selected subject appears inside the line.
        if wanted in clean_line:
            score = 1.0
        else:
            # Compare against the full line.
            score = _similarity(wanted, clean_line)

            # Also compare against chunks of the line.
            words = clean_line.split()

            for start in range(len(words)):
                for end in range(start + 1, min(len(words), start + 10) + 1):
                    chunk = " ".join(words[start:end])
                    score = max(
                        score,
                        _similarity(wanted, chunk),
                    )

        if score > best_score:
            best_score = score
            best_line = line

    return best_line, best_score


# ---------------------------------------------------------------------------
# Mark extraction
# ---------------------------------------------------------------------------

def _extract_numbers_from_line(line: str) -> list[int]:
    """
    Extract numeric values from one OCR line.

    Example:
        184 | ENGLISH LNG & LIT. 079 020 099

    becomes:
        [184, 79, 20, 99]
    """

    return [
        int(value)
        for value in re.findall(r"\b\d{1,3}\b", line)
    ]


def _extract_subject_marks(
    line: str,
) -> tuple[int | None, int | None, int | None]:
    """
    Extract theory, internal-assessment and total marks.

    Typical CBSE line:
        184 | ENGLISH LNG & LIT. 079 020 099

    Numeric values:
        184 = subject code
        079 = theory
        020 = internal
        099 = total

    We therefore use the last 3 numeric values when available.

    For an OCR line with only one mark value, that value is treated as the
    total when there is no evidence of separate components.
    """

    numbers = _extract_numbers_from_line(line)

    if not numbers:
        return None, None, None

    # Remove likely subject code at the beginning.
    if len(numbers) >= 4:
        marks = numbers[-3:]

        theory = marks[0]
        internal = marks[1]
        total = marks[2]

        return theory, internal, total

    if len(numbers) == 3:
        theory = numbers[0]
        internal = numbers[1]
        total = numbers[2]

        return theory, internal, total

    if len(numbers) == 2:
        # Usually theory + total or theory + internal.
        # Use the larger/later value as the total.
        theory = numbers[0]
        total = numbers[1]

        return theory, None, total

    # One number: most likely the total.
    return None, None, numbers[0]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _clean_selection(selected_subjects: list[str]) -> list[str]:
    cleaned = []

    for subject in selected_subjects:
        subject = subject.strip()

        if subject:
            cleaned.append(subject)

    return cleaned


def _validate_selection(
    selected_subjects: list[str],
) -> tuple[bool, list[str]]:
    selected_subjects = _clean_selection(selected_subjects)

    if len(selected_subjects) != 5:
        return False, [SELECTION_COUNT_INVALID]

    normalized = [
        _normalise_subject(subject)
        for subject in selected_subjects
    ]

    if len(set(normalized)) != 5:
        return False, [SELECTION_COUNT_INVALID]

    return True, []


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    ocr_text: str,
    image_bytes: bytes | None = None,
    selected_subjects: list[str] | None = None,
) -> MarksheetResult:

    result = MarksheetResult()

    selected_subjects = selected_subjects or []

    valid_selection, selection_reasons = _validate_selection(
        selected_subjects
    )

    result.selected_subjects = _clean_selection(
        selected_subjects
    )

    if not valid_selection:
        result.reasons.extend(selection_reasons)
        return result

    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    found_subjects: list[dict] = []

    # ---------------------------------------------------------
    # Find ONLY the five requested subjects.
    # ---------------------------------------------------------

    for selected_subject in result.selected_subjects:

        line, score = _find_best_subject_line(
            selected_subject,
            lines,
        )

        # Require a reasonably strong match.
        if line is None or score < 0.65:
            found_subjects.append(
                {
                    "subject": selected_subject,
                    "selected": True,
                    "found": False,
                    "match_score": round(score, 3),
                    "obtained": None,
                    "maximum": 100,
                    "theory": None,
                    "internal": None,
                    "total": None,
                }
            )

            result.reasons.append(
                SELECTED_SUBJECT_NOT_FOUND
            )

            continue

        theory, internal, total = _extract_subject_marks(
            line
        )

        # CBSE total marks are normally 100.
        maximum = 100

        found_subjects.append(
            {
                "subject": selected_subject,
                "matched_ocr_subject": line,
                "selected": True,
                "found": True,
                "match_score": round(score, 3),
                "theory": theory,
                "internal": internal,
                "obtained": total,
                "total": total,
                "maximum": maximum,
            }
        )

    result.subjects = found_subjects

    # ---------------------------------------------------------
    # Stop if one of the five requested subjects wasn't found.
    # ---------------------------------------------------------

    missing = [
        item
        for item in found_subjects
        if not item["found"]
    ]

    if missing:
        return result

    result.reasons.append(
        MARKSHEET_SELECTION_OK
    )

    # ---------------------------------------------------------
    # Validate individual marks.
    # ---------------------------------------------------------

    for row in found_subjects:

        obtained = row["obtained"]
        maximum = row["maximum"]

        if obtained is None:
            continue

        if obtained < 0 or obtained > maximum:
            result.reasons.append(
                MARKS_EXCEED_MAXIMUM
            )
            row["flag"] = "exceeds_maximum"

    # ---------------------------------------------------------
    # Calculate ONLY the five selected subjects.
    # ---------------------------------------------------------

    obtained_marks = [
        row["obtained"]
        for row in found_subjects
        if row["obtained"] is not None
    ]

    maxima = [
        row["maximum"]
        for row in found_subjects
        if row["maximum"] is not None
    ]

    if len(obtained_marks) == 5:
        result.selected_total = sum(
            obtained_marks
        )

        result.selected_maximum = sum(
            maxima
        )

        result.selected_percentage = round(
            100 * result.selected_total / result.selected_maximum,
            2,
        )

        # Compatibility with existing response.
        result.computed_total = result.selected_total

    # ---------------------------------------------------------
    # Sanity-check percentage.
    # ---------------------------------------------------------

    if (
        result.selected_percentage is not None
        and (
            result.selected_percentage < 0
            or result.selected_percentage > 100
        )
    ):
        result.reasons.append(
            PERCENTAGE_IMPOSSIBLE
        )

    # ---------------------------------------------------------
    # Baseline analysis remains advisory.
    # ---------------------------------------------------------

    if image_bytes:
        result.baseline_outliers = _baseline_outliers(
            image_bytes
        )

        if result.baseline_outliers:
            result.reasons.append(
                BASELINE_IRREGULARITY_ADVISORY
            )

    # ---------------------------------------------------------
    # Remove duplicate reason codes.
    # ---------------------------------------------------------

    result.reasons = list(
        dict.fromkeys(result.reasons)
    )

    # If everything selected was found and there are no structural
    # problems, report internal consistency.
    structural_problems = {
        SELECTION_COUNT_INVALID,
        SELECTED_SUBJECT_NOT_FOUND,
        MARKS_EXCEED_MAXIMUM,
        TOTAL_MISMATCH,
        PERCENTAGE_MISMATCH,
        PERCENTAGE_IMPOSSIBLE,
    }

    if not any(
        reason in structural_problems
        for reason in result.reasons
    ):
        result.reasons.append(
            MARKSHEET_NO_INTERNAL_INCONSISTENCY
        )

    return result


# ---------------------------------------------------------------------------
# Baseline analysis
# ---------------------------------------------------------------------------

def _baseline_outliers(
    image_bytes: bytes,
) -> list[int]:
    """
    Row-wise ink projection.

    This remains advisory only.
    """

    image = ImageOps.autocontrast(
        Image.open(
            io.BytesIO(image_bytes)
        ).convert("L")
    )

    array = 255 - np.asarray(
        image,
        dtype=np.float32,
    )

    profile = array.mean(axis=1)

    threshold = (
        profile.mean()
        + 0.5 * profile.std()
    )

    bands: list[tuple[int, int]] = []

    start = None

    for index, value in enumerate(profile):

        if value > threshold and start is None:
            start = index

        elif (
            value <= threshold
            and start is not None
        ):
            if index - start >= 6:
                bands.append(
                    (start, index)
                )

            start = None

    if start is not None:
        bands.append(
            (start, len(profile))
        )

    if len(bands) < 3:
        return []

    centers = [
        (start + end) / 2
        for start, end in bands
    ]

    outliers: list[int] = []

    for index in range(1, len(centers) - 1):

        previous_gap = (
            centers[index]
            - centers[index - 1]
        )

        next_gap = (
            centers[index + 1]
            - centers[index]
        )

        if previous_gap <= 0:
            continue

        ratio = next_gap / previous_gap

        if ratio > 2.0 or ratio < 0.5:
            outliers.append(
                int(centers[index])
            )

    return outliers[:20]