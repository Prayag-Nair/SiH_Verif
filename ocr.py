"""
OCR wrapper for VERIFai.

General OCR:
    - Tesseract is the primary engine.
    - EasyOCR is used as a fallback when Tesseract is unavailable.

PAN OCR:
    - Uses multiple preprocessing passes.
    - OCRs the whole document and the likely PAN-number region.
    - Uses PAN-specific Tesseract configurations.
    - Recovers common OCR character mistakes.
    - Ranks PAN candidates instead of trusting the first OCR match.

This module extracts text only. PAN validation remains in pan.py.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from functools import lru_cache

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

AADHAAR_NUMBER = re.compile(
    r"\b(\d{4}\s?\d{4}\s?\d{4})\b"
)

PAN_NUMBER = re.compile(
    r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
)

DATE = re.compile(
    r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b"
)

YEAR = re.compile(
    r"\b(19\d{2}|20\d{2})\b"
)


# ---------------------------------------------------------------------------
# OCR result
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    engine: str
    text: str
    aadhaar_numbers: list[str] = field(default_factory=list)
    pan_numbers: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    years: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# General preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_bytes: bytes) -> Image.Image:
    """
    Original/general preprocessing path.

    Used for Aadhaar and general OCR.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = ImageOps.autocontrast(image)

    if max(image.size) < 1600:
        image = image.resize(
            (image.width * 2, image.height * 2),
            Image.LANCZOS,
        )

    return image


# ---------------------------------------------------------------------------
# PAN-specific preprocessing
# ---------------------------------------------------------------------------

def _load_color_image(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _upscale(image: Image.Image, scale: int = 3) -> Image.Image:
    return image.resize(
        (image.width * scale, image.height * scale),
        Image.LANCZOS,
    )


def _gray(image: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(
        image.convert("L")
    )


def _sharpen(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def _threshold(image: Image.Image, threshold: int = 165) -> Image.Image:
    gray = image.convert("L")
    return gray.point(
        lambda pixel: 255 if pixel > threshold else 0
    )


def _pan_regions(image: Image.Image) -> list[Image.Image]:
    """
    Generate likely PAN-number regions.

    Indian PAN cards generally place the PAN number around the upper-middle
    portion of the card. We deliberately use overlapping regions so that
    slightly different layouts/crops still have a chance of being detected.
    """

    width, height = image.size

    regions = []

    # 1. Broad upper-middle region.
    regions.append(
        image.crop(
            (
                int(width * 0.20),
                int(height * 0.25),
                int(width * 0.78),
                int(height * 0.62),
            )
        )
    )

    # 2. Narrower PAN-number area.
    regions.append(
        image.crop(
            (
                int(width * 0.28),
                int(height * 0.32),
                int(width * 0.70),
                int(height * 0.56),
            )
        )
    )

    # 3. Center region for cards where the PAN is shifted.
    regions.append(
        image.crop(
            (
                int(width * 0.30),
                int(height * 0.25),
                int(width * 0.85),
                int(height * 0.65),
            )
        )
    )

    return regions


def _pan_preprocessed_images(image_bytes: bytes) -> list[Image.Image]:
    """
    Build several PAN OCR images.

    We keep the list intentionally small so verification remains fast.
    """

    original = _load_color_image(image_bytes)

    images: list[Image.Image] = []

    # Whole-card variants.
    gray = _gray(original)
    gray = _upscale(gray, 3)

    images.append(gray)
    images.append(_sharpen(gray))

    # Threshold variants.
    images.append(_threshold(gray, 150))
    images.append(_threshold(gray, 180))

    # Targeted PAN regions.
    for region in _pan_regions(original):
        region_gray = _gray(region)
        region_gray = _upscale(region_gray, 4)

        images.append(region_gray)
        images.append(_sharpen(region_gray))
        images.append(_threshold(region_gray, 150))
        images.append(_threshold(region_gray, 180))

    return images


# ---------------------------------------------------------------------------
# PAN OCR normalization
# ---------------------------------------------------------------------------

LETTER_TO_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "T": "7",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}


def _clean_pan_text(value: str) -> str:
    """
    Remove characters that are unlikely to belong to a PAN.

    Spaces and punctuation are removed because OCR frequently returns
    something such as:

        QFVPS 0764 H
        QFVPS-0764-H
    """

    value = value.upper()

    value = value.replace("|", "I")
    value = value.replace("—", "-")
    value = value.replace("–", "-")

    return re.sub(r"[^A-Z0-9]", "", value)


def _correct_pan_candidate(candidate: str) -> str:
    """
    Correct OCR confusions according to the expected PAN structure.

    PAN structure:

        A A A A A 9 9 9 9 A

    Therefore positions 1-5 and 10 should be letters.
    Positions 6-9 should be digits.
    """

    value = _clean_pan_text(candidate)

    if len(value) != 10:
        return value

    chars = list(value)

    # First five characters must be letters.
    for index in range(5):
        char = chars[index]

        if char in DIGIT_TO_LETTER:
            chars[index] = DIGIT_TO_LETTER[char]

    # Characters 6-9 must be digits.
    for index in range(5, 9):
        char = chars[index]

        if char in LETTER_TO_DIGIT:
            chars[index] = LETTER_TO_DIGIT[char]

    # Final character must be a letter.
    if chars[9] in DIGIT_TO_LETTER:
        chars[9] = DIGIT_TO_LETTER[chars[9]]

    return "".join(chars)


def _is_valid_pan_candidate(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"[A-Z]{5}[0-9]{4}[A-Z]",
            value,
        )
    )


def _extract_pan_candidates(text: str) -> list[str]:
    """
    Extract PAN candidates from OCR text.

    This first searches for exact PANs, then searches for 10-character
    alphanumeric sequences that may contain OCR errors.
    """

    candidates: list[str] = []

    upper = text.upper()

    # ---------------------------------------------------------
    # Pass 1: exact PAN regex
    # ---------------------------------------------------------

    for match in PAN_NUMBER.findall(upper):
        value = _correct_pan_candidate(match)

        if _is_valid_pan_candidate(value):
            candidates.append(value)

    # ---------------------------------------------------------
    # Pass 2: compact alphanumeric chunks
    # ---------------------------------------------------------

    compact = re.sub(
        r"[^A-Z0-9]",
        "",
        upper,
    )

    # Search every possible 10-character window.
    for index in range(max(0, len(compact) - 9)):
        candidate = compact[index:index + 10]

        if len(candidate) != 10:
            continue

        corrected = _correct_pan_candidate(candidate)

        if _is_valid_pan_candidate(corrected):
            candidates.append(corrected)

    # ---------------------------------------------------------
    # Pass 3: line-based extraction
    # ---------------------------------------------------------

    for line in upper.splitlines():
        compact_line = _clean_pan_text(line)

        if len(compact_line) >= 10:
            for index in range(len(compact_line) - 9):
                candidate = compact_line[index:index + 10]
                corrected = _correct_pan_candidate(candidate)

                if _is_valid_pan_candidate(corrected):
                    candidates.append(corrected)

    # Preserve order while removing duplicates.
    unique: list[str] = []

    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)

    return unique


def _score_pan_candidate(candidate: str) -> int:
    """
    Score a PAN candidate.

    A structurally valid candidate receives a strong score.
    The fourth character is especially useful because it represents
    the holder category.
    """

    if not _is_valid_pan_candidate(candidate):
        return -100

    score = 100

    holder_codes = set("PCHABGJLFT")

    if candidate[3] in holder_codes:
        score += 20
    else:
        score -= 30

    return score


def _best_pan_candidate(candidates: list[str]) -> str | None:
    if not candidates:
        return None

    ranked = sorted(
        candidates,
        key=_score_pan_candidate,
        reverse=True,
    )

    return ranked[0]


# ---------------------------------------------------------------------------
# EasyOCR fallback
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(
        ["en"],
        gpu=False,
    )


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_text(image_bytes: bytes) -> OCRResult:
    """
    Extract text, Aadhaar numbers, PAN numbers, dates and years.

    PAN extraction gets additional targeted OCR passes.
    """

    # ---------------------------------------------------------
    # General OCR
    # ---------------------------------------------------------

    general_image = _preprocess(image_bytes)

    engine = "none"
    text_parts: list[str] = []

    try:
        import pytesseract

        general_text = pytesseract.image_to_string(
            general_image,
            config="--oem 3 --psm 6",
        )

        text_parts.append(general_text)
        engine = "tesseract"

    except Exception:
        try:
            import numpy as np

            lines = _easyocr_reader().readtext(
                np.asarray(general_image),
                detail=0,
            )

            general_text = "\n".join(lines)
            text_parts.append(general_text)
            engine = "easyocr"

        except Exception:
            general_text = ""
            engine = "unavailable"

    # ---------------------------------------------------------
    # PAN-specific OCR
    # ---------------------------------------------------------

    pan_candidates: list[str] = []

    try:
        import pytesseract

        pan_images = _pan_preprocessed_images(image_bytes)

        # PAN OCR configurations.
        configs = [
            "--oem 3 --psm 6",
            "--oem 3 --psm 7",
            "--oem 3 --psm 11",
        ]

        for image in pan_images:
            for config in configs:

                try:
                    pan_text = pytesseract.image_to_string(
                        image,
                        config=config,
                    )

                    if pan_text:
                        text_parts.append(pan_text)

                        found = _extract_pan_candidates(
                            pan_text
                        )

                        pan_candidates.extend(found)

                except Exception:
                    continue

    except Exception:
        pass

    # ---------------------------------------------------------
    # Combine OCR text
    # ---------------------------------------------------------

    text = "\n".join(
        part for part in text_parts if part
    )

    upper = text.upper()

    # ---------------------------------------------------------
    # PAN candidates from ALL OCR text
    # ---------------------------------------------------------

    pan_candidates.extend(
        _extract_pan_candidates(text)
    )

    # Remove duplicates.
    pan_candidates = list(
        dict.fromkeys(pan_candidates)
    )

    best_pan = _best_pan_candidate(
        pan_candidates
    )

    final_pan_numbers = (
        [best_pan]
        if best_pan
        else []
    )

    # ---------------------------------------------------------
    # Aadhaar extraction
    # ---------------------------------------------------------

    aadhaar_numbers = [
        match.replace(" ", "")
        for match in AADHAAR_NUMBER.findall(text)
    ]

    aadhaar_numbers = list(
        dict.fromkeys(aadhaar_numbers)
    )

    # ---------------------------------------------------------
    # Dates and years
    # ---------------------------------------------------------

    dates = list(
        dict.fromkeys(
            DATE.findall(text)
        )
    )

    years = list(
        dict.fromkeys(
            YEAR.findall(text)
        )
    )

    return OCRResult(
        engine=engine,
        text=text,
        aadhaar_numbers=aadhaar_numbers,
        pan_numbers=final_pan_numbers,
        dates=dates,
        years=years,
    )