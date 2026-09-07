"""
Aadhaar QR decoding and UIDAI digital-signature verification.

There are two generations of Aadhaar QR code and they are completely different
animals. Your demo needs to detect which one it is looking at, because only one
of them can actually be verified.

V1 (legacy, printed on older cards)
    Payload is plain XML: <PrintLetterBarcodeData uid="..." name="..." .../>
    There is NO signature. Anyone can author this string in a text editor and
    render a QR from it. If your app reports "genuine" for a V1 QR, your app is
    lying. We return UNSIGNED and refuse to give a genuine verdict.

V2 (Secure QR, current)
    Payload is a very long decimal integer string. Pipeline:
        decimal string -> big integer -> raw bytes -> gzip decompress
        -> 0xFF-delimited text fields, then a JPEG2000 photo,
           then optional SHA-256 mobile/email hashes,
           then a trailing 256-byte RSA signature.
    The signature is SHA256-with-RSA (PKCS#1 v1.5) over every byte that
    precedes it, made with UIDAI's private key. We verify it against UIDAI's
    published public certificate.

    Verifying that signature proves ONE thing precisely: these exact field
    values were issued by UIDAI and have not been altered by a single bit.
    It does NOT prove the person handing you the document is the subject of it.
    That gap is what the face-match module closes.

Field ordering in V2 has shifted between UIDAI spec revisions. Confirm the
index map below against the current "Aadhaar Secure QR Code" spec before the
demo, and keep FIELD_ORDER in one place so a spec change is a one-line fix.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import zlib
from dataclasses import dataclass, field
from typing import Any

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.x509 import load_der_x509_certificate, load_pem_x509_certificate

SIGNATURE_LENGTH = 256  # RSA-2048 signature, always the final 256 bytes
HASH_LENGTH = 32        # SHA-256 digest of mobile / email, when present
DELIMITER = 0xFF

GZIP_MAGIC = b"\x1f\x8b"

# Legacy Secure QR text fields. Some newer Secure QR payloads prepend a
# format/version field (for example "V5") before the contact-present flag.
# Keep the canonical demographic order separate so the parser can support
# both layouts without silently shifting every field by one position.
FIELD_ORDER = (
    "email_mobile_flag",
    "reference_id",
    "name",
    "dob",
    "gender",
    "care_of",
    "district",
    "landmark",
    "house",
    "location",
    "pincode",
    "post_office",
    "state",
    "street",
    "sub_district",
    "vtc",
)

VERSION_FIELD = "qr_format_version"


class QRDecodeError(Exception):
    """Payload could not be decoded as any known Aadhaar QR format."""


@dataclass
class AadhaarQRResult:
    version: str                       # "V1" | "V2" | "UNKNOWN"
    signature_verified: bool | None    # None when the format carries no signature
    fields: dict[str, Any] = field(default_factory=dict)
    photo_jp2: bytes | None = None     # raw JPEG2000 bytes, held in memory only
    reasons: list[str] = field(default_factory=list)

    def redacted(self) -> dict[str, Any]:
        """Field view safe to log or render on a demo screen."""
        out = dict(self.fields)
        ref = out.get("reference_id") or ""
        if len(ref) >= 4:
            # reference_id starts with the last 4 digits of the Aadhaar number.
            out["reference_id"] = "XXXX" + ref[4:]
        out.pop("photo_present", None)
        return out


# ---------------------------------------------------------------------------
# QR image -> payload string
# ---------------------------------------------------------------------------

def read_qr_payloads(image_bytes: bytes) -> list[str]:
    """
    Extract QR payloads from real-world Aadhaar photos and screenshots.

    The QR on the newer two-sided card can be relatively small in a phone
    screenshot. A single full-image pyzbar pass is therefore too brittle.

    We try:
      1. the original image,
      2. grayscale/autocontrast,
      3. 2x and 4x upscaling,
      4. contrast + sharpening,
      5. adaptive and Otsu thresholding,
      6. OpenCV QRCodeDetector as a second decoder.

    Cropping is intentionally conservative: first try the whole image, then
    likely QR regions, so we don't accidentally throw away a QR that is not
    in the expected location.
    """
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    from pyzbar.pyzbar import decode as zbar_decode

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    candidates: list[Image.Image] = [image]

    gray = ImageOps.autocontrast(image.convert("L"))
    candidates.append(gray)

    # Larger QR modules become much easier for barcode decoders to resolve.
    for scale in (2, 3, 4):
        candidates.append(
            gray.resize(
                (gray.width * scale, gray.height * scale),
                Image.Resampling.LANCZOS,
            )
        )

    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    candidates.append(enhanced.resize(
        (enhanced.width * 3, enhanced.height * 3),
        Image.Resampling.LANCZOS,
    ))

    # Adaptive thresholding is useful when the card has uneven lighting.
    try:
        import cv2
        import numpy as np

        gray_np = np.asarray(gray, dtype=np.uint8)
        blurred = cv2.GaussianBlur(gray_np, (3, 3), 0)

        otsu = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )

        for processed in (otsu, adaptive):
            pil = Image.fromarray(processed)
            candidates.append(
                pil.resize(
                    (pil.width * 3, pil.height * 3),
                    Image.Resampling.NEAREST,
                )
            )
    except ImportError:
        pass

    # Try likely QR areas as well. These are overlapping crops rather than a
    # single hard-coded box, so the detector remains useful if the card is
    # rotated, slightly cropped, or photographed at a different framing.
    width, height = image.size
    qr_regions = [
        (int(width * 0.55), 0, width, int(height * 0.72)),
        (int(width * 0.45), 0, width, int(height * 0.85)),
        (int(width * 0.35), 0, width, height),
    ]

    for box in qr_regions:
        crop = image.crop(box)
        crop_gray = ImageOps.autocontrast(crop.convert("L"))
        for scale in (3, 4):
            candidates.append(
                crop_gray.resize(
                    (crop_gray.width * scale, crop_gray.height * scale),
                    Image.Resampling.LANCZOS,
                )
            )

    seen: list[str] = []

    def add_payload(value: str) -> None:
        value = value.strip()
        if value and value not in seen:
            seen.append(value)

    # Decoder 1: pyzbar / ZBar.
    for candidate in candidates:
        try:
            for symbol in zbar_decode(candidate):
                add_payload(symbol.data.decode("utf-8", errors="replace"))
            if seen:
                return seen
        except Exception:
            continue

    # Decoder 2: OpenCV. This can succeed where ZBar misses a small or noisy
    # QR, and it is already a natural dependency for the face/liveness stack.
    try:
        import cv2
        import numpy as np

        detector = cv2.QRCodeDetector()

        for candidate in candidates:
            array = np.asarray(candidate)
            if array.ndim == 2:
                bgr = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            try:
                value, points, _ = detector.detectAndDecode(bgr)
                if value:
                    add_payload(value)
            except Exception:
                pass

            # Multi-code API can be more tolerant of QR placement.
            try:
                values, _, _ = detector.detectAndDecodeMulti(bgr)
                if values:
                    for value in values:
                        if value:
                            add_payload(value)
            except Exception:
                pass

            if seen:
                return seen
    except ImportError:
        pass

    return seen


# ---------------------------------------------------------------------------
# Payload -> structured result
# ---------------------------------------------------------------------------

def parse_payload(payload: str, uidai_public_key: rsa.RSAPublicKey | None) -> AadhaarQRResult:
    payload = payload.strip()

    if payload.startswith("<?xml") or "PrintLetterBarcodeData" in payload:
        return _parse_v1(payload)

    if payload.isdigit():
        return _parse_v2(payload, uidai_public_key)

    raise QRDecodeError("Payload is neither legacy XML nor a Secure QR integer string")


def _parse_v1(payload: str) -> AadhaarQRResult:
    import xml.etree.ElementTree as ElementTree

    try:
        node = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise QRDecodeError(f"Malformed V1 XML: {exc}") from exc

    attributes = {key.lower(): value for key, value in node.attrib.items()}
    return AadhaarQRResult(
        version="V1",
        signature_verified=None,
        fields={
            "reference_id": (attributes.get("uid") or "")[-4:],
            "name": attributes.get("name", ""),
            "dob": attributes.get("dob") or attributes.get("yob", ""),
            "gender": attributes.get("gender", ""),
            "pincode": attributes.get("pc", ""),
        },
        reasons=["QR_V1_LEGACY_UNSIGNED"],
    )


def _decompress(raw: bytes) -> bytes:
    if raw[:2] == GZIP_MAGIC:
        return gzip.decompress(raw)
    # Some encoders emit a bare deflate stream. Try both wrappers before failing.
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
        try:
            return zlib.decompress(raw, wbits)
        except zlib.error:
            continue
    return raw  # already-plain payloads exist in test fixtures


def _split_text_fields(data: bytes, count: int) -> tuple[list[bytes], int]:
    """
    Take the first `count` 0xFF-delimited fields.

    We deliberately do NOT split the whole buffer on 0xFF: the embedded
    JPEG2000 photo is binary and contains 0xFF bytes in its marker segments,
    so a naive `data.split(b"\\xff")` shreds the image and shifts every
    subsequent offset. Scan for exactly the delimiters we need and stop.
    """
    fields: list[bytes] = []
    start = 0
    cursor = 0
    while len(fields) < count:
        cursor = data.find(bytes([DELIMITER]), start)
        if cursor == -1:
            raise QRDecodeError(
                f"Expected {count} delimited fields, found {len(fields)}"
            )
        fields.append(data[start:cursor])
        start = cursor + 1
    return fields, start


def _parse_v2(payload: str, uidai_public_key: rsa.RSAPublicKey | None) -> AadhaarQRResult:
    try:
        big_integer = int(payload)
    except ValueError as exc:
        raise QRDecodeError("Secure QR payload is not a valid integer") from exc

    byte_length = (big_integer.bit_length() + 7) // 8
    raw = big_integer.to_bytes(byte_length, "big")
    data = _decompress(raw)

    if len(data) <= SIGNATURE_LENGTH:
        raise QRDecodeError("Payload too short to contain a signature")

    reasons: list[str] = []

    # --- signature verification, over everything before the trailing 256 bytes
    signed_region = data[:-SIGNATURE_LENGTH]
    signature = data[-SIGNATURE_LENGTH:]

    signature_verified: bool | None
    if uidai_public_key is None:
        signature_verified = None
        reasons.append("UIDAI_CERT_NOT_CONFIGURED")
    else:
        try:
            uidai_public_key.verify(
                signature,
                signed_region,
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            signature_verified = True
            reasons.append("QR_SIGNATURE_VALID")
        except Exception:
            signature_verified = False
            reasons.append("QR_SIGNATURE_INVALID")

    # --- text fields
    #
    # Older Secure QR payloads start directly with the email/mobile-present
    # flag. Newer payloads can prepend a format marker such as "V5". Detect
    # that marker instead of assuming the first field is always the flag.
    first_fields, cursor_after_first = _split_text_fields(
        signed_region, 1
    )
    first_value = first_fields[0].decode("utf-8", errors="replace").strip()

    if first_value.upper().startswith("V") and first_value[1:].isdigit():
        remaining_fields, photo_start = _split_text_fields(
            signed_region[cursor_after_first:], len(FIELD_ORDER)
        )
        # `_split_text_fields` above starts at zero relative to the slice.
        # Convert its returned offset back into the original signed region.
        photo_start += cursor_after_first

        fields = {
            name: value.decode("utf-8", errors="replace")
            for name, value in zip(FIELD_ORDER, remaining_fields)
        }
        fields[VERSION_FIELD] = first_value
    else:
        text_fields, photo_start = _split_text_fields(
            signed_region, len(FIELD_ORDER)
        )
        fields = {
            name: value.decode("utf-8", errors="replace")
            for name, value in zip(FIELD_ORDER, text_fields)
        }

    # --- trailing contact hashes, counted backwards from the signature
    flag = fields.get("email_mobile_flag", "")
    trailing_hashes = _expected_hash_count(flag)
    photo_end = len(signed_region) - (trailing_hashes * HASH_LENGTH)
    if photo_end <= photo_start:
        raise QRDecodeError("Photo segment resolved to a negative length")

    photo = signed_region[photo_start:photo_end]
    contact_hashes = [
        signed_region[photo_end + index * HASH_LENGTH : photo_end + (index + 1) * HASH_LENGTH]
        for index in range(trailing_hashes)
    ]

    fields["photo_present"] = bool(photo)
    fields["contact_hash_count"] = trailing_hashes

    return AadhaarQRResult(
        version="V2",
        signature_verified=signature_verified,
        fields=fields,
        photo_jp2=photo or None,
        reasons=reasons,
    )


def _expected_hash_count(flag: str) -> int:
    """
    Index 0 of the payload signals which contact fields were hashed in.
    Conventionally: 0=none, 1=email only, 2=mobile only, 3=both.
    Unknown values fall back to 0 rather than guessing an offset, because a
    wrong guess silently corrupts the photo slice.
    """
    digits = "".join(ch for ch in flag if ch.isdigit())
    return {"0": 0, "1": 1, "2": 1, "3": 2}.get(digits[-1:], 0)


def verify_contact_hash(contact_value: str, last_four: str, digest: bytes) -> bool:
    """
    Optional extra binding: confirm a user-supplied mobile/email matches the
    hash inside the signed payload.

    UIDAI's construction is sha256(contact + last_four_of_aadhaar), iterated a
    number of times derived from the last digit. Because the iteration rule has
    changed across revisions, we try single and iterated forms rather than
    hardcoding one. Only useful if the user types their number during the demo.
    """
    base = (contact_value + last_four).encode()
    single = hashlib.sha256(base).digest()
    if single == digest:
        return True

    iterations = int(last_four[-1]) if last_four[-1:].isdigit() else 0
    current = base
    for _ in range(max(iterations, 1)):
        current = hashlib.sha256(current).hexdigest().encode()
    return bytes.fromhex(current.decode()) == digest


# ---------------------------------------------------------------------------
# Certificate loading
# ---------------------------------------------------------------------------

def load_uidai_public_key(path: str) -> rsa.RSAPublicKey | None:
    """
    Load UIDAI's signing certificate (PEM or DER) or a bare public key.

    Download this from UIDAI directly and commit the fingerprint to your repo
    README. If a judge asks how you know the key is genuine, "we pinned the
    fingerprint we fetched from UIDAI" is a real answer; "we found it on
    GitHub" is not.
    """
    try:
        with open(path, "rb") as handle:
            blob = handle.read()
    except OSError:
        return None

    for loader in (load_pem_x509_certificate, load_der_x509_certificate):
        try:
            return loader(blob).public_key()
        except Exception:
            continue
    try:
        return load_pem_public_key(blob)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def verify_aadhaar_image(
    image_bytes: bytes,
    uidai_public_key: rsa.RSAPublicKey | None,
) -> AadhaarQRResult:
    payloads = read_qr_payloads(image_bytes)
    if not payloads:
        return AadhaarQRResult(
            version="UNKNOWN",
            signature_verified=None,
            reasons=["QR_NOT_FOUND"],
        )

    last_error: Exception | None = None
    for payload in payloads:
        try:
            return parse_payload(payload, uidai_public_key)
        except QRDecodeError as exc:
            last_error = exc
    return AadhaarQRResult(
        version="UNKNOWN",
        signature_verified=None,
        reasons=["QR_DECODE_FAILED", str(last_error or "")],
    )
