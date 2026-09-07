"""
Verdict engine.

This is the part of your plan I changed, so here is the argument.

Your table lists seven checks feeding one verified/flagged verdict. If those
get averaged or weighted into a score, two bad things happen. First, a genuine
document photographed badly picks up ELA noise and gets marked down despite
carrying a valid UIDAI signature, which is a mathematical proof of issuance.
Second, and worse, a forged document that happens to have a clean compression
history gains points it did not earn. A proof and a heuristic do not belong in
the same arithmetic.

So checks are sorted into tiers and the strongest available tier decides.

    Tier 1  CRYPTOGRAPHIC   UIDAI QR signature.
                            Binary, authoritative. If present it decides,
                            full stop. Nothing in Tier 2 or 3 can overturn it
                            in either direction.
    Tier 2  STRUCTURAL      Verhoeff, PAN pattern, marksheet arithmetic,
                            field cross-consistency.
                            Cheap to forge, so a pass means little, but a
                            failure is a hard, explainable flag.
    Tier 3  HEURISTIC       ELA, baseline alignment.
                            Advisory only. Never the sole cause of a flag.
                            Surfaced to a human reviewer with a heatmap.

And the verdict vocabulary is deliberately not "verified/flagged", because for
PAN and marksheets you have no authoritative source and "verified" would be a
false claim. Five outcomes:

    GENUINE_SIGNED       Tier 1 passed. Issuer-signed data, unaltered.
    FORGED_SIGNATURE     Tier 1 failed. The strongest claim you can make.
    STRUCTURALLY_INVALID Tier 2 failed. Explainable, document-specific.
    UNVERIFIABLE         No Tier 1 available and Tier 2 clean. This is the
                         honest answer for every PAN and every marksheet.
                         It is not a pass.
    NEEDS_REVIEW         Tier 3 raised something on an otherwise clean doc.

Note that GENUINE_SIGNED still does not mean "this person is who they say".
Document authenticity and identity binding are separate axes, and
`identity_binding` below reports the second one independently. Keep them
separate in the UI too. Collapsing them is how the "forger copies a real QR"
question sinks a demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Tier(str, Enum):
    CRYPTOGRAPHIC = "cryptographic"
    STRUCTURAL = "structural"
    HEURISTIC = "heuristic"


class Verdict(str, Enum):
    GENUINE_SIGNED = "GENUINE_SIGNED"
    FORGED_SIGNATURE = "FORGED_SIGNATURE"
    STRUCTURALLY_INVALID = "STRUCTURALLY_INVALID"
    UNVERIFIABLE = "UNVERIFIABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Binding(str, Enum):
    BOUND = "BOUND"                  # live face matched the ID photo, liveness ok
    NOT_BOUND = "NOT_BOUND"          # face did not match
    LIVENESS_FAILED = "LIVENESS_FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


# Human-readable text for every reason code the services can emit. The
# dashboard should render these rather than raw codes, and any code missing
# from this map is a bug worth failing loudly on in tests.
REASON_TEXT: dict[str, str] = {
    "QR_SIGNATURE_VALID": "UIDAI digital signature verified. Field data is issuer-signed and unaltered.",
    "QR_SIGNATURE_INVALID": "UIDAI digital signature did not verify. Data was altered after issuance, or the QR was fabricated.",
    "QR_V1_LEGACY_UNSIGNED": "Legacy V1 QR code carries no signature and cannot be verified.",
    "QR_NOT_FOUND": "No QR code detected in the uploaded image.",
    "QR_DECODE_FAILED": "QR code found but its payload did not match any known Aadhaar format.",
    "UIDAI_CERT_NOT_CONFIGURED": "UIDAI public certificate not loaded, so signature checking was skipped.",
    "AADHAAR_VERHOEFF_OK": "Aadhaar number passes the Verhoeff checksum.",
    "AADHAAR_VERHOEFF_FAILED": "Aadhaar number fails the Verhoeff checksum and cannot have been issued.",
    "AADHAAR_LENGTH_INVALID": "Aadhaar number is not 12 digits.",
    "AADHAAR_LEADING_DIGIT_INVALID": "Aadhaar numbers are not issued starting with 0 or 1.",
    "PAN_FORMAT_OK": "PAN matches the expected structure.",
    "PAN_PATTERN_INVALID": "PAN does not match the five-letter, four-digit, one-letter structure.",
    "PAN_LENGTH_INVALID": "PAN is not 10 characters.",
    "PAN_HOLDER_TYPE_UNKNOWN": "Fourth character of the PAN is not a recognised holder-type code.",
    "PAN_SURNAME_INITIAL_MISMATCH": "Fifth PAN character does not match any name initial read from the card.",
    "MARKS_EXCEED_MAXIMUM": "A subject score exceeds its stated maximum.",
    "TOTAL_MISMATCH": "Subject marks do not sum to the printed total.",
    "TOTAL_CONSISTENT": "Subject marks sum to the printed total.",
    "PERCENTAGE_MISMATCH": "Printed percentage does not follow from the printed marks.",
    "PERCENTAGE_IMPOSSIBLE": "Printed percentage exceeds 100.",
    "MARKSHEET_NO_INTERNAL_INCONSISTENCY": "No internal inconsistency found. This is not board verification.",
    "BASELINE_IRREGULARITY_ADVISORY": "Some text rows sit off the common baseline. Often benign; review the highlighted rows.",
    "ELA_LOCALISED_ANOMALY": "Compression analysis shows a localised anomaly. Advisory only, and common around printed text.",
    "ELA_NO_LOCALISED_ANOMALY": "Compression analysis found no localised anomaly.",
    "ELA_NOT_APPLICABLE_NON_JPEG": "Image is not a JPEG, so compression analysis cannot be applied.",
    "FACE_MATCH": "Live face matches the photo on the document.",
    "FACE_NO_MATCH": "Live face does not match the photo on the document.",
    "FACE_NOT_DETECTED": "No face could be located in one of the images.",
    "FACE_MATCH_ERROR": "Face comparison failed to run.",
    "LIVENESS_BLINK_OK": "Blink challenge satisfied across frames.",
    "LIVENESS_BLINK_FAILED": "Blink challenge not satisfied. A still photo would fail this way.",
    "LIVENESS_HEAD_TURN_OK": "Head-turn challenge satisfied across frames.",
    "LIVENESS_HEAD_TURN_FAILED": "Head-turn challenge not satisfied.",
    "LIVENESS_FACE_NOT_TRACKED": "Face could not be tracked across enough frames.",
    "LIVENESS_INSUFFICIENT_FRAMES": "At least three frames are required for a liveness check.",
    "LIVENESS_BACKEND_UNAVAILABLE": "Liveness backend is not installed.",
    "LIVENESS_REPLAY_ATTACK_NOT_COVERED": "This check defeats printed photos but not video replay or screen replay.",
    "IDENTITY_NOT_CHECKED": "No selfie supplied, so the document was not bound to a live person.",
    "MARKSHEET_SELECTION_COUNT_INVALID":
    "Exactly five subjects must be selected.",

"SELECTED_SUBJECT_NOT_FOUND":
    "One of the selected subjects could not be reliably found in the marksheet.",

"MARKSHEET_SELECTION_OK":
    "All five selected subjects were found and included in the calculation.",
}

TIER_OF_REASON: dict[str, Tier] = {}
for _code in (
    "QR_SIGNATURE_VALID",
    "QR_SIGNATURE_INVALID",
):
    TIER_OF_REASON[_code] = Tier.CRYPTOGRAPHIC
for _code in (
    "AADHAAR_VERHOEFF_FAILED",
    "AADHAAR_LENGTH_INVALID",
    "AADHAAR_LEADING_DIGIT_INVALID",
    "PAN_PATTERN_INVALID",
    "PAN_LENGTH_INVALID",
    "PAN_HOLDER_TYPE_UNKNOWN",
    "MARKS_EXCEED_MAXIMUM",
    "TOTAL_MISMATCH",
    "PERCENTAGE_MISMATCH",
    "PERCENTAGE_IMPOSSIBLE",
    "MARKSHEET_SELECTION_COUNT_INVALID",
    "SELECTED_SUBJECT_NOT_FOUND",
):
    TIER_OF_REASON[_code] = Tier.STRUCTURAL
for _code in (
    "ELA_LOCALISED_ANOMALY",
    "BASELINE_IRREGULARITY_ADVISORY",
    "PAN_SURNAME_INITIAL_MISMATCH",
):
    TIER_OF_REASON[_code] = Tier.HEURISTIC

FATAL_CRYPTOGRAPHIC = {"QR_SIGNATURE_INVALID"}
PASSING_CRYPTOGRAPHIC = {"QR_SIGNATURE_VALID"}
FATAL_STRUCTURAL = {
    code for code, tier in TIER_OF_REASON.items() if tier is Tier.STRUCTURAL
}


@dataclass
class Assessment:
    verdict: Verdict
    identity_binding: Binding
    decided_by: Tier | None
    reasons: list[dict] = field(default_factory=list)
    advisory: list[dict] = field(default_factory=list)

    def headline(self) -> str:
        return {
            Verdict.GENUINE_SIGNED: "Issuer-signed and unaltered",
            Verdict.FORGED_SIGNATURE: "Signature check failed",
            Verdict.STRUCTURALLY_INVALID: "Structurally invalid",
            Verdict.UNVERIFIABLE: "Not independently verifiable",
            Verdict.NEEDS_REVIEW: "Needs human review",
        }[self.verdict]


def _describe(code: str) -> dict:
    return {
        "code": code,
        "tier": TIER_OF_REASON.get(code, Tier.HEURISTIC).value,
        "message": REASON_TEXT.get(code, code),
    }


def assess(
    reason_codes: list[str],
    face_matched: bool | None = None,
    liveness_passed: bool | None = None,
) -> Assessment:
    """
    Fold a flat list of reason codes into one verdict, strongest tier wins.
    """
    codes = [code for code in dict.fromkeys(reason_codes) if code]

    crypto_fail = [code for code in codes if code in FATAL_CRYPTOGRAPHIC]
    crypto_pass = [code for code in codes if code in PASSING_CRYPTOGRAPHIC]
    structural_fail = [code for code in codes if code in FATAL_STRUCTURAL]
    heuristic_flags = [
        code
        for code in codes
        if TIER_OF_REASON.get(code) is Tier.HEURISTIC
    ]

    if crypto_fail:
        verdict, decided_by, primary = Verdict.FORGED_SIGNATURE, Tier.CRYPTOGRAPHIC, crypto_fail
    elif crypto_pass:
        # Tier 1 passed. Heuristics are demoted to advisory even if they fired:
        # the signature covers these exact bytes, so ELA noise on a photo of a
        # signed document tells you about the camera, not the document.
        verdict, decided_by, primary = Verdict.GENUINE_SIGNED, Tier.CRYPTOGRAPHIC, crypto_pass
    elif structural_fail:
        verdict, decided_by, primary = Verdict.STRUCTURALLY_INVALID, Tier.STRUCTURAL, structural_fail
    elif heuristic_flags:
        verdict, decided_by, primary = Verdict.NEEDS_REVIEW, Tier.HEURISTIC, heuristic_flags
    else:
        verdict, decided_by, primary = Verdict.UNVERIFIABLE, None, []

    binding = _binding(face_matched, liveness_passed)

    advisory_codes = [code for code in codes if code not in primary]
    return Assessment(
        verdict=verdict,
        identity_binding=binding,
        decided_by=decided_by,
        reasons=[_describe(code) for code in primary],
        advisory=[_describe(code) for code in advisory_codes],
    )


def _binding(face_matched: bool | None, liveness_passed: bool | None) -> Binding:
    if face_matched is None:
        return Binding.NOT_ATTEMPTED
    if liveness_passed is False:
        # Order matters: a spoofed presentation that "matches" is not a bind.
        # Report the liveness failure rather than the match.
        return Binding.LIVENESS_FAILED
    return Binding.BOUND if face_matched else Binding.NOT_BOUND
