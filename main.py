"""
FastAPI application.

Endpoints
    POST /verify/aadhaar    QR decode + signature verify + Verhoeff + ELA
    POST /verify/pan        OCR + structural check + ELA
    POST /verify/marksheet  OCR + arithmetic/range checks + ELA + baselines
    POST /verify/face       ID photo vs live frames, in-memory only
    POST /verify/aadhaar-full  QR verify AND bind to a live selfie in one call
    GET  /dashboard/history
    GET  /dashboard/summary
    GET  /reasons           the full reason-code dictionary, for the UI
    GET  /health

Every response carries `disclaimer`. It is not decoration. Your own constraints
say no claim of official government verification, and the single most likely way
that constraint gets broken is a frontend that renders a green tick without
context. Putting the text in the payload means the frontend has to actively
discard it to mislead someone.
"""


from __future__ import annotations

import pymupdf

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .services import aadhaar_qr, ela, face_match, marksheet, ocr, pan, verhoeff
from .store import AuditStore
from .verdict import REASON_TEXT, assess

DISCLAIMER = (
    "This tool checks whether a document is internally consistent and, for "
    "Aadhaar, whether its QR payload carries a valid UIDAI signature. It does "
    "not query any government database and is not official verification."
)

app = FastAPI(title="Document & Identity Verification", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = AuditStore(settings.database_path)
UIDAI_KEY = aadhaar_qr.load_uidai_public_key(settings.uidai_cert_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def read_upload(upload: UploadFile) -> bytes:
    payload = await upload.read()
    if not payload:
        raise HTTPException(400, "Empty upload")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(413, "File exceeds the configured size limit")
    return payload


def consent_gate(consent_subject: str | None) -> str:
    """
    Enforces the teammate-consent rule for anything touching a live face.
    """
    if not settings.require_consent_subject:
        return consent_subject or "unspecified"
    if not consent_subject:
        raise HTTPException(
            400,
            "consent_subject is required. Name the teammate who agreed to have "
            "their face and ID used, and add them to CONSENT_SUBJECTS.",
        )
    if consent_subject.strip().lower() not in settings.consent_subjects:
        raise HTTPException(
            403,
            f"'{consent_subject}' is not in the consent list. Ask them directly, "
            "then add them to CONSENT_SUBJECTS.",
        )
    return consent_subject.strip().lower()


def envelope(assessment, extra: dict, record_id: str) -> dict:
    return {
        "record_id": record_id,
        "verdict": assessment.verdict.value,
        "headline": assessment.headline(),
        "decided_by": assessment.decided_by.value if assessment.decided_by else None,
        "identity_binding": assessment.identity_binding.value,
        "reasons": assessment.reasons,
        "advisory": assessment.advisory,
        "details": extra,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Aadhaar
# ---------------------------------------------------------------------------

@app.post("/verify/aadhaar")
async def verify_aadhaar(
    document: UploadFile = File(...),
    back_document: UploadFile | None = File(None),
):
    """
    Aadhaar layout handling:
      - New format: photo/front and Secure QR/back are uploaded separately.
      - Legacy format: photo + QR may be on the same side; one upload still works.

    `document` is the FRONT image. If `back_document` is supplied, QR scanning
    is performed on the back first and falls back to the front for legacy cards.
    """
    front_bytes = await read_upload(document)
    back_bytes = await read_upload(back_document) if back_document else None

    qr_source = back_bytes if back_bytes is not None else front_bytes
    qr_result = aadhaar_qr.verify_aadhaar_image(qr_source, UIDAI_KEY)

    # If a back image was supplied but no QR was found there, fall back to the
    # front. This keeps the endpoint tolerant of users uploading the wrong side
    # first while still supporting the new two-sided layout.
    if back_bytes is not None and qr_result.version == "UNKNOWN":
        front_qr_result = aadhaar_qr.verify_aadhaar_image(front_bytes, UIDAI_KEY)
        if front_qr_result.version != "UNKNOWN":
            qr_result = front_qr_result

    image_bytes = front_bytes
    codes = list(qr_result.reasons)

    # The reference_id in a signed payload holds the last four digits, which is
    # not enough to run Verhoeff. Fall back to OCR for the full number so the
    # checksum is still demonstrable, and note where the number came from.
    text_result = ocr.extract_text(image_bytes)
    number_source = None
    if text_result.aadhaar_numbers:
        number_source = "ocr"
        ok, reason = verhoeff.validate_aadhaar_number(text_result.aadhaar_numbers[0])
        codes.append(reason)

    ela_result = ela.analyse(image_bytes)
    codes.extend(ela_result.reasons)

    assessment = assess(codes)
    record_id = store.record(
        doc_type="aadhaar",
        verdict=assessment.verdict.value,
        decided_by=assessment.decided_by.value if assessment.decided_by else None,
        binding=assessment.identity_binding.value,
        reason_codes=[item["code"] for item in assessment.reasons],
        advisory_codes=[item["code"] for item in assessment.advisory],
        qr_version=qr_result.version,
    )

    return envelope(
        assessment,
        {
            "qr_version": qr_result.version,
            "signature_verified": qr_result.signature_verified,
            "layout": "two_sided_new" if back_bytes is not None else "legacy_or_single_image",
            "qr_side": "back" if back_bytes is not None else "uploaded_document",
            "photo_side": "front",
            "fields": qr_result.redacted() if settings.redact_identifiers else qr_result.fields,
            "id_photo_available": qr_result.photo_jp2 is not None,
            "aadhaar_number_source": number_source,
            "ocr_engine": text_result.engine,
            "ela": {
                "applicable": ela_result.applicable,
                "flagged_blocks": ela_result.flagged_blocks,
                "flagged_fraction": round(ela_result.flagged_fraction, 4),
                "heatmap_png_base64": ela_result.heatmap_png_base64,
            },
        },
        record_id,
    )


@app.post("/verify/aadhaar-full")
async def verify_aadhaar_full(
    document: UploadFile = File(...),
    frames: list[UploadFile] = File(...),
    consent_subject: str = Form(...),
    back_document: UploadFile | None = File(None),
):
    """
    The demo centrepiece: authenticity and identity binding in one call.

    Current two-sided Aadhaar handling:
      - FRONT: printed photograph.
      - BACK: QR code.
    The QR is still verified from the signed payload, while the live face is
    compared with the printed photograph on the FRONT when a back image is
    supplied. Legacy one-sided cards remain supported, and their signed QR
    photo is used when available.
    """
    consent_gate(consent_subject)

    # `document` is always the FRONT side.
    # `back_document` is optional and should contain the BACK side for the
    # current two-sided Aadhaar layout. Legacy cards can still be verified with
    # only `document`.
    front_bytes = await read_upload(document)
    back_bytes = await read_upload(back_document) if back_document else None
    frame_bytes = [await read_upload(frame) for frame in frames]

    qr_source = back_bytes if back_bytes is not None else front_bytes
    qr_result = aadhaar_qr.verify_aadhaar_image(qr_source, UIDAI_KEY)

    # Tolerate an incorrectly supplied side and preserve legacy support.
    if back_bytes is not None and qr_result.version == "UNKNOWN":
        front_qr_result = aadhaar_qr.verify_aadhaar_image(front_bytes, UIDAI_KEY)
        if front_qr_result.version != "UNKNOWN":
            qr_result = front_qr_result

    codes = list(qr_result.reasons)

    if back_bytes is not None:
        # New cards: the printed photograph is on the FRONT, while the QR is
        # on the BACK. Compare the live face with the FRONT card image.
        face_source = front_bytes
    elif qr_result.photo_jp2 is not None:
        # Legacy Secure QR: keep the existing signed QR photo path.
        face_source = qr_result.photo_jp2
    else:
        # Legacy/V1 cards do not carry a usable signed QR photo. The front
        # image may still contain the printed photograph, so try it directly.
        face_source = front_bytes

    liveness = face_match.check_liveness(list(frame_bytes), challenge="blink")
    codes.extend(liveness.reasons)

    match = face_match.compare_faces(
        face_source,
        frame_bytes[len(frame_bytes) // 2],
    )
    codes.extend(match.reasons)

    ela_result = ela.analyse(front_bytes)
    if len(frame_bytes) < 3:
        raise HTTPException(
        status_code=400,
        detail="At least 3 live frames are required for face verification.",
    )
    codes.extend(ela_result.reasons)

    assessment = assess(
        codes, face_matched=match.is_match, liveness_passed=liveness.passed
    )
    record_id = store.record(
        doc_type="aadhaar",
        verdict=assessment.verdict.value,
        decided_by=assessment.decided_by.value if assessment.decided_by else None,
        binding=assessment.identity_binding.value,
        reason_codes=[item["code"] for item in assessment.reasons],
        advisory_codes=[item["code"] for item in assessment.advisory],
        qr_version=qr_result.version,
        face_distance=match.distance,
        liveness=liveness.challenge if liveness.checked else None,
    )

    # Frames and the extracted photo go out of scope here and are never
    # written anywhere. Nothing biometric reaches the response or the store.
    return envelope(
        assessment,
        {
            "qr_version": qr_result.version,
            "signature_verified": qr_result.signature_verified,
            "layout": "two_sided_new" if back_bytes is not None else "legacy_or_single_image",
            "qr_side": "back" if back_bytes is not None else "uploaded_document",
            "photo_side": "front",
            "face_source": "printed_front" if back_bytes is not None else (
                "signed_qr_photo" if qr_result.photo_jp2 is not None else "front_image"
            ),
            "fields": qr_result.redacted() if settings.redact_identifiers else qr_result.fields,
            "face": {
                "compared": match.compared,
                "similarity": match.similarity,
                "threshold": match.threshold,
                "model": match.model,
                "is_match": match.is_match,
            },
            "liveness": {
                "checked": liveness.checked,
                "passed": liveness.passed,
                "challenge": liveness.challenge,
                "detail": liveness.detail,
            },
            "ela": {
                "applicable": ela_result.applicable,
                "flagged_fraction": round(ela_result.flagged_fraction, 4),
            },
        },
        record_id,
    )


# ---------------------------------------------------------------------------
# PAN
# ---------------------------------------------------------------------------

@app.post("/verify/pan")
async def verify_pan(
    document: UploadFile = File(...),
    pan_number: str | None = Form(None),
):
    image_bytes = await read_upload(document)
    text_result = ocr.extract_text(image_bytes)

    candidate = pan_number or (text_result.pan_numbers[0] if text_result.pan_numbers else "")
    pan_report = pan.validate_pan(candidate)
    codes = list(pan_report["reasons"])

    name_check = pan.cross_check_name(candidate, text_result.text)
    codes.extend(name_check["reasons"])

    ela_result = ela.analyse(image_bytes)
    codes.extend(ela_result.reasons)

    assessment = assess(codes)
    record_id = store.record(
        doc_type="pan",
        verdict=assessment.verdict.value,
        decided_by=assessment.decided_by.value if assessment.decided_by else None,
        binding=assessment.identity_binding.value,
        reason_codes=[item["code"] for item in assessment.reasons],
        advisory_codes=[item["code"] for item in assessment.advisory],
    )

    return envelope(
        assessment,
        {
            "pan": pan_report,
            "name_cross_check": name_check,
            "ocr_engine": text_result.engine,
            "ela": {
                "applicable": ela_result.applicable,
                "flagged_blocks": ela_result.flagged_blocks,
                "heatmap_png_base64": ela_result.heatmap_png_base64,
            },
            "note": (
                "PAN has no offline cryptographic material and no public check "
                "digit. A clean result here means the structure is plausible, "
                "nothing more."
            ),
        },
        record_id,
    )


# ---------------------------------------------------------------------------
# Marksheet
# ---------------------------------------------------------------------------

@app.post("/verify/marksheet")
async def verify_marksheet(
    document: UploadFile = File(...),
    subject1: str = Form(...),
    subject2: str = Form(...),
    subject3: str = Form(...),
    subject4: str = Form(...),
    subject5: str = Form(...),
):
    image_bytes = await read_upload(document)

    # Convert PDF first page to JPEG for OCR/forensics.
    if document.content_type == "application/pdf":
        pdf = pymupdf.open(
            stream=image_bytes,
            filetype="pdf",
        )

        if len(pdf) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty PDF",
            )

        page = pdf[0]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False,
        )

        image_bytes = pix.tobytes("jpeg")

        pdf.close()

    text_result = ocr.extract_text(
        image_bytes
    )

    selected_subjects = [
        subject1,
        subject2,
        subject3,
        subject4,
        subject5,
    ]

    sheet = marksheet.analyse(
        text_result.text,
        image_bytes,
        selected_subjects=selected_subjects,
    )

    codes = list(sheet.reasons)

    ela_result = ela.analyse(
        image_bytes
    )

    codes.extend(
        ela_result.reasons
    )

    assessment = assess(codes)

    record_id = store.record(
        doc_type="marksheet",
        verdict=assessment.verdict.value,
        decided_by=(
            assessment.decided_by.value
            if assessment.decided_by
            else None
        ),
        binding=assessment.identity_binding.value,
        reason_codes=[
            item["code"]
            for item in assessment.reasons
        ],
        advisory_codes=[
            item["code"]
            for item in assessment.advisory
        ],
    )

    return envelope(
        assessment,
        {
            "selected_subjects": sheet.selected_subjects,

            "subjects": sheet.subjects,

            "selected_total": sheet.selected_total,

            "selected_maximum": sheet.selected_maximum,

            "selected_percentage": sheet.selected_percentage,

            "computed_total": sheet.computed_total,

            "printed_total": sheet.printed_total,

            "printed_percentage": sheet.printed_percentage,

            "baseline_outlier_rows": (
                sheet.baseline_outliers
            ),

            "ocr_engine": text_result.engine,

            "ela": {
                "applicable": ela_result.applicable,
                "flagged_blocks": (
                    ela_result.flagged_blocks
                ),
                "heatmap_png_base64": (
                    ela_result.heatmap_png_base64
                ),
            },

            "calculation_note": (
                "Only the five subjects selected by the "
                "user are included in the total and percentage. "
                "Additional subjects are excluded."
            ),
        },
        record_id,
    )


# ---------------------------------------------------------------------------
# Standalone face match
# ---------------------------------------------------------------------------

@app.post("/verify/face")
async def verify_face(
    id_photo: UploadFile = File(...),
    frames: list[UploadFile] = File(...),
    consent_subject: str = Form(...),
    challenge: str = Form("blink"),
):
    consent_gate(consent_subject)

    id_bytes = await read_upload(id_photo)
    frame_bytes = [await read_upload(frame) for frame in frames]

    if len(frame_bytes) < 3:
        raise HTTPException(
        status_code=400,
        detail="At least 3 live frames are required for face verification.",
    )

    liveness = face_match.check_liveness(list(frame_bytes), challenge=challenge)
    match = face_match.compare_faces(id_bytes, frame_bytes[len(frame_bytes) // 2])

    codes = liveness.reasons + match.reasons
    assessment = assess(
        codes, face_matched=match.is_match, liveness_passed=liveness.passed
    )
    record_id = store.record(
        doc_type="face",
        verdict=assessment.verdict.value,
        decided_by=assessment.decided_by.value if assessment.decided_by else None,
        binding=assessment.identity_binding.value,
        reason_codes=[item["code"] for item in assessment.reasons],
        advisory_codes=[item["code"] for item in assessment.advisory],
        face_distance=match.distance,
        liveness=liveness.challenge if liveness.checked else None,
    )

    return envelope(
        assessment,
        {
            "face": {
                "compared": match.compared,
                "similarity": match.similarity,
                "threshold": match.threshold,
                "model": match.model,
                "is_match": match.is_match,
            },
            "liveness": {
                "checked": liveness.checked,
                "passed": liveness.passed,
                "challenge": liveness.challenge,
                "detail": liveness.detail,
            },
        },
        record_id,
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.get("/dashboard/history")
def dashboard_history(limit: int = 50):
    return {"records": store.history(limit), "disclaimer": DISCLAIMER}


@app.get("/dashboard/summary")
def dashboard_summary():
    return {**store.summary(), "disclaimer": DISCLAIMER}


@app.get("/reasons")
def reasons():
    return {"reasons": REASON_TEXT}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uidai_certificate_loaded": UIDAI_KEY is not None,
        "consent_enforcement": settings.require_consent_subject,
        "consent_subjects_configured": len(settings.consent_subjects),
    }
