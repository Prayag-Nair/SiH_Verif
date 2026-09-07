"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    uidai_cert_path = "certs/uidai_signing.cer"
    database_path: str = os.getenv("AUDIT_DB_PATH", "audit.sqlite3")
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))

    # Your plan says demo data comes only from consenting teammates. That works
    # as a rule on a whiteboard and fails the moment someone tests with a
    # cousin's card at 2am. The face-match endpoint requires an explicit
    # consent subject and refuses without one, so the rule lives in the code.
    require_consent_subject: bool = (
        os.getenv("REQUIRE_CONSENT_SUBJECT", "true").lower() == "true"
    )

    # Comma-separated list of teammates who have given consent, e.g.
    # CONSENT_SUBJECTS="vishu,priya,arjun"
    consent_subjects: tuple[str, ...] = tuple(
        item.strip().lower()
        for item in os.getenv("CONSENT_SUBJECTS", "").split(",")
        if item.strip()
    )

    # On-screen redaction of ID numbers, per your constraints. Server-side, so
    # a frontend bug cannot leak a real number into a screenshot.
    redact_identifiers: bool = (
        os.getenv("REDACT_IDENTIFIERS", "true").lower() == "true"
    )


settings = Settings()
