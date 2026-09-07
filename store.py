"""
Audit store for the dashboard.

Schema design is a privacy decision, not a data-modelling one. This table
physically cannot hold the things your constraints forbid, because there is
nowhere to put them: no image column, no embedding column, no full identifier
column. A column you never populate eventually gets populated by someone at
3am; a column that does not exist does not.

What is stored: document type, verdict, identity binding, reason codes, the
face distance rounded to two places, timestamps. That is enough to draw
flagged-vs-verified trends and to explain any individual flag.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS checks (
    id             TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    doc_type       TEXT NOT NULL,
    qr_version     TEXT,
    verdict        TEXT NOT NULL,
    decided_by     TEXT,
    binding        TEXT NOT NULL,
    reason_codes   TEXT NOT NULL,
    advisory_codes TEXT NOT NULL,
    face_distance  REAL,
    liveness       TEXT
);
CREATE INDEX IF NOT EXISTS idx_checks_created ON checks(created_at);
CREATE INDEX IF NOT EXISTS idx_checks_verdict ON checks(verdict);
"""


class AuditStore:
    def __init__(self, path: str):
        self._path = path
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def record(
        self,
        doc_type: str,
        verdict: str,
        decided_by: str | None,
        binding: str,
        reason_codes: list[str],
        advisory_codes: list[str],
        qr_version: str | None = None,
        face_distance: float | None = None,
        liveness: str | None = None,
    ) -> str:
        record_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO checks (
                    id, created_at, doc_type, qr_version, verdict, decided_by,
                    binding, reason_codes, advisory_codes, face_distance, liveness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    doc_type,
                    qr_version,
                    verdict,
                    decided_by,
                    binding,
                    json.dumps(reason_codes),
                    json.dumps(advisory_codes),
                    round(face_distance, 2) if face_distance is not None else None,
                    liveness,
                ),
            )
        return record_id

    def history(self, limit: int = 50) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM checks ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def summary(self) -> dict:
        with self._connect() as connection:
            by_verdict = connection.execute(
                "SELECT verdict, COUNT(*) AS n FROM checks GROUP BY verdict"
            ).fetchall()
            by_doc = connection.execute(
                "SELECT doc_type, COUNT(*) AS n FROM checks GROUP BY doc_type"
            ).fetchall()
            by_day = connection.execute(
                """
                SELECT substr(created_at, 1, 10) AS day, verdict, COUNT(*) AS n
                FROM checks GROUP BY day, verdict ORDER BY day
                """
            ).fetchall()
            top_reasons = connection.execute(
                "SELECT reason_codes FROM checks"
            ).fetchall()

        counter: dict[str, int] = {}
        for row in top_reasons:
            for code in json.loads(row["reason_codes"]):
                counter[code] = counter.get(code, 0) + 1

        return {
            "by_verdict": {row["verdict"]: row["n"] for row in by_verdict},
            "by_doc_type": {row["doc_type"]: row["n"] for row in by_doc},
            "trend": [
                {"day": row["day"], "verdict": row["verdict"], "count": row["n"]}
                for row in by_day
            ],
            "top_reasons": sorted(
                ({"code": code, "count": count} for code, count in counter.items()),
                key=lambda item: item["count"],
                reverse=True,
            )[:10],
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["reason_codes"] = json.loads(record["reason_codes"])
        record["advisory_codes"] = json.loads(record["advisory_codes"])
        return record
