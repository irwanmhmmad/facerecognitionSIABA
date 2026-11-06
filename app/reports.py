from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from datetime import datetime
from io import StringIO
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import EnrollmentSession, VerificationSession


def _safe_cell(value: Any) -> Any:
    """Prevent spreadsheet applications from evaluating exported text as formulas."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return f"'{value}"
    return value


def csv_stream(headers: list[str], rows: Iterable[list[Any]]) -> Iterator[str]:
    buffer = StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(headers)
    yield "\ufeff" + buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    for row in rows:
        writer.writerow([_safe_cell(value) for value in row])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)


def _apply_filters(
    query: Select,
    model: type[EnrollmentSession] | type[VerificationSession],
    *,
    status: str | None,
    subject_id: str | None,
    created_from: datetime | None,
    created_to: datetime | None,
) -> Select:
    if status:
        query = query.where(model.status == status)
    if subject_id:
        query = query.where(model.subject_id == subject_id)
    if created_from:
        query = query.where(model.created_at >= created_from)
    if created_to:
        query = query.where(model.created_at <= created_to)
    return query


def enrollment_report_rows(
    db: Session,
    *,
    status: str | None,
    subject_id: str | None,
    created_from: datetime | None,
    created_to: datetime | None,
) -> Iterator[list[Any]]:
    query = _apply_filters(
        select(EnrollmentSession),
        EnrollmentSession,
        status=status,
        subject_id=subject_id,
        created_from=created_from,
        created_to=created_to,
    ).order_by(EnrollmentSession.created_at.desc())
    for item in db.scalars(query).yield_per(500):
        yield [
            item.id,
            item.subject_id,
            item.status,
            item.sample_count,
            item.client_liveness_passed,
            item.liveness_method,
            item.candidate_template_id,
            item.rejection_reason,
            item.created_at,
            item.expires_at,
            item.completed_at,
            item.reviewed_at,
            item.reviewed_by,
        ]


def verification_report_rows(
    db: Session,
    *,
    status: str | None,
    subject_id: str | None,
    created_from: datetime | None,
    created_to: datetime | None,
) -> Iterator[list[Any]]:
    query = _apply_filters(
        select(VerificationSession),
        VerificationSession,
        status=status,
        subject_id=subject_id,
        created_from=created_from,
        created_to=created_to,
    ).order_by(VerificationSession.created_at.desc())
    for item in db.scalars(query).yield_per(500):
        yield [
            item.id,
            item.subject_id,
            item.status,
            item.score,
            item.threshold,
            item.matched,
            item.liveness_passed,
            item.failure_reason,
            item.created_at,
            item.expires_at,
            item.completed_at,
        ]
