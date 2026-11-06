from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class BiometricProfile(Base):
    __tablename__ = "biometric_profiles"

    subject_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="not_registered", index=True)
    active_template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    templates: Mapped[list[BiometricTemplate]] = relationship(back_populates="profile")


class BiometricTemplate(Base):
    __tablename__ = "biometric_templates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("tpl"))
    subject_id: Mapped[str] = mapped_column(ForeignKey("biometric_profiles.subject_id"), index=True)
    enrollment_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    encrypted_embedding: Mapped[bytes] = mapped_column(LargeBinary)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[str] = mapped_column(String(160))
    sample_count: Mapped[int] = mapped_column(Integer)
    quality_summary: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    profile: Mapped[BiometricProfile] = relationship(back_populates="templates")


class EnrollmentSession(Base):
    __tablename__ = "enrollment_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("enr"))
    subject_id: Mapped[str] = mapped_column(ForeignKey("biometric_profiles.subject_id"), index=True)
    struktur_org: Mapped[str | None] = mapped_column(String(1000), nullable=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="collecting", index=True)
    challenge: Mapped[str] = mapped_column(Text)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    candidate_template_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    client_liveness_passed: Mapped[bool] = mapped_column(default=False)
    liveness_method: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)

    samples: Mapped[list[EnrollmentSample]] = relationship(
        back_populates="enrollment", cascade="all, delete-orphan"
    )


class EnrollmentSample(Base):
    __tablename__ = "enrollment_samples"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("smp"))
    enrollment_id: Mapped[str] = mapped_column(ForeignKey("enrollment_sessions.id"), index=True)
    sample_type: Mapped[str] = mapped_column(String(32))
    encrypted_embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    model_id: Mapped[str] = mapped_column(String(160))
    quality: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    enrollment: Mapped[EnrollmentSession] = relationship(back_populates="samples")


class EnrollmentSampleImage(Base):
    """Encrypted, temporary enrollment photo retained only for manual review."""

    __tablename__ = "enrollment_sample_images"

    sample_id: Mapped[str] = mapped_column(
        ForeignKey("enrollment_samples.id", ondelete="CASCADE"), primary_key=True
    )
    enrollment_id: Mapped[str] = mapped_column(
        ForeignKey("enrollment_sessions.id", ondelete="CASCADE"), index=True
    )
    encrypted_image: Mapped[bytes] = mapped_column(LargeBinary)
    content_type: Mapped[str] = mapped_column(String(32))
    byte_size: Mapped[int] = mapped_column(Integer)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    capture_metadata: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ver"))
    subject_id: Mapped[str] = mapped_column(ForeignKey("biometric_profiles.subject_id"), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="created", index=True)
    challenge: Mapped[str] = mapped_column(Text)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    matched: Mapped[bool | None] = mapped_column(nullable=True)
    liveness_passed: Mapped[bool | None] = mapped_column(nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditLog(Base):
    __tablename__ = "biometric_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(80), index=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


Index("ix_templates_subject_status", BiometricTemplate.subject_id, BiometricTemplate.status)
Index("ix_enrollments_subject_status", EnrollmentSession.subject_id, EnrollmentSession.status)
