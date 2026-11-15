from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def normalize_struktur_org(value: str) -> str:
    raw = value.strip().removesuffix("%").rstrip(",")
    segments = raw.split(",")
    if not segments or any(not segment.isdigit() or int(segment) <= 0 for segment in segments):
        raise ValueError("struktur_org must contain positive organization IDs separated by commas")
    normalized = ",".join(segments) + ","
    if len(normalized) > 1000:
        raise ValueError("struktur_org exceeds 1000 characters")
    return normalized


class EnrollmentCreate(BaseModel):
    subject_id: str = Field(
        min_length=22,
        max_length=22,
        pattern=r"^nip-[0-9]{18}$",
        description="NIP baru 18 digit dengan awalan nip-",
    )
    struktur_org: str | None = None

    @field_validator("struktur_org")
    @classmethod
    def validate_struktur_org(cls, value: str | None) -> str | None:
        return normalize_struktur_org(value) if value is not None else None


class SessionCreateResponse(BaseModel):
    id: str
    subject_id: str
    status: str
    challenge: list[str]
    upload_token: str
    expires_at: datetime


class EnrollmentComplete(BaseModel):
    liveness_passed: bool
    liveness_method: str = Field(default="client_active_challenge", max_length=64)
    completed_challenges: list[str] = Field(min_length=2, max_length=6)
    challenge_duration_ms: int = Field(ge=500, le=120_000)


class CaptureMetadata(BaseModel):
    """Optional, bounded metadata supplied by the mobile capture client."""

    model_config = ConfigDict(extra="forbid")

    captured_at: datetime | None = None
    device_id: str | None = Field(default=None, max_length=128)
    platform: Literal["android", "ios"] | None = None
    app_version: str | None = Field(default=None, max_length=32)
    camera: Literal["front", "back"] | None = None
    sequence: int | None = Field(default=None, ge=1, le=8)
    challenge_action: str | None = Field(default=None, max_length=64)


class MobileEnrollmentCreate(BaseModel):
    struktur_org: str

    @field_validator("struktur_org")
    @classmethod
    def validate_struktur_org(cls, value: str) -> str:
        return normalize_struktur_org(value)


class MobileProfileResponse(BaseModel):
    status: str
    face_trained: bool
    latest_enrollment_id: str | None = None
    rejection_reason: str | None = None


class MobileVerificationCreate(BaseModel):
    pass


class ReviewRequest(BaseModel):
    actor: str = Field(min_length=2, max_length=128)
    reason: str | None = Field(default=None, max_length=255)


class EnrollmentResponse(BaseModel):
    id: str
    subject_id: str
    struktur_org: str | None
    status: str
    challenge: list[str]
    sample_count: int
    candidate_template_id: str | None
    liveness_passed: bool
    liveness_method: str | None
    rejection_reason: str | None
    created_at: datetime
    expires_at: datetime
    completed_at: datetime | None
    reviewed_at: datetime | None
    reviewed_by: str | None


class EnrollmentListResponse(BaseModel):
    items: list[EnrollmentResponse]
    total: int


class EnrollmentSampleReviewItem(BaseModel):
    sample_id: str
    sample_type: str
    content_type: str
    byte_size: int
    checksum_sha256: str
    capture_metadata: dict[str, object]
    quality: dict[str, float | int | str | bool]
    created_at: datetime
    image_url: str


class EnrollmentSamplesReviewResponse(BaseModel):
    enrollment_id: str
    subject_id: str
    status: str
    samples: list[EnrollmentSampleReviewItem]


class CaptureResponse(BaseModel):
    sample_id: str
    enrollment_id: str
    sample_type: str
    sample_count: int
    model_id: str
    quality: dict[str, float | int | str | bool]


class EnrollmentCompleteResponse(BaseModel):
    enrollment_id: str
    subject_id: str
    template_id: str
    status: str
    sample_count: int
    model_id: str
    liveness_assurance: str


class EnrollmentSubmissionResponse(BaseModel):
    enrollment_id: str
    subject_id: str
    status: str
    sample_count: int
    liveness_assurance: str


class ProfileResponse(BaseModel):
    subject_id: str
    status: str
    active_template_id: str | None
    created_at: datetime
    updated_at: datetime


class TemplateDeletionResponse(BaseModel):
    subject_id: str
    status: str
    deleted_template_count: int
    cancelled_enrollment_count: int
    message: str


class VerificationCreate(EnrollmentCreate):
    pass


class VerificationCaptureResponse(BaseModel):
    verification_id: str
    subject_id: str
    status: str
    face_detected: bool
    quality_passed: bool
    liveness_passed: bool
    match: bool
    score: float
    threshold: float
    model_id: str
    reason_code: str | None


class OfflinePackageData(BaseModel):
    payload: str
    signature: str
    key_id: str


class OfflinePackageResponse(BaseModel):
    status: Literal["Success"] = "Success"
    message: str | None = None
    data: OfflinePackageData


class OfflineVerificationResponse(BaseModel):
    verification_id: str
    match: bool
    score: float
    threshold: float
    model_id: str
    template_version: int
    reason_code: str | None


class MessageResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_id: str
    environment: str
