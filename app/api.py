from __future__ import annotations

import json
import re
from collections.abc import Generator
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.container import Container
from app.models import (
    BiometricProfile,
    BiometricTemplate,
    EnrollmentSample,
    EnrollmentSampleImage,
    EnrollmentSession,
)
from app.offline_services import issue_offline_package, verify_offline_attendance
from app.reports import csv_stream, enrollment_report_rows, verification_report_rows
from app.schemas import (
    CaptureMetadata,
    CaptureResponse,
    EnrollmentComplete,
    EnrollmentCompleteResponse,
    EnrollmentCreate,
    EnrollmentListResponse,
    EnrollmentResponse,
    EnrollmentSampleReviewItem,
    EnrollmentSamplesReviewResponse,
    EnrollmentSubmissionResponse,
    HealthResponse,
    MessageResponse,
    MobileEnrollmentCreate,
    MobileProfileResponse,
    MobileVerificationCreate,
    OfflinePackageResponse,
    OfflineVerificationResponse,
    ProfileResponse,
    ReviewRequest,
    SessionCreateResponse,
    TemplateDeletionResponse,
    VerificationCaptureResponse,
    VerificationCreate,
    normalize_struktur_org,
)
from app.security import require_internal_service, require_mobile_subject, require_upload_scope
from app.services import (
    UPLOADED_ENROLLMENT_STATUSES,
    activate_enrollment,
    add_enrollment_capture,
    approve_ktu_enrollment,
    complete_enrollment,
    complete_verification,
    count_enrollments,
    count_uploaded_enrollments,
    create_enrollment,
    create_verification,
    delete_templates_for_reenrollment,
    review_enrollment,
    revoke_profile,
)

router = APIRouter()
internal_router = APIRouter(prefix="/internal/v1", dependencies=[Depends(require_internal_service)])
upload_router = APIRouter(prefix="/v1")
mobile_router = APIRouter(prefix="/mobile/v1")

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_db(request: Request) -> Generator[Session, None, None]:
    yield from request.app.state.container.database.session()


def struktur_org_prefix(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return normalize_struktur_org(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def parse_capture_metadata(value: str | None) -> dict[str, object]:
    if value is None or not value.strip():
        return {}
    if len(value.encode("utf-8")) > 4096:
        raise HTTPException(status_code=413, detail="capture_metadata exceeds 4096 bytes")
    try:
        metadata = CaptureMetadata.model_validate_json(value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "capture_metadata must be valid JSON with supported fields",
                "errors": exc.errors(include_url=False),
            },
        ) from exc
    return metadata.model_dump(mode="json", exclude_none=True)


def reviewable_enrollment(db: Session, enrollment_id: str) -> EnrollmentSession:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status not in {"pending_ktu_review", "pending_bkd_review"}:
        raise HTTPException(status_code=409, detail="Enrollment is not awaiting review")
    return enrollment


def enrollment_samples_response(
    db: Session, enrollment: EnrollmentSession
) -> EnrollmentSamplesReviewResponse:
    rows = db.execute(
        select(EnrollmentSample, EnrollmentSampleImage)
        .join(EnrollmentSampleImage, EnrollmentSampleImage.sample_id == EnrollmentSample.id)
        .where(EnrollmentSample.enrollment_id == enrollment.id)
        .order_by(EnrollmentSample.created_at)
    ).all()
    return EnrollmentSamplesReviewResponse(
        enrollment_id=enrollment.id,
        subject_id=enrollment.subject_id,
        status=enrollment.status,
        samples=[
            EnrollmentSampleReviewItem(
                sample_id=sample.id,
                sample_type=sample.sample_type,
                content_type=stored_image.content_type,
                byte_size=stored_image.byte_size,
                checksum_sha256=stored_image.checksum_sha256,
                capture_metadata=json.loads(stored_image.capture_metadata),
                quality={
                    key: value for key, value in json.loads(sample.quality).items() if not key.startswith("_")
                },
                created_at=stored_image.created_at,
                image_url=(f"/internal/v1/enrollments/{enrollment.id}/samples/{sample.id}/image"),
            )
            for sample, stored_image in rows
        ],
    )


async def read_image(upload: UploadFile, container: Container) -> bytes:
    if upload.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG, and WebP images are accepted")
    content = await upload.read(container.settings.max_upload_bytes + 1)
    if not content:
        raise HTTPException(status_code=422, detail="Image is empty")
    if len(content) > container.settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit")
    return content


def enrollment_response(item: EnrollmentSession) -> EnrollmentResponse:
    return EnrollmentResponse(
        id=item.id,
        subject_id=item.subject_id,
        struktur_org=item.struktur_org,
        status=item.status,
        challenge=json.loads(item.challenge),
        sample_count=item.sample_count,
        candidate_template_id=item.candidate_template_id,
        liveness_passed=item.client_liveness_passed,
        liveness_method=item.liveness_method,
        rejection_reason=item.rejection_reason,
        created_at=item.created_at,
        expires_at=item.expires_at,
        completed_at=item.completed_at,
        reviewed_at=item.reviewed_at,
        reviewed_by=item.reviewed_by,
    )


def profile_response(item: BiometricProfile) -> ProfileResponse:
    return ProfileResponse(
        subject_id=item.subject_id,
        status=item.status,
        active_template_id=item.active_template_id,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@mobile_router.get("/profile", response_model=MobileProfileResponse, tags=["mobile"])
async def mobile_profile(
    request: Request,
    subject_id: str = Depends(require_mobile_subject),
    db: Session = Depends(get_db),
) -> MobileProfileResponse:
    profile = db.get(BiometricProfile, subject_id)
    active_template = (
        db.get(BiometricTemplate, profile.active_template_id)
        if profile and profile.active_template_id
        else None
    )
    latest = db.scalar(
        select(EnrollmentSession)
        .where(EnrollmentSession.subject_id == subject_id)
        .order_by(EnrollmentSession.created_at.desc())
        .limit(1)
    )
    return MobileProfileResponse(
        status=profile.status if profile else "not_registered",
        face_trained=bool(
            profile
            and profile.status == "active"
            and active_template
            and active_template.status == "active"
            and active_template.model_id == get_container(request).engine.model_id
        ),
        latest_enrollment_id=latest.id if latest else None,
        rejection_reason=latest.rejection_reason if latest else None,
    )


@mobile_router.get(
    "/offline-package",
    response_model=OfflinePackageResponse,
    tags=["mobile", "offline-biometric"],
)
async def mobile_offline_package(
    request: Request,
    subject_id: str = Depends(require_mobile_subject),
    db: Session = Depends(get_db),
    device_id: str = Header(..., alias="X-Device-Id"),
    device_public_key: str = Header(..., alias="X-Device-Public-Key"),
) -> OfflinePackageResponse:
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", device_id):
        raise HTTPException(status_code=422, detail="X-Device-Id is invalid")
    container = get_container(request)
    if not container.offline_package_limiter.allow(subject_id):
        raise HTTPException(status_code=429, detail="Offline package rate limit exceeded")
    return OfflinePackageResponse.model_validate(
        issue_offline_package(
            db,
            container,
            subject_id=subject_id,
            device_id=device_id,
            device_public_key_header=device_public_key,
        )
    )


@mobile_router.post("/enrollments", response_model=SessionCreateResponse, status_code=201, tags=["mobile"])
async def mobile_enrollment_start(
    payload: MobileEnrollmentCreate,
    request: Request,
    subject_id: str = Depends(require_mobile_subject),
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SessionCreateResponse:
    enrollment, token = create_enrollment(
        db,
        get_container(request),
        subject_id=subject_id,
        idempotency_key=idempotency_key,
        struktur_org=payload.struktur_org,
    )
    return SessionCreateResponse(
        id=enrollment.id,
        subject_id=enrollment.subject_id,
        status=enrollment.status,
        challenge=json.loads(enrollment.challenge),
        upload_token=token,
        expires_at=enrollment.expires_at,
    )


@mobile_router.post("/verifications", response_model=SessionCreateResponse, status_code=201, tags=["mobile"])
async def mobile_verification_start(
    _payload: MobileVerificationCreate,
    request: Request,
    subject_id: str = Depends(require_mobile_subject),
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SessionCreateResponse:
    verification, token = create_verification(
        db,
        get_container(request),
        subject_id=subject_id,
        idempotency_key=idempotency_key,
    )
    return SessionCreateResponse(
        id=verification.id,
        subject_id=verification.subject_id,
        status=verification.status,
        challenge=json.loads(verification.challenge),
        upload_token=token,
        expires_at=verification.expires_at,
    )


@router.get("/health/live", response_model=HealthResponse, tags=["health"])
def live(request: Request) -> HealthResponse:
    container = get_container(request)
    return HealthResponse(
        status="ok",
        engine=container.settings.face_engine,
        model_id=container.engine.model_id,
        environment=container.settings.app_env,
    )


@router.get("/health/ready", response_model=HealthResponse, tags=["health"])
def ready(request: Request) -> HealthResponse:
    container = get_container(request)
    try:
        with container.database.engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database is not ready") from exc
    return HealthResponse(
        status="ready",
        engine=container.settings.face_engine,
        model_id=container.engine.model_id,
        environment=container.settings.app_env,
    )


@internal_router.post(
    "/enrollments", response_model=SessionCreateResponse, status_code=201, tags=["enrollment"]
)
def enrollment_start(
    payload: EnrollmentCreate,
    request: Request,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SessionCreateResponse:
    enrollment, token = create_enrollment(
        db,
        get_container(request),
        subject_id=payload.subject_id,
        idempotency_key=idempotency_key,
        struktur_org=payload.struktur_org,
    )
    return SessionCreateResponse(
        id=enrollment.id,
        subject_id=enrollment.subject_id,
        status=enrollment.status,
        challenge=json.loads(enrollment.challenge),
        upload_token=token,
        expires_at=enrollment.expires_at,
    )


@internal_router.get("/enrollments", response_model=EnrollmentListResponse, tags=["enrollment"])
def enrollment_list(
    db: Session = Depends(get_db),
    status_filter: str | None = Query(default=None, alias="status", max_length=32),
    struktur_org_filter: str | None = Query(default=None, alias="struktur_org", max_length=1001),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> EnrollmentListResponse:
    organization_prefix = struktur_org_prefix(struktur_org_filter)
    query = select(EnrollmentSession).order_by(EnrollmentSession.created_at.desc())
    if status_filter:
        query = query.where(EnrollmentSession.status == status_filter)
    if organization_prefix:
        query = query.where(EnrollmentSession.struktur_org.like(f"{organization_prefix}%"))
    items = list(db.scalars(query.offset(offset).limit(limit)))
    return EnrollmentListResponse(
        items=[enrollment_response(item) for item in items],
        total=count_enrollments(db, status_filter, organization_prefix),
    )


@internal_router.get("/enrollments/uploaded", response_model=EnrollmentListResponse, tags=["enrollment"])
def enrollment_uploaded_list(
    db: Session = Depends(get_db),
    status_filter: str | None = Query(default=None, alias="status", max_length=32),
    struktur_org_filter: str | None = Query(default=None, alias="struktur_org", max_length=1001),
    nip_filter: str | None = Query(default=None, alias="nip", pattern=r"^[0-9]{18}$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> EnrollmentListResponse:
    if status_filter and status_filter not in UPLOADED_ENROLLMENT_STATUSES:
        raise HTTPException(status_code=422, detail="Unsupported uploaded enrollment status")
    organization_prefix = struktur_org_prefix(struktur_org_filter)
    query = select(EnrollmentSession).order_by(EnrollmentSession.completed_at.desc())
    if status_filter:
        query = query.where(EnrollmentSession.status == status_filter)
    else:
        query = query.where(EnrollmentSession.status.in_(UPLOADED_ENROLLMENT_STATUSES))
    if organization_prefix:
        query = query.where(EnrollmentSession.struktur_org.like(f"{organization_prefix}%"))
    subject_id = f"nip-{nip_filter}" if nip_filter else None
    if subject_id:
        query = query.where(EnrollmentSession.subject_id == subject_id)
    items = list(db.scalars(query.offset(offset).limit(limit)))
    return EnrollmentListResponse(
        items=[enrollment_response(item) for item in items],
        total=count_uploaded_enrollments(db, status_filter, organization_prefix, subject_id),
    )


@internal_router.get(
    "/enrollments/by-nip/{nip}/samples",
    response_model=EnrollmentSamplesReviewResponse,
    tags=["enrollment"],
)
def enrollment_samples_for_review_by_nip(
    nip: str,
    db: Session = Depends(get_db),
    struktur_org_filter: str | None = Query(default=None, alias="struktur_org", max_length=1001),
) -> EnrollmentSamplesReviewResponse:
    if len(nip) != 18 or not nip.isdigit():
        raise HTTPException(status_code=422, detail="nip must contain exactly 18 digits")
    organization_prefix = struktur_org_prefix(struktur_org_filter)
    query = (
        select(EnrollmentSession)
        .where(
            EnrollmentSession.subject_id == f"nip-{nip}",
            EnrollmentSession.status.in_({"pending_ktu_review", "pending_bkd_review"}),
        )
        .order_by(EnrollmentSession.completed_at.desc())
        .limit(1)
    )
    if organization_prefix:
        query = query.where(EnrollmentSession.struktur_org.like(f"{organization_prefix}%"))
    enrollment = db.scalar(query)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Reviewable enrollment for NIP not found")
    return enrollment_samples_response(db, enrollment)


@internal_router.get(
    "/enrollments/{enrollment_id}/samples",
    response_model=EnrollmentSamplesReviewResponse,
    tags=["enrollment"],
)
def enrollment_samples_for_review(
    enrollment_id: str, db: Session = Depends(get_db)
) -> EnrollmentSamplesReviewResponse:
    enrollment = reviewable_enrollment(db, enrollment_id)
    return enrollment_samples_response(db, enrollment)


@internal_router.get(
    "/enrollments/{enrollment_id}/samples/{sample_id}/image",
    response_class=Response,
    tags=["enrollment"],
)
def enrollment_sample_image(
    enrollment_id: str,
    sample_id: str,
    request: Request,
    db: Session = Depends(get_db),
) -> Response:
    reviewable_enrollment(db, enrollment_id)
    stored_image = db.get(EnrollmentSampleImage, sample_id)
    if not stored_image or stored_image.enrollment_id != enrollment_id:
        raise HTTPException(status_code=404, detail="Enrollment sample image not found")
    try:
        content = get_container(request).cipher.decrypt_bytes(stored_image.encrypted_image)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Enrollment sample image cannot be read") from exc
    extension = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}.get(
        stored_image.content_type, "bin"
    )
    return Response(
        content=content,
        media_type=stored_image.content_type,
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": f'inline; filename="{sample_id}.{extension}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@internal_router.get("/enrollments/{enrollment_id}", response_model=EnrollmentResponse, tags=["enrollment"])
def enrollment_detail(enrollment_id: str, db: Session = Depends(get_db)) -> EnrollmentResponse:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    return enrollment_response(enrollment)


@upload_router.post(
    "/enrollments/{enrollment_id}/captures",
    response_model=CaptureResponse,
    status_code=201,
    tags=["enrollment"],
)
async def enrollment_capture(
    enrollment_id: str,
    request: Request,
    sample_type: str = Form(...),
    capture_metadata: str | None = Form(default=None),
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> CaptureResponse:
    claims = require_upload_scope(request, authorization, purpose="enrollment", session_id=enrollment_id)
    container = get_container(request)
    content = await read_image(image, container)
    sample = add_enrollment_capture(
        db,
        container,
        enrollment_id=enrollment_id,
        subject_id=claims["sub"],
        sample_type=sample_type,
        image_bytes=content,
        image_content_type=image.content_type or "application/octet-stream",
        capture_metadata=parse_capture_metadata(capture_metadata),
    )
    enrollment = db.get(EnrollmentSession, enrollment_id)
    quality = {key: value for key, value in json.loads(sample.quality).items() if not key.startswith("_")}
    return CaptureResponse(
        sample_id=sample.id,
        enrollment_id=enrollment_id,
        sample_type=sample.sample_type,
        sample_count=enrollment.sample_count,
        model_id=sample.model_id,
        quality=quality,
    )


@upload_router.post(
    "/enrollments/{enrollment_id}/complete",
    response_model=EnrollmentSubmissionResponse,
    tags=["enrollment"],
)
def enrollment_finish(
    enrollment_id: str,
    payload: EnrollmentComplete,
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> EnrollmentSubmissionResponse:
    claims = require_upload_scope(request, authorization, purpose="enrollment", session_id=enrollment_id)
    enrollment = complete_enrollment(
        db,
        get_container(request),
        enrollment_id=enrollment_id,
        subject_id=claims["sub"],
        liveness_passed=payload.liveness_passed,
        liveness_method=payload.liveness_method,
        completed_challenges=payload.completed_challenges,
        challenge_duration_ms=payload.challenge_duration_ms,
    )
    return EnrollmentSubmissionResponse(
        enrollment_id=enrollment.id,
        subject_id=enrollment.subject_id,
        status=enrollment.status,
        sample_count=enrollment.sample_count,
        liveness_assurance="client_active_plus_passive_sequence_v1",
    )


@internal_router.post(
    "/enrollments/{enrollment_id}/ktu-approve",
    response_model=EnrollmentResponse,
    tags=["enrollment"],
)
def enrollment_ktu_approve(
    enrollment_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> EnrollmentResponse:
    enrollment = approve_ktu_enrollment(db, enrollment_id=enrollment_id, actor=payload.actor)
    return enrollment_response(enrollment)


@internal_router.post(
    "/enrollments/{enrollment_id}/activate",
    response_model=EnrollmentCompleteResponse,
    tags=["enrollment"],
)
@internal_router.post(
    "/enrollments/{enrollment_id}/bkd-approve",
    response_model=EnrollmentCompleteResponse,
    tags=["enrollment"],
)
def enrollment_activate(
    enrollment_id: str,
    payload: ReviewRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> EnrollmentCompleteResponse:
    enrollment, template = activate_enrollment(
        db,
        get_container(request),
        enrollment_id=enrollment_id,
        actor=payload.actor,
    )
    return EnrollmentCompleteResponse(
        enrollment_id=enrollment.id,
        subject_id=enrollment.subject_id,
        template_id=template.id,
        status=enrollment.status,
        sample_count=template.sample_count,
        model_id=template.model_id,
        liveness_assurance="client_active_plus_passive_sequence_v1",
    )


@internal_router.post(
    "/enrollments/{enrollment_id}/reject", response_model=EnrollmentResponse, tags=["enrollment"]
)
def enrollment_reject(
    enrollment_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> EnrollmentResponse:
    enrollment = review_enrollment(
        db,
        enrollment_id=enrollment_id,
        actor=payload.actor,
        action="rejected",
        reason=payload.reason,
    )
    return enrollment_response(enrollment)


@internal_router.post(
    "/enrollments/{enrollment_id}/request-recapture",
    response_model=EnrollmentResponse,
    tags=["enrollment"],
)
def enrollment_request_recapture(
    enrollment_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> EnrollmentResponse:
    enrollment = review_enrollment(
        db,
        enrollment_id=enrollment_id,
        actor=payload.actor,
        action="needs_recapture",
        reason=payload.reason,
    )
    return enrollment_response(enrollment)


@internal_router.get("/profiles/{subject_id}", response_model=ProfileResponse, tags=["profiles"])
def profile_detail(subject_id: str, db: Session = Depends(get_db)) -> ProfileResponse:
    profile = db.get(BiometricProfile, subject_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Biometric profile not found")
    return profile_response(profile)


@internal_router.post("/profiles/{subject_id}/revoke", response_model=ProfileResponse, tags=["profiles"])
def profile_revoke(
    subject_id: str,
    payload: ReviewRequest,
    db: Session = Depends(get_db),
) -> ProfileResponse:
    profile = revoke_profile(db, subject_id=subject_id, actor=payload.actor, reason=payload.reason)
    return profile_response(profile)


@internal_router.delete(
    "/profiles/by-nip/{nip}/template",
    response_model=TemplateDeletionResponse,
    tags=["profiles"],
)
def profile_template_delete_for_reenrollment(
    payload: ReviewRequest,
    nip: str = Path(pattern=r"^[0-9]{18}$"),
    db: Session = Depends(get_db),
) -> TemplateDeletionResponse:
    subject_id = f"nip-{nip}"
    profile, deleted_count, cancelled_count = delete_templates_for_reenrollment(
        db,
        subject_id=subject_id,
        actor=payload.actor,
        reason=payload.reason,
    )
    return TemplateDeletionResponse(
        subject_id=profile.subject_id,
        status=profile.status,
        deleted_template_count=deleted_count,
        cancelled_enrollment_count=cancelled_count,
        message="Biometric templates deleted; employee may enroll again",
    )


def _normalize_report_range(
    created_from: datetime | None, created_to: datetime | None
) -> tuple[datetime | None, datetime | None]:
    def normalize(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    normalized_from = normalize(created_from)
    normalized_to = normalize(created_to)
    if normalized_from and normalized_to and normalized_from > normalized_to:
        raise HTTPException(status_code=422, detail="created_from must be before or equal to created_to")
    return normalized_from, normalized_to


@internal_router.get("/reports/enrollments.csv", tags=["reports"])
def enrollment_report(
    status_filter: str | None = Query(default=None, alias="status", max_length=32),
    subject_id: str | None = Query(default=None, max_length=128),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    created_from, created_to = _normalize_report_range(created_from, created_to)
    rows = enrollment_report_rows(
        db,
        status=status_filter,
        subject_id=subject_id,
        created_from=created_from,
        created_to=created_to,
    )
    return StreamingResponse(
        csv_stream(
            [
                "enrollment_id",
                "subject_id",
                "status",
                "sample_count",
                "liveness_passed",
                "liveness_method",
                "candidate_template_id",
                "rejection_reason",
                "created_at",
                "expires_at",
                "completed_at",
                "reviewed_at",
                "reviewed_by",
            ],
            rows,
        ),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="enrollments.csv"'},
    )


@internal_router.get("/reports/verifications.csv", tags=["reports"])
def verification_report(
    status_filter: str | None = Query(default=None, alias="status", max_length=32),
    subject_id: str | None = Query(default=None, max_length=128),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    created_from, created_to = _normalize_report_range(created_from, created_to)
    rows = verification_report_rows(
        db,
        status=status_filter,
        subject_id=subject_id,
        created_from=created_from,
        created_to=created_to,
    )
    return StreamingResponse(
        csv_stream(
            [
                "verification_id",
                "subject_id",
                "status",
                "score",
                "threshold",
                "matched",
                "liveness_passed",
                "failure_reason",
                "created_at",
                "expires_at",
                "completed_at",
            ],
            rows,
        ),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="verifications.csv"'},
    )


@internal_router.post(
    "/verifications", response_model=SessionCreateResponse, status_code=201, tags=["verification"]
)
def verification_start(
    payload: VerificationCreate,
    request: Request,
    db: Session = Depends(get_db),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> SessionCreateResponse:
    verification, token = create_verification(
        db,
        get_container(request),
        subject_id=payload.subject_id,
        idempotency_key=idempotency_key,
    )
    return SessionCreateResponse(
        id=verification.id,
        subject_id=verification.subject_id,
        status=verification.status,
        challenge=json.loads(verification.challenge),
        upload_token=token,
        expires_at=verification.expires_at,
    )


@internal_router.post(
    "/offline-verifications",
    response_model=OfflineVerificationResponse,
    tags=["offline-biometric"],
)
async def offline_verification(
    request: Request,
    subject_id: str = Form(...),
    model_id: str = Form(...),
    face_package_id: str = Form(...),
    captured_at: datetime = Form(...),
    liveness_method: str = Form(...),
    evidence_hash: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> OfflineVerificationResponse:
    client = request.client.host if request.client else "unknown"
    container = get_container(request)
    if not container.offline_verification_limiter.allow(client):
        raise HTTPException(status_code=429, detail="Offline verification rate limit exceeded")
    if captured_at.tzinfo is None:
        raise HTTPException(status_code=422, detail="captured_at must include a timezone offset")
    if len(model_id) > 160 or not model_id:
        raise HTTPException(status_code=422, detail="model_id is invalid")
    if len(face_package_id) > 64 or not face_package_id:
        raise HTTPException(status_code=422, detail="face_package_id is invalid")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", liveness_method):
        raise HTTPException(status_code=422, detail="liveness_method is invalid")
    if not re.fullmatch(r"(?:sha256:)?[0-9a-fA-F]{64}", evidence_hash):
        raise HTTPException(status_code=422, detail="evidence_hash must be a SHA-256 digest")
    content = await read_image(image, container)
    return OfflineVerificationResponse.model_validate(
        verify_offline_attendance(
            db,
            container,
            subject_id=subject_id,
            model_id=model_id,
            face_package_id=face_package_id,
            captured_at=captured_at,
            liveness_method=liveness_method,
            evidence_hash=evidence_hash,
            image_bytes=content,
        )
    )


@upload_router.post(
    "/verifications/{verification_id}/capture",
    response_model=VerificationCaptureResponse,
    tags=["verification"],
)
async def verification_capture(
    verification_id: str,
    request: Request,
    liveness_passed: bool = Form(...),
    image: UploadFile = File(...),
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> VerificationCaptureResponse:
    claims = require_upload_scope(request, authorization, purpose="verification", session_id=verification_id)
    container = get_container(request)
    content = await read_image(image, container)
    verification, model_id, quality_passed, _quality = complete_verification(
        db,
        container,
        verification_id=verification_id,
        subject_id=claims["sub"],
        image_bytes=content,
        liveness_passed=liveness_passed,
    )
    return VerificationCaptureResponse(
        verification_id=verification.id,
        subject_id=verification.subject_id,
        status=verification.status,
        face_detected=verification.failure_reason != "face_count_invalid",
        quality_passed=quality_passed,
        liveness_passed=bool(verification.liveness_passed),
        match=bool(verification.matched),
        score=float(verification.score if verification.score is not None else -1.0),
        threshold=float(
            verification.threshold
            if verification.threshold is not None
            else container.settings.verification_threshold
        ),
        model_id=model_id,
        reason_code=verification.failure_reason,
    )


@router.get("/", response_model=MessageResponse, include_in_schema=False)
def root() -> MessageResponse:
    return MessageResponse(status="ok", message="ABSEN Face Recognition Service")


router.include_router(mobile_router)
router.include_router(internal_router)
router.include_router(upload_router)
