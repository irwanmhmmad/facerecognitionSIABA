from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi import HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.container import Container
from app.engines.base import cosine_similarity, normalize_embedding
from app.models import (
    AuditLog,
    BiometricProfile,
    BiometricTemplate,
    EnrollmentSample,
    EnrollmentSampleImage,
    EnrollmentSession,
    OfflineBiometricPackage,
    VerificationSession,
    utc_now,
)
from app.security import issue_upload_token

ACTIVE_ENROLLMENT_STATUSES = {"collecting", "pending_ktu_review", "pending_bkd_review"}
UPLOADED_ENROLLMENT_STATUSES = {
    "pending_ktu_review",
    "pending_bkd_review",
    "active",
    "rejected",
    "needs_recapture",
}
CAPTURE_TYPES = {"frontal", "slight_left", "slight_right", "additional"}


def _face_count_error_detail(face_count: int) -> str:
    if face_count == 0:
        return (
            "Wajah tidak terdeteksi pada foto. Pastikan wajah terlihat utuh, berada di tengah "
            "bingkai, dan mendapat cahaya dari arah depan."
        )
    return (
        f"Terdeteksi {face_count} wajah pada foto. Pastikan hanya satu orang berada di depan "
        "kamera dan tidak ada foto wajah lain di latar belakang."
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _challenge() -> list[str]:
    return random.sample(["blink", "turn-left", "turn-right"], k=3)


def _audit(
    db: Session,
    *,
    actor: str,
    action: str,
    entity_type: str,
    entity_id: str,
    subject_id: str | None,
    metadata: dict | None = None,
) -> None:
    db.add(
        AuditLog(
            actor=actor,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata or {}, separators=(",", ":")),
        )
    )


def _upload_token(container: Container, purpose: str, session_id: str, subject_id: str) -> str:
    settings = container.settings
    return issue_upload_token(
        secret=settings.upload_token_secret,
        issuer=settings.token_issuer,
        purpose=purpose,
        session_id=session_id,
        subject_id=subject_id,
        ttl_seconds=settings.upload_token_ttl_seconds,
    )


def _discard_enrollment_evidence(db: Session, enrollment_id: str) -> None:
    """Remove temporary photos and per-photo embeddings once review no longer needs them."""

    samples = db.scalars(select(EnrollmentSample).where(EnrollmentSample.enrollment_id == enrollment_id))
    for sample in samples:
        sample.encrypted_embedding = None
    db.execute(delete(EnrollmentSampleImage).where(EnrollmentSampleImage.enrollment_id == enrollment_id))


def _revoke_offline_packages(db: Session, subject_id: str, now: datetime) -> int:
    packages = list(
        db.scalars(
            select(OfflineBiometricPackage).where(
                OfflineBiometricPackage.subject_id == subject_id,
                OfflineBiometricPackage.revoked_at.is_(None),
            )
        )
    )
    for package in packages:
        package.revoked_at = now
    return len(packages)


def create_enrollment(
    db: Session,
    container: Container,
    *,
    subject_id: str,
    idempotency_key: str | None,
    struktur_org: str | None = None,
) -> tuple[EnrollmentSession, str]:
    if idempotency_key:
        existing = db.scalar(
            select(EnrollmentSession).where(EnrollmentSession.idempotency_key == idempotency_key)
        )
        if existing:
            return existing, _upload_token(container, "enrollment", existing.id, existing.subject_id)

    active = db.scalar(
        select(EnrollmentSession).where(
            EnrollmentSession.subject_id == subject_id,
            EnrollmentSession.status.in_(ACTIVE_ENROLLMENT_STATUSES),
        )
    )
    if active:
        if active.status == "collecting":
            active.status = "expired"
            _discard_enrollment_evidence(db, active.id)
            db.commit()
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An active enrollment already exists",
            )

    profile = db.get(BiometricProfile, subject_id)
    if not profile:
        profile = BiometricProfile(subject_id=subject_id)
        db.add(profile)

    session = EnrollmentSession(
        subject_id=subject_id,
        struktur_org=struktur_org,
        idempotency_key=idempotency_key,
        challenge=json.dumps(_challenge()),
        expires_at=utc_now() + timedelta(seconds=container.settings.enrollment_ttl_seconds),
    )
    db.add(session)
    db.flush()
    _audit(
        db,
        actor="internal-service",
        action="enrollment.created",
        entity_type="enrollment",
        entity_id=session.id,
        subject_id=subject_id,
    )
    db.commit()
    return session, _upload_token(container, "enrollment", session.id, subject_id)


def add_enrollment_capture(
    db: Session,
    container: Container,
    *,
    enrollment_id: str,
    subject_id: str,
    sample_type: str,
    image_bytes: bytes,
    image_content_type: str,
    capture_metadata: dict[str, object] | None = None,
) -> EnrollmentSample:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment or enrollment.subject_id != subject_id:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status != "collecting":
        raise HTTPException(status_code=409, detail="Enrollment is not accepting samples")
    if _aware(enrollment.expires_at) <= utc_now():
        enrollment.status = "expired"
        _discard_enrollment_evidence(db, enrollment.id)
        db.commit()
        raise HTTPException(status_code=410, detail="Enrollment has expired")
    if sample_type not in CAPTURE_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported sample_type")
    if enrollment.sample_count >= container.settings.enrollment_sample_count:
        raise HTTPException(status_code=409, detail="Maximum enrollment samples reached")

    analysis = container.engine.analyze(image_bytes)
    if analysis.face_count != 1:
        raise HTTPException(
            status_code=422,
            detail=_face_count_error_detail(analysis.face_count),
        )
    if not analysis.quality_passed or analysis.embedding is None:
        quality_messages = {
            "invalid_image": "Berkas foto tidak dapat dibaca. Silakan ambil ulang foto.",
            "image_quality_failed": (
                "Kualitas foto belum memenuhi syarat. Pastikan cahaya cukup, kamera tidak buram, "
                "dan wajah tidak terlalu jauh."
            ),
            "embedding_unavailable": "Ciri wajah belum dapat dibentuk. Silakan ambil ulang foto.",
        }
        raise HTTPException(
            status_code=422,
            detail=quality_messages.get(
                analysis.reason or "",
                "Foto wajah belum dapat diproses. Silakan ambil ulang foto.",
            ),
        )

    checksum = hashlib.sha256(image_bytes).hexdigest()
    quality = {**analysis.quality, "_content_sha256": checksum}
    sample = EnrollmentSample(
        enrollment_id=enrollment.id,
        sample_type=sample_type,
        encrypted_embedding=container.cipher.encrypt_embedding(analysis.embedding),
        embedding_dimension=int(analysis.embedding.size),
        model_id=analysis.model_id,
        quality=json.dumps(quality, separators=(",", ":")),
    )
    db.add(sample)
    enrollment.sample_count += 1
    db.flush()
    db.add(
        EnrollmentSampleImage(
            sample_id=sample.id,
            enrollment_id=enrollment.id,
            encrypted_image=container.cipher.encrypt_bytes(image_bytes),
            content_type=image_content_type,
            byte_size=len(image_bytes),
            checksum_sha256=checksum,
            capture_metadata=json.dumps(capture_metadata or {}, separators=(",", ":")),
        )
    )
    _audit(
        db,
        actor="mobile-upload-token",
        action="enrollment.sample_accepted",
        entity_type="enrollment_sample",
        entity_id=sample.id,
        subject_id=enrollment.subject_id,
        metadata={"sample_type": sample_type, "model_id": analysis.model_id},
    )
    db.commit()
    return sample


def complete_enrollment(
    db: Session,
    container: Container,
    *,
    enrollment_id: str,
    subject_id: str,
    liveness_passed: bool,
    liveness_method: str,
    completed_challenges: list[str],
    challenge_duration_ms: int,
) -> EnrollmentSession:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment or enrollment.subject_id != subject_id:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status != "collecting":
        raise HTTPException(status_code=409, detail="Enrollment cannot be completed")
    if _aware(enrollment.expires_at) <= utc_now():
        enrollment.status = "expired"
        _discard_enrollment_evidence(db, enrollment.id)
        db.commit()
        raise HTTPException(status_code=410, detail="Enrollment has expired")
    if not liveness_passed:
        raise HTTPException(status_code=422, detail="Active liveness challenge did not pass")
    expected_challenges = json.loads(enrollment.challenge)
    if completed_challenges != expected_challenges:
        raise HTTPException(status_code=422, detail="Liveness challenge evidence does not match the session")

    samples = list(
        db.scalars(
            select(EnrollmentSample)
            .where(EnrollmentSample.enrollment_id == enrollment.id)
            .order_by(EnrollmentSample.created_at)
        )
    )
    required_count = container.settings.enrollment_sample_count
    if len(samples) != required_count:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {required_count} enrollment samples are required",
        )

    # Manual review in Absen Web needs the original captures, not only embeddings.
    # Do not allow an enrollment to enter the review queue unless every accepted
    # sample has its encrypted temporary image available.
    stored_images = list(
        db.scalars(select(EnrollmentSampleImage).where(EnrollmentSampleImage.enrollment_id == enrollment.id))
    )
    sample_ids = {sample.id for sample in samples}
    stored_sample_ids = {image.sample_id for image in stored_images}
    if len(stored_images) != required_count or stored_sample_ids != sample_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Temporary enrollment photos are incomplete. "
                "Restart face enrollment and upload all samples again."
            ),
        )
    if any(not image.encrypted_image or image.byte_size <= 0 for image in stored_images):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A temporary enrollment photo is empty or unreadable",
        )
    if not any(sample.sample_type == "frontal" for sample in samples):
        raise HTTPException(status_code=422, detail="At least one frontal sample is required")
    required_types = {"frontal", "slight_left", "slight_right"}
    if not required_types.issubset({sample.sample_type for sample in samples}):
        raise HTTPException(status_code=422, detail="Frontal, left, and right samples are required")
    if len({sample.model_id for sample in samples}) != 1:
        raise HTTPException(status_code=409, detail="Samples were produced by different models")

    sample_quality = [json.loads(sample.quality) for sample in samples]
    fingerprints = [item.get("_content_sha256") for item in sample_quality]
    if any(not fingerprint for fingerprint in fingerprints) or len(set(fingerprints)) != len(fingerprints):
        raise HTTPException(status_code=422, detail="Passive liveness rejected duplicate frames")

    enrollment.status = "pending_ktu_review"
    enrollment.client_liveness_passed = True
    enrollment.liveness_method = liveness_method
    enrollment.completed_at = utc_now()
    profile = db.get(BiometricProfile, subject_id)
    if profile and profile.status != "active":
        profile.status = "pending_ktu_review"

    _audit(
        db,
        actor="mobile-upload-token",
        action="enrollment.completed",
        entity_type="enrollment",
        entity_id=enrollment.id,
        subject_id=subject_id,
        metadata={
            "liveness_method": liveness_method,
            "passive_liveness": "sequence_unique_frames_v1",
            "completed_challenges": completed_challenges,
            "challenge_duration_ms": challenge_duration_ms,
            "sample_count": len(samples),
        },
    )
    db.commit()
    return enrollment


def approve_ktu_enrollment(db: Session, *, enrollment_id: str, actor: str) -> EnrollmentSession:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status != "pending_ktu_review":
        raise HTTPException(status_code=409, detail="Enrollment is not awaiting KTU review")

    enrollment.status = "pending_bkd_review"
    enrollment.reviewed_at = utc_now()
    enrollment.reviewed_by = actor
    profile = db.get(BiometricProfile, enrollment.subject_id)
    if profile and profile.status != "active":
        profile.status = "pending_bkd_review"
    _audit(
        db,
        actor=actor,
        action="enrollment.ktu_approved",
        entity_type="enrollment",
        entity_id=enrollment.id,
        subject_id=enrollment.subject_id,
    )
    db.commit()
    return enrollment


def _build_candidate_template(
    db: Session, container: Container, enrollment: EnrollmentSession
) -> BiometricTemplate:
    samples = list(
        db.scalars(
            select(EnrollmentSample)
            .where(EnrollmentSample.enrollment_id == enrollment.id)
            .order_by(EnrollmentSample.created_at)
        )
    )
    if len(samples) != container.settings.enrollment_sample_count:
        raise HTTPException(status_code=409, detail="Enrollment sample set is incomplete")
    if any(sample.encrypted_embedding is None for sample in samples):
        raise HTTPException(status_code=409, detail="Enrollment samples are no longer available")

    embeddings = [
        container.cipher.decrypt_embedding(sample.encrypted_embedding, sample.embedding_dimension)
        for sample in samples
        if sample.encrypted_embedding is not None
    ]
    centroid = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
    similarities = [cosine_similarity(item, centroid) for item in embeddings]
    if min(similarities) < container.settings.enrollment_consistency_threshold:
        raise HTTPException(status_code=422, detail="Enrollment samples are not sufficiently consistent")

    quality_summary = {
        "minimum_consistency": round(min(similarities), 6),
        "average_consistency": round(float(np.mean(similarities)), 6),
        "sample_types": [sample.sample_type for sample in samples],
        "passive_liveness": "sequence_unique_frames_v1",
        "trained_after_bkd_approval": True,
    }
    template = BiometricTemplate(
        subject_id=enrollment.subject_id,
        enrollment_id=enrollment.id,
        status="pending",
        encrypted_embedding=container.cipher.encrypt_embedding(centroid),
        embedding_dimension=int(centroid.size),
        model_id=samples[0].model_id,
        sample_count=len(samples),
        quality_summary=json.dumps(quality_summary, separators=(",", ":")),
    )
    db.add(template)
    db.flush()
    _discard_enrollment_evidence(db, enrollment.id)
    enrollment.candidate_template_id = template.id
    return template


def activate_enrollment(
    db: Session, container: Container, *, enrollment_id: str, actor: str
) -> tuple[EnrollmentSession, BiometricTemplate]:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status != "pending_bkd_review":
        raise HTTPException(status_code=409, detail="Enrollment is not awaiting BKD review")

    profile = db.get(BiometricProfile, enrollment.subject_id)
    if not profile:
        raise HTTPException(status_code=409, detail="Enrollment profile is incomplete")
    template = _build_candidate_template(db, container, enrollment)

    previous_templates = list(
        db.scalars(
            select(BiometricTemplate).where(
                BiometricTemplate.subject_id == enrollment.subject_id,
                BiometricTemplate.status == "active",
            )
        )
    )
    now = utc_now()
    for previous in previous_templates:
        previous.status = "revoked"
        previous.revoked_at = now

    # A reset increments the version immediately. A direct replacement increments it here.
    # This makes every package from an older biometric lifecycle unambiguously stale.
    if previous_templates or profile.template_version == 0:
        profile.template_version += 1
    template.template_version = profile.template_version
    revoked_package_count = _revoke_offline_packages(db, enrollment.subject_id, now)

    template.status = "active"
    template.activated_at = now
    enrollment.status = "active"
    enrollment.reviewed_at = now
    enrollment.reviewed_by = actor
    profile.status = "active"
    profile.active_template_id = template.id
    _audit(
        db,
        actor=actor,
        action="enrollment.bkd_approved_and_activated",
        entity_type="biometric_template",
        entity_id=template.id,
        subject_id=enrollment.subject_id,
        metadata={
            "template_version": template.template_version,
            "revoked_offline_package_count": revoked_package_count,
        },
    )
    db.commit()
    return enrollment, template


def review_enrollment(
    db: Session,
    *,
    enrollment_id: str,
    actor: str,
    action: str,
    reason: str | None,
) -> EnrollmentSession:
    enrollment = db.get(EnrollmentSession, enrollment_id)
    if not enrollment:
        raise HTTPException(status_code=404, detail="Enrollment not found")
    if enrollment.status not in {"pending_ktu_review", "pending_bkd_review"}:
        raise HTTPException(status_code=409, detail="Enrollment is not awaiting review")
    if action not in {"rejected", "needs_recapture"}:
        raise ValueError("Unsupported review action")

    review_stage = enrollment.status
    enrollment.status = action
    enrollment.rejection_reason = reason
    enrollment.reviewed_at = utc_now()
    enrollment.reviewed_by = actor
    if enrollment.candidate_template_id:
        template = db.get(BiometricTemplate, enrollment.candidate_template_id)
        if template:
            template.status = action
    _discard_enrollment_evidence(db, enrollment.id)
    profile = db.get(BiometricProfile, enrollment.subject_id)
    if profile and profile.status != "active":
        profile.status = action
    _audit(
        db,
        actor=actor,
        action=f"enrollment.{action}",
        entity_type="enrollment",
        entity_id=enrollment.id,
        subject_id=enrollment.subject_id,
        metadata={"reason": reason, "review_stage": review_stage},
    )
    db.commit()
    return enrollment


def create_verification(
    db: Session,
    container: Container,
    *,
    subject_id: str,
    idempotency_key: str | None,
) -> tuple[VerificationSession, str]:
    if idempotency_key:
        existing = db.scalar(
            select(VerificationSession).where(VerificationSession.idempotency_key == idempotency_key)
        )
        if existing:
            return existing, _upload_token(container, "verification", existing.id, existing.subject_id)

    profile = db.get(BiometricProfile, subject_id)
    template = (
        db.get(BiometricTemplate, profile.active_template_id)
        if profile and profile.active_template_id
        else None
    )
    if not profile or profile.status != "active" or not template or template.status != "active":
        raise HTTPException(status_code=409, detail="Subject has no active biometric template")
    if template.model_id != container.engine.model_id:
        raise HTTPException(
            status_code=409,
            detail="Subject biometric template requires reenrollment for the active face model",
        )

    verification = VerificationSession(
        subject_id=subject_id,
        idempotency_key=idempotency_key,
        challenge=json.dumps(_challenge()),
        expires_at=utc_now() + timedelta(seconds=container.settings.verification_ttl_seconds),
    )
    db.add(verification)
    db.flush()
    _audit(
        db,
        actor="internal-service",
        action="verification.created",
        entity_type="verification",
        entity_id=verification.id,
        subject_id=subject_id,
    )
    db.commit()
    return verification, _upload_token(container, "verification", verification.id, subject_id)


def complete_verification(
    db: Session,
    container: Container,
    *,
    verification_id: str,
    subject_id: str,
    image_bytes: bytes,
    liveness_passed: bool,
) -> tuple[VerificationSession, str, bool, dict]:
    verification = db.get(VerificationSession, verification_id)
    if not verification or verification.subject_id != subject_id:
        raise HTTPException(status_code=404, detail="Verification not found")
    if verification.status != "created":
        raise HTTPException(status_code=409, detail="Verification was already completed")
    if _aware(verification.expires_at) <= utc_now():
        verification.status = "expired"
        db.commit()
        raise HTTPException(status_code=410, detail="Verification has expired")

    profile = db.get(BiometricProfile, subject_id)
    template = db.get(BiometricTemplate, profile.active_template_id) if profile else None
    if not profile or not template or template.status != "active":
        raise HTTPException(status_code=409, detail="Subject has no active biometric template")

    analysis = container.engine.analyze(image_bytes)
    reason: str | None = None
    score = -1.0
    matched = False
    if not liveness_passed:
        reason = "liveness_failed"
    elif analysis.face_count != 1:
        reason = "face_count_invalid"
    elif not analysis.quality_passed or analysis.embedding is None:
        reason = analysis.reason or "image_quality_failed"
    elif analysis.model_id != template.model_id:
        reason = "model_mismatch"
    else:
        stored = container.cipher.decrypt_embedding(
            template.encrypted_embedding, template.embedding_dimension
        )
        score = cosine_similarity(analysis.embedding, stored)
        matched = score >= container.settings.verification_threshold
        if not matched:
            reason = "face_not_matched"

    verification.status = "matched" if matched else "failed"
    verification.score = score
    verification.threshold = container.settings.verification_threshold
    verification.matched = matched
    verification.liveness_passed = liveness_passed
    verification.failure_reason = reason
    verification.completed_at = utc_now()
    _audit(
        db,
        actor="mobile-upload-token",
        action="verification.completed",
        entity_type="verification",
        entity_id=verification.id,
        subject_id=subject_id,
        metadata={"matched": matched, "reason": reason, "model_id": analysis.model_id},
    )
    db.commit()
    return verification, analysis.model_id, analysis.quality_passed, analysis.quality


def revoke_profile(db: Session, *, subject_id: str, actor: str, reason: str | None) -> BiometricProfile:
    profile = db.get(BiometricProfile, subject_id)
    if not profile or not profile.active_template_id:
        raise HTTPException(status_code=404, detail="Active biometric profile not found")
    template = db.get(BiometricTemplate, profile.active_template_id)
    now = utc_now()
    if template:
        template.status = "revoked"
        template.revoked_at = now
    previous_id = profile.active_template_id
    profile.active_template_id = None
    profile.status = "revoked"
    profile.template_version += 1
    revoked_package_count = _revoke_offline_packages(db, subject_id, now)
    _audit(
        db,
        actor=actor,
        action="profile.revoked",
        entity_type="biometric_profile",
        entity_id=subject_id,
        subject_id=subject_id,
        metadata={
            "template_id": previous_id,
            "reason": reason,
            "template_version": profile.template_version,
            "revoked_offline_package_count": revoked_package_count,
        },
    )
    db.commit()
    return profile


def delete_templates_for_reenrollment(
    db: Session, *, subject_id: str, actor: str, reason: str | None
) -> tuple[BiometricProfile, int, int]:
    """Delete a subject's biometric templates and allow a fresh enrollment.

    Enrollment history and the audit trail are retained, but no reusable face
    embedding remains. Any unfinished enrollment is closed and its temporary
    evidence is discarded so it cannot block the replacement enrollment.
    """

    profile = db.get(BiometricProfile, subject_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Biometric profile not found")

    templates = list(db.scalars(select(BiometricTemplate).where(BiometricTemplate.subject_id == subject_id)))

    # A retried request after a successful reset must not cancel a new capture
    # session that the employee has already started.
    if profile.status == "needs_recapture" and profile.active_template_id is None and not templates:
        return profile, 0, 0

    now = utc_now()
    open_enrollments = list(
        db.scalars(
            select(EnrollmentSession).where(
                EnrollmentSession.subject_id == subject_id,
                EnrollmentSession.status.in_({"collecting", "pending_ktu_review", "pending_bkd_review"}),
            )
        )
    )
    for enrollment in open_enrollments:
        _discard_enrollment_evidence(db, enrollment.id)
        enrollment.status = "needs_recapture"
        enrollment.rejection_reason = reason or "Template deleted for face reenrollment"
        enrollment.reviewed_at = now
        enrollment.reviewed_by = actor

    deleted_template_ids = [template.id for template in templates]
    if deleted_template_ids:
        for enrollment in db.scalars(
            select(EnrollmentSession).where(EnrollmentSession.candidate_template_id.in_(deleted_template_ids))
        ):
            enrollment.candidate_template_id = None
        db.execute(delete(BiometricTemplate).where(BiometricTemplate.subject_id == subject_id))

    profile.active_template_id = None
    profile.status = "needs_recapture"
    profile.template_version += 1
    revoked_package_count = _revoke_offline_packages(db, subject_id, now)
    _audit(
        db,
        actor=actor,
        action="profile.templates_deleted_for_reenrollment",
        entity_type="biometric_profile",
        entity_id=subject_id,
        subject_id=subject_id,
        metadata={
            "deleted_template_ids": deleted_template_ids,
            "cancelled_enrollment_ids": [item.id for item in open_enrollments],
            "reason": reason,
            "template_version": profile.template_version,
            "revoked_offline_package_count": revoked_package_count,
        },
    )
    db.commit()
    return profile, len(deleted_template_ids), len(open_enrollments)


def count_enrollments(db: Session, status_filter: str | None, struktur_org_prefix: str | None = None) -> int:
    query = select(func.count()).select_from(EnrollmentSession)
    if status_filter:
        query = query.where(EnrollmentSession.status == status_filter)
    if struktur_org_prefix:
        query = query.where(EnrollmentSession.struktur_org.like(f"{struktur_org_prefix}%"))
    return int(db.scalar(query) or 0)


def count_uploaded_enrollments(
    db: Session,
    status_filter: str | None,
    struktur_org_prefix: str | None = None,
    subject_id: str | None = None,
) -> int:
    query = select(func.count()).select_from(EnrollmentSession)
    if status_filter:
        query = query.where(EnrollmentSession.status == status_filter)
    else:
        query = query.where(EnrollmentSession.status.in_(UPLOADED_ENROLLMENT_STATUSES))
    if struktur_org_prefix:
        query = query.where(EnrollmentSession.struktur_org.like(f"{struktur_org_prefix}%"))
    if subject_id:
        query = query.where(EnrollmentSession.subject_id == subject_id)
    return int(db.scalar(query) or 0)
