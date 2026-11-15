from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.container import Container
from app.engines.base import cosine_similarity, normalize_embedding
from app.models import (
    AuditLog,
    BiometricProfile,
    BiometricTemplate,
    OfflineBiometricPackage,
    OfflineVerification,
)
from app.offline_crypto import deterministic_json_bytes, load_device_public_key

WIT = timezone(timedelta(hours=9))


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def _nip(subject_id: str) -> str:
    value = subject_id.removeprefix("nip-")
    if len(value) != 18 or not value.isdigit():
        raise HTTPException(status_code=422, detail="Subject must contain a valid 18-digit NIP")
    return value


def _audit(
    db: Session,
    *,
    action: str,
    entity_type: str,
    entity_id: str,
    subject_id: str,
    metadata: dict[str, object],
) -> None:
    db.add(
        AuditLog(
            actor="internal-service" if action.startswith("offline_verification") else "mobile",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            subject_id=subject_id,
            metadata_json=json.dumps(metadata, separators=(",", ":"), sort_keys=True),
        )
    )


def issue_offline_package(
    db: Session,
    container: Container,
    *,
    subject_id: str,
    device_id: str,
    device_public_key_header: str,
    now: datetime | None = None,
) -> dict[str, object]:
    signer = container.offline_signer
    if signer is None:
        raise HTTPException(status_code=503, detail="Offline package signing is not configured")

    profile = db.scalar(
        select(BiometricProfile).where(BiometricProfile.subject_id == subject_id).with_for_update()
    )
    template = (
        db.scalar(
            select(BiometricTemplate)
            .where(BiometricTemplate.id == profile.active_template_id)
            .with_for_update()
        )
        if profile and profile.active_template_id
        else None
    )
    if not profile or profile.status != "active" or not template or template.status != "active":
        raise HTTPException(status_code=409, detail="Subject has no active biometric template")
    if template.model_id != container.engine.model_id:
        raise HTTPException(status_code=409, detail="Biometric template model is no longer current")
    if profile.template_version < 1 or template.template_version != profile.template_version:
        raise HTTPException(status_code=409, detail="Biometric template version is inconsistent")

    try:
        device_key, fingerprint = load_device_public_key(
            device_public_key_header,
            max_header_bytes=container.settings.offline_device_public_key_max_header_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # The decrypted vector exists only in this local scope and is never logged or persisted.
    embedding = normalize_embedding(
        container.cipher.decrypt_embedding(
            template.encrypted_embedding,
            template.embedding_dimension,
        )
    )
    encrypted = signer.encrypt_for_device(embedding, device_key)

    issued_at = (now or datetime.now(WIT)).astimezone(WIT).replace(microsecond=0)
    expires_at = issued_at + timedelta(seconds=container.settings.offline_package_ttl_seconds)
    package_id = str(uuid4())
    payload = {
        "packageId": package_id,
        "nip": _nip(subject_id),
        "deviceId": device_id,
        "modelId": template.model_id,
        "templateVersion": template.template_version,
        "threshold": container.settings.verification_threshold,
        "issuedAt": issued_at.isoformat(),
        "expiresAt": expires_at.isoformat(),
        "wrappedKey": encrypted.wrapped_key,
        "iv": encrypted.iv,
        "encryptedEmbedding": encrypted.ciphertext,
    }
    payload_bytes = deterministic_json_bytes(payload)
    signature = signer.sign(payload_bytes)

    db.add(
        OfflineBiometricPackage(
            id=package_id,
            subject_id=subject_id,
            device_id=device_id,
            device_public_key_fingerprint=fingerprint,
            template_id=template.id,
            template_version=template.template_version,
            model_id=template.model_id,
            signing_key_id=signer.key_id,
            issued_at=issued_at.astimezone(UTC),
            expires_at=expires_at.astimezone(UTC),
        )
    )
    _audit(
        db,
        action="offline_package.issued",
        entity_type="offline_biometric_package",
        entity_id=package_id,
        subject_id=subject_id,
        metadata={
            "device_id": device_id,
            "device_public_key_fingerprint": fingerprint,
            "template_version": template.template_version,
            "model_id": template.model_id,
            "expires_at": expires_at.isoformat(),
        },
    )
    db.commit()
    return {
        "status": "Success",
        "message": None,
        "data": {
            "payload": base64.b64encode(payload_bytes).decode("ascii"),
            "signature": signature,
            "key_id": signer.key_id,
        },
    }


def _verification_result(
    db: Session,
    *,
    subject_id: str,
    package: OfflineBiometricPackage,
    captured_at: datetime,
    liveness_method: str,
    evidence_hash: str,
    threshold: float,
    score: float = -1.0,
    matched: bool = False,
    reason_code: str | None = None,
) -> dict[str, object]:
    verification = OfflineVerification(
        subject_id=subject_id,
        package_id=package.id,
        model_id=package.model_id,
        template_version=package.template_version,
        captured_at=captured_at,
        liveness_method=liveness_method,
        evidence_hash=evidence_hash,
        score=score,
        threshold=threshold,
        matched=matched,
        reason_code=reason_code,
    )
    db.add(verification)
    db.flush()
    _audit(
        db,
        action="offline_verification.completed",
        entity_type="offline_verification",
        entity_id=verification.id,
        subject_id=subject_id,
        metadata={
            "package_id": package.id,
            "model_id": package.model_id,
            "template_version": package.template_version,
            "matched": matched,
            "reason_code": reason_code,
        },
    )
    db.commit()
    return {
        "verification_id": verification.id,
        "match": matched,
        "score": score,
        "threshold": threshold,
        "model_id": package.model_id,
        "template_version": package.template_version,
        "reason_code": reason_code,
    }


def verify_offline_attendance(
    db: Session,
    container: Container,
    *,
    subject_id: str,
    model_id: str,
    face_package_id: str,
    captured_at: datetime,
    liveness_method: str,
    evidence_hash: str,
    image_bytes: bytes,
    now: datetime | None = None,
) -> dict[str, object]:
    package = db.scalar(
        select(OfflineBiometricPackage).where(OfflineBiometricPackage.id == face_package_id).with_for_update()
    )
    if not package:
        raise HTTPException(status_code=404, detail="Offline biometric package not found")

    threshold = container.settings.verification_threshold
    request_subject = f"nip-{_nip(subject_id)}"
    captured = _aware(captured_at)
    current = _aware(now or datetime.now(UTC))
    normalized_hash = evidence_hash.lower().removeprefix("sha256:")

    def reject(code: str) -> dict[str, object]:
        return _verification_result(
            db,
            subject_id=request_subject,
            package=package,
            captured_at=captured,
            liveness_method=liveness_method,
            evidence_hash=normalized_hash,
            threshold=threshold,
            reason_code=code,
        )

    if package.subject_id != request_subject:
        return reject("subject_mismatch")
    if package.revoked_at is not None:
        return reject("package_revoked")
    if current > _aware(package.expires_at) or not (
        _aware(package.issued_at) <= captured <= _aware(package.expires_at)
    ):
        return reject("package_expired")
    if model_id != package.model_id or model_id != container.engine.model_id:
        return reject("model_mismatch")

    profile = db.scalar(
        select(BiometricProfile).where(BiometricProfile.subject_id == request_subject).with_for_update()
    )
    template = (
        db.scalar(
            select(BiometricTemplate)
            .where(BiometricTemplate.id == profile.active_template_id)
            .with_for_update()
        )
        if profile and profile.active_template_id
        else None
    )
    if not profile or profile.status != "active" or not template or template.status != "active":
        return reject("subject_inactive")
    if (
        profile.template_version != package.template_version
        or template.template_version != package.template_version
        or template.id != package.template_id
    ):
        return reject("template_changed")
    if template.model_id != model_id:
        return reject("model_mismatch")

    if not hashlib.sha256(image_bytes).hexdigest() == normalized_hash:
        return reject("evidence_hash_mismatch")

    analysis = container.engine.analyze(image_bytes)
    if analysis.face_count != 1:
        return reject("face_count_invalid")
    if not analysis.quality_passed or analysis.embedding is None:
        return reject(analysis.reason or "image_quality_failed")
    if analysis.model_id != model_id:
        return reject("model_mismatch")

    stored = container.cipher.decrypt_embedding(
        template.encrypted_embedding,
        template.embedding_dimension,
    )
    score = cosine_similarity(analysis.embedding, stored)
    matched = score >= threshold
    return _verification_result(
        db,
        subject_id=request_subject,
        package=package,
        captured_at=captured,
        liveness_method=liveness_method,
        evidence_hash=normalized_hash,
        threshold=threshold,
        score=score,
        matched=matched,
        reason_code=None if matched else "face_not_matched",
    )
