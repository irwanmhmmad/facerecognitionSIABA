from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from io import BytesIO

import numpy as np
import pytest
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import HTTPException
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.models import BiometricProfile, BiometricTemplate, OfflineBiometricPackage
from app.offline_services import issue_offline_package, verify_offline_attendance
from app.services import delete_templates_for_reenrollment

SUBJECT = "nip-199412022025041005"


def sample_image(*, inverted: bool = False) -> bytes:
    y, x = np.indices((160, 160))
    pixels = ((x * 3 + y * 5 + ((x // 12) % 2) * 80) % 256).astype(np.uint8)
    if inverted:
        pixels = 255 - pixels
    rgb = np.stack([pixels, np.roll(pixels, 7, axis=0), np.roll(pixels, 11, axis=1)], axis=-1)
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="JPEG", quality=90)
    return output.getvalue()


def signing_key_b64() -> str:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    return base64.b64encode(pem).decode("ascii")


@pytest.fixture()
def configured_app(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'offline.db'}",
        internal_service_key="internal-test-key",
        upload_token_secret="upload-test-key",
        biometric_encryption_key=Fernet.generate_key().decode("ascii"),
        offline_signing_private_key_b64=signing_key_b64(),
        offline_package_ttl_seconds=604_800,
        face_engine="development",
        verification_threshold=0.55,
    )
    app = create_app(settings)
    app.state.container.database.create_all()
    return app


def seed_active_template(app, db) -> tuple[BiometricProfile, BiometricTemplate, bytes]:
    image = sample_image()
    analysis = app.state.container.engine.analyze(image)
    assert analysis.embedding is not None
    profile = BiometricProfile(subject_id=SUBJECT, status="active", template_version=1)
    db.add(profile)
    db.flush()
    template = BiometricTemplate(
        subject_id=SUBJECT,
        enrollment_id="enr_offline_test",
        status="active",
        encrypted_embedding=app.state.container.cipher.encrypt_embedding(analysis.embedding),
        embedding_dimension=int(analysis.embedding.size),
        model_id=analysis.model_id,
        sample_count=6,
        template_version=1,
        activated_at=datetime.now(UTC),
    )
    db.add(template)
    db.flush()
    profile.active_template_id = template.id
    db.commit()
    return profile, template, image


def issue_package(app, db):
    device_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    device_der = device_private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    response = issue_offline_package(
        db,
        app.state.container,
        subject_id=SUBJECT,
        device_id="android-test-device",
        device_public_key_header=base64.b64encode(device_der).decode("ascii"),
    )
    payload_bytes = base64.b64decode(response["data"]["payload"])
    return response, json.loads(payload_bytes), payload_bytes, device_private_key


def test_offline_package_can_be_verified_and_decrypted_by_device(configured_app):
    with configured_app.state.container.database.session_factory() as db:
        _profile, template, _image = seed_active_template(configured_app, db)
        response, payload, payload_bytes, device_key = issue_package(configured_app, db)

        assert set(payload) == {
            "packageId",
            "nip",
            "deviceId",
            "modelId",
            "templateVersion",
            "threshold",
            "issuedAt",
            "expiresAt",
            "wrappedKey",
            "iv",
            "encryptedEmbedding",
        }
        assert payload["nip"] == "199412022025041005"
        assert payload["templateVersion"] == 1
        assert datetime.fromisoformat(payload["expiresAt"]) - datetime.fromisoformat(
            payload["issuedAt"]
        ) == timedelta(days=7)

        signing_public_der = base64.b64decode(
            configured_app.state.container.offline_signer.public_key_x509_der_b64
        )
        signing_public_key = serialization.load_der_public_key(signing_public_der)
        signing_public_key.verify(
            base64.b64decode(response["data"]["signature"]),
            payload_bytes,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )

        aes_key = device_key.decrypt(
            base64.b64decode(payload["wrappedKey"]),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        decrypted = AESGCM(aes_key).decrypt(
            base64.b64decode(payload["iv"]),
            base64.b64decode(payload["encryptedEmbedding"]),
            None,
        )
        embedding = np.asarray(json.loads(decrypted), dtype=np.float32)
        assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)
        stored = configured_app.state.container.cipher.decrypt_embedding(
            template.encrypted_embedding, template.embedding_dimension
        )
        assert embedding == pytest.approx(stored, abs=1e-6)

        metadata = db.get(OfflineBiometricPackage, payload["packageId"])
        assert metadata is not None
        assert metadata.device_public_key_fingerprint
        assert not hasattr(metadata, "encrypted_embedding")
        assert not hasattr(metadata, "aes_key")


def test_offline_package_rejects_invalid_key_inactive_profile_and_model(configured_app):
    with configured_app.state.container.database.session_factory() as db:
        profile, template, _image = seed_active_template(configured_app, db)
        with pytest.raises(HTTPException) as invalid:
            issue_offline_package(
                db,
                configured_app.state.container,
                subject_id=SUBJECT,
                device_id="android-test-device",
                device_public_key_header="not-base64",
            )
        assert invalid.value.status_code == 422

        weak_key = rsa.generate_private_key(public_exponent=65537, key_size=1024)
        weak_der = weak_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with pytest.raises(HTTPException) as weak:
            issue_offline_package(
                db,
                configured_app.state.container,
                subject_id=SUBJECT,
                device_id="android-test-device",
                device_public_key_header=base64.b64encode(weak_der).decode("ascii"),
            )
        assert weak.value.status_code == 422

        profile.status = "revoked"
        db.commit()
        with pytest.raises(HTTPException) as inactive:
            issue_package(configured_app, db)
        assert inactive.value.status_code == 409

        profile.status = "active"
        template.model_id = "retired-model"
        db.commit()
        with pytest.raises(HTTPException) as mismatch:
            issue_package(configured_app, db)
        assert mismatch.value.status_code == 409


def test_internal_offline_verification_success_and_failure(configured_app):
    with configured_app.state.container.database.session_factory() as db:
        _profile, _template, image = seed_active_template(configured_app, db)
        _response, payload, _payload_bytes, _device_key = issue_package(configured_app, db)
        captured_at = datetime.fromisoformat(payload["issuedAt"]) + timedelta(minutes=1)

        matched = verify_offline_attendance(
            db,
            configured_app.state.container,
            subject_id=SUBJECT,
            model_id=payload["modelId"],
            face_package_id=payload["packageId"],
            captured_at=captured_at,
            liveness_method="active-challenge-v1",
            evidence_hash=hashlib.sha256(image).hexdigest(),
            image_bytes=image,
            now=captured_at,
        )
        assert matched["match"] is True
        assert matched["reason_code"] is None

        different = sample_image(inverted=True)
        failed = verify_offline_attendance(
            db,
            configured_app.state.container,
            subject_id=SUBJECT,
            model_id=payload["modelId"],
            face_package_id=payload["packageId"],
            captured_at=captured_at,
            liveness_method="active-challenge-v1",
            evidence_hash=hashlib.sha256(different).hexdigest(),
            image_bytes=different,
            now=captured_at,
        )
        assert failed["match"] is False
        assert failed["reason_code"] == "face_not_matched"

        unreadable = b"not-a-real-image"
        poor_quality = verify_offline_attendance(
            db,
            configured_app.state.container,
            subject_id=SUBJECT,
            model_id=payload["modelId"],
            face_package_id=payload["packageId"],
            captured_at=captured_at,
            liveness_method="active-challenge-v1",
            evidence_hash=hashlib.sha256(unreadable).hexdigest(),
            image_bytes=unreadable,
            now=captured_at,
        )
        assert poor_quality["match"] is False
        assert poor_quality["reason_code"] == "face_count_invalid"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("different_nip", "subject_mismatch"),
        ("expired", "package_expired"),
        ("revoked", "package_revoked"),
        ("model", "model_mismatch"),
        ("template", "template_changed"),
    ],
)
def test_internal_offline_verification_rejects_stale_context(configured_app, mutation, reason):
    with configured_app.state.container.database.session_factory() as db:
        profile, _template, image = seed_active_template(configured_app, db)
        _response, payload, _payload_bytes, _device_key = issue_package(configured_app, db)
        package = db.get(OfflineBiometricPackage, payload["packageId"])
        captured_at = datetime.fromisoformat(payload["issuedAt"]) + timedelta(minutes=1)
        now = captured_at
        subject = SUBJECT
        model_id = payload["modelId"]

        if mutation == "different_nip":
            subject = "nip-198001012005011001"
        elif mutation == "expired":
            now = datetime.fromisoformat(payload["expiresAt"]) + timedelta(seconds=1)
        elif mutation == "revoked":
            package.revoked_at = datetime.now(UTC)
            db.commit()
        elif mutation == "model":
            model_id = "different-model"
        elif mutation == "template":
            profile.template_version += 1
            db.commit()

        result = verify_offline_attendance(
            db,
            configured_app.state.container,
            subject_id=subject,
            model_id=model_id,
            face_package_id=payload["packageId"],
            captured_at=captured_at,
            liveness_method="active-challenge-v1",
            evidence_hash=hashlib.sha256(image).hexdigest(),
            image_bytes=image,
            now=now,
        )
        assert result["match"] is False
        assert result["reason_code"] == reason


def test_template_reset_increments_version_and_revokes_package(configured_app):
    with configured_app.state.container.database.session_factory() as db:
        profile, _template, _image = seed_active_template(configured_app, db)
        _response, payload, _payload_bytes, _device_key = issue_package(configured_app, db)

        updated, deleted, _cancelled = delete_templates_for_reenrollment(
            db,
            subject_id=SUBJECT,
            actor="admin-bkd",
            reason="registrasi ulang",
        )
        package = db.get(OfflineBiometricPackage, payload["packageId"])
        assert deleted == 1
        assert updated.template_version == 2
        assert profile.template_version == 2
        assert package.revoked_at is not None


def test_offline_endpoints_expose_required_authentication_and_multipart_contract(configured_app):
    schema = configured_app.openapi()
    package_operation = schema["paths"]["/mobile/v1/offline-package"]["get"]
    package_headers = {
        parameter["name"] for parameter in package_operation["parameters"] if parameter["in"] == "header"
    }
    assert {"Authorization", "X-Device-Id", "X-Device-Public-Key"} <= package_headers

    verification_operation = schema["paths"]["/internal/v1/offline-verifications"]["post"]
    verification_headers = {
        parameter["name"] for parameter in verification_operation["parameters"] if parameter["in"] == "header"
    }
    assert "X-Service-Key" in verification_headers
    request_schema = verification_operation["requestBody"]["content"]["multipart/form-data"]
    assert request_schema["schema"]["$ref"].startswith("#/components/schemas/Body_offline_verification")
