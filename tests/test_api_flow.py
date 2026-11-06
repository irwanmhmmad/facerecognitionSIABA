from __future__ import annotations

import asyncio
import json
from io import BytesIO

import httpx
import numpy as np
import pytest
from cryptography.fernet import Fernet
from PIL import Image
from pydantic import ValidationError
from sqlalchemy import func, select

from app.config import Settings
from app.main import create_app
from app.models import BiometricProfile, BiometricTemplate, EnrollmentSample, EnrollmentSampleImage
from app.security import require_mobile_subject, subject_from_profile_payload


class ASGITestClient:
    def __init__(self, app) -> None:
        self.app = app

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(send())

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)


def sample_image(*, inverted: bool = False, offset: int = 0) -> bytes:
    y, x = np.indices((160, 160))
    pixels = ((x * 3 + y * 5 + ((x // 12) % 2) * 80 + offset) % 256).astype(np.uint8)
    if inverted:
        pixels = 255 - pixels
    rgb = np.stack([pixels, np.roll(pixels, 7, axis=0), np.roll(pixels, 11, axis=1)], axis=-1)
    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(output, format="JPEG", quality=90)
    return output.getvalue()


@pytest.fixture()
def client(tmp_path):
    settings = Settings(
        _env_file=None,
        app_env="test",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        internal_service_key="test-internal-key",
        upload_token_secret="test-upload-secret-with-enough-entropy",
        biometric_encryption_key=Fernet.generate_key().decode("ascii"),
        face_engine="development",
        enrollment_sample_count=6,
        verification_threshold=0.55,
    )
    app = create_app(settings)
    app.state.container.database.create_all()
    return ASGITestClient(app)


def internal_headers() -> dict[str, str]:
    return {"X-Service-Key": "test-internal-key"}


def test_subject_is_derived_from_nip_not_employee_id():
    subject_id = subject_from_profile_payload(
        {"data": {"id": 759, "nip_baru": "19800101 200501 1 001"}}
    )
    assert subject_id == "nip-198001012005011001"


def test_health_and_internal_auth(client: ASGITestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["engine"] == "development"

    unauthorized = client.post(
        "/internal/v1/enrollments", json={"subject_id": "nip-198001012005011001"}
    )
    assert unauthorized.status_code == 401

    invalid_identity = client.post(
        "/internal/v1/enrollments",
        headers=internal_headers(),
        json={"subject_id": "employee-759"},
    )
    assert invalid_identity.status_code == 422


def test_delete_template_allows_face_reenrollment(client: ASGITestClient):
    subject_id = "nip-199412022025041005"
    with client.app.state.container.database.session_factory() as db:
        profile = BiometricProfile(subject_id=subject_id, status="active")
        db.add(profile)
        db.flush()
        template = BiometricTemplate(
            subject_id=subject_id,
            enrollment_id="enr_previous_registration",
            status="active",
            encrypted_embedding=b"encrypted-template",
            embedding_dimension=128,
            model_id=client.app.state.container.engine.model_id,
            sample_count=6,
        )
        db.add(template)
        db.flush()
        profile.active_template_id = template.id
        db.commit()

    unfinished = client.post(
        "/internal/v1/enrollments",
        headers={**internal_headers(), "Idempotency-Key": "unfinished-reenrollment"},
        json={"subject_id": subject_id},
    )
    assert unfinished.status_code == 201, unfinished.text

    unauthorized = client.delete(
        "/internal/v1/profiles/by-nip/199412022025041005/template",
        json={"actor": "admin-bkd", "reason": "Pegawai mengambil foto ulang"},
    )
    assert unauthorized.status_code == 401

    deleted = client.delete(
        "/internal/v1/profiles/by-nip/199412022025041005/template",
        headers=internal_headers(),
        json={"actor": "admin-bkd", "reason": "Pegawai mengambil foto ulang"},
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {
        "subject_id": subject_id,
        "status": "needs_recapture",
        "deleted_template_count": 1,
        "cancelled_enrollment_count": 1,
        "message": "Biometric templates deleted; employee may enroll again",
    }

    with client.app.state.container.database.session_factory() as db:
        profile = db.get(BiometricProfile, subject_id)
        assert profile is not None
        assert profile.status == "needs_recapture"
        assert profile.active_template_id is None
        assert (
            db.scalar(
                select(func.count())
                .select_from(BiometricTemplate)
                .where(BiometricTemplate.subject_id == subject_id)
            )
            == 0
        )

    previous = client.get(
        f"/internal/v1/enrollments/{unfinished.json()['id']}", headers=internal_headers()
    )
    assert previous.status_code == 200
    assert previous.json()["status"] == "needs_recapture"

    # Retrying the reset is a no-op and must not fail.
    retried = client.delete(
        "/internal/v1/profiles/by-nip/199412022025041005/template",
        headers=internal_headers(),
        json={"actor": "admin-bkd", "reason": "Pegawai mengambil foto ulang"},
    )
    assert retried.status_code == 200
    assert retried.json()["deleted_template_count"] == 0

    client.app.dependency_overrides[require_mobile_subject] = lambda: subject_id
    try:
        mobile_profile = client.get(
            "/mobile/v1/profile", headers={"Authorization": "Bearer login-token"}
        )
        assert mobile_profile.status_code == 200
        assert mobile_profile.json()["status"] == "needs_recapture"
        assert mobile_profile.json()["face_trained"] is False

        restarted = client.post(
            "/mobile/v1/enrollments",
            headers={
                "Authorization": "Bearer login-token",
                "Idempotency-Key": "fresh-reenrollment",
            },
            json={"struktur_org": "1,2,949,950,"},
        )
        assert restarted.status_code == 201, restarted.text
        assert restarted.json()["status"] == "collecting"
    finally:
        client.app.dependency_overrides.clear()


def test_new_idempotency_key_restarts_an_unfinished_enrollment(client: ASGITestClient):
    subject_id = "nip-198001012005011002"
    first = client.post(
        "/internal/v1/enrollments",
        headers={**internal_headers(), "Idempotency-Key": "first-attempt"},
        json={"subject_id": subject_id},
    )
    second = client.post(
        "/internal/v1/enrollments",
        headers={**internal_headers(), "Idempotency-Key": "second-attempt"},
        json={"subject_id": subject_id},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] != second.json()["id"]
    previous = client.get(
        f"/internal/v1/enrollments/{first.json()['id']}", headers=internal_headers()
    )
    assert previous.json()["status"] == "expired"


def test_full_enrollment_approval_and_verification_flow(client: ASGITestClient):
    start = client.post(
        "/internal/v1/enrollments",
        headers={**internal_headers(), "Idempotency-Key": "enrollment-198001012005011001-v1"},
        json={
            "subject_id": "nip-198001012005011001",
            "struktur_org": "1,2,949,950,",
        },
    )
    assert start.status_code == 201, start.text
    enrollment = start.json()
    upload_headers = {"Authorization": f"Bearer {enrollment['upload_token']}"}

    sample_types = (
        "frontal",
        "frontal",
        "slight_left",
        "slight_right",
        "additional",
        "additional",
    )
    uploaded_images: list[bytes] = []
    for index, sample_type in enumerate(sample_types):
        image_bytes = sample_image(offset=index * 7)
        uploaded_images.append(image_bytes)
        capture = client.post(
            f"/v1/enrollments/{enrollment['id']}/captures",
            headers=upload_headers,
            data={
                "sample_type": sample_type,
                "capture_metadata": json.dumps(
                    {
                        "device_id": "android-test-device",
                        "platform": "android",
                        "app_version": "2.3.0",
                        "camera": "front",
                        "sequence": index + 1,
                    }
                ),
            },
            files={"image": (f"{sample_type}.jpg", image_bytes, "image/jpeg")},
        )
        assert capture.status_code == 201, capture.text

    completed = client.post(
        f"/v1/enrollments/{enrollment['id']}/complete",
        headers=upload_headers,
        json={
            "liveness_passed": True,
            "liveness_method": "mlkit_blink_and_turn",
            "completed_challenges": enrollment["challenge"],
            "challenge_duration_ms": 4200,
        },
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "pending_ktu_review"
    assert "template_id" not in completed.json()
    with client.app.state.container.database.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(BiometricTemplate)) == 0
        stored_samples = list(
            db.scalars(
                select(EnrollmentSample).where(
                    EnrollmentSample.enrollment_id == enrollment["id"]
                )
            )
        )
        assert all(sample.encrypted_embedding is not None for sample in stored_samples)
        stored_images = list(
            db.scalars(
                select(EnrollmentSampleImage)
                .where(EnrollmentSampleImage.enrollment_id == enrollment["id"])
                .order_by(EnrollmentSampleImage.created_at)
            )
        )
        assert len(stored_images) == 6
        assert stored_images[0].encrypted_image != uploaded_images[0]
        assert json.loads(stored_images[0].capture_metadata)["device_id"] == "android-test-device"

    # Photos remain available during both manual review stages.

    review_samples = client.get(
        f"/internal/v1/enrollments/{enrollment['id']}/samples",
        headers=internal_headers(),
    )
    assert review_samples.status_code == 200, review_samples.text
    assert len(review_samples.json()["samples"]) == 6
    assert review_samples.json()["samples"][0]["capture_metadata"]["sequence"] == 1
    first_image = client.get(
        review_samples.json()["samples"][0]["image_url"], headers=internal_headers()
    )
    assert first_image.status_code == 200
    assert first_image.content == uploaded_images[0]
    assert first_image.headers["content-type"] == "image/jpeg"
    assert "no-store" in first_image.headers["cache-control"]

    filtered_by_org_and_nip = client.get(
        "/internal/v1/enrollments/uploaded"
        "?status=pending_ktu_review&struktur_org=1,2,949,%&nip=198001012005011001",
        headers=internal_headers(),
    )
    assert filtered_by_org_and_nip.status_code == 200
    assert filtered_by_org_and_nip.json()["total"] == 1
    assert filtered_by_org_and_nip.json()["items"][0]["id"] == enrollment["id"]

    review_by_nip = client.get(
        "/internal/v1/enrollments/by-nip/198001012005011001/samples"
        "?struktur_org=1,2,949,%",
        headers=internal_headers(),
    )
    assert review_by_nip.status_code == 200, review_by_nip.text
    assert review_by_nip.json()["enrollment_id"] == enrollment["id"]
    assert len(review_by_nip.json()["samples"]) == 6

    blocked_for_other_org = client.get(
        "/internal/v1/enrollments/by-nip/198001012005011001/samples"
        "?struktur_org=1,2,999,%",
        headers=internal_headers(),
    )
    assert blocked_for_other_org.status_code == 404

    pending = client.get(
        "/internal/v1/enrollments/uploaded?status=pending_ktu_review",
        headers=internal_headers(),
    )
    assert pending.status_code == 200
    assert pending.json()["total"] == 1

    blocked_verification = client.post(
        "/internal/v1/verifications",
        headers=internal_headers(),
        json={"subject_id": "nip-198001012005011001"},
    )
    assert blocked_verification.status_code == 409

    ktu_approved = client.post(
        f"/internal/v1/enrollments/{enrollment['id']}/ktu-approve",
        headers=internal_headers(),
        json={"actor": "ktu-opd"},
    )
    assert ktu_approved.status_code == 200, ktu_approved.text
    assert ktu_approved.json()["status"] == "pending_bkd_review"
    assert ktu_approved.json()["candidate_template_id"] is None
    review_after_ktu = client.get(
        f"/internal/v1/enrollments/{enrollment['id']}/samples",
        headers=internal_headers(),
    )
    assert review_after_ktu.status_code == 200, review_after_ktu.text
    assert len(review_after_ktu.json()["samples"]) == 6

    activated = client.post(
        f"/internal/v1/enrollments/{enrollment['id']}/bkd-approve",
        headers=internal_headers(),
        json={"actor": "=admin-bkd"},
    )
    assert activated.status_code == 200, activated.text
    assert activated.json()["status"] == "active"
    with client.app.state.container.database.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(BiometricTemplate)) == 1
        stored_samples = list(
            db.scalars(
                select(EnrollmentSample).where(
                    EnrollmentSample.enrollment_id == enrollment["id"]
                )
            )
        )
        assert all(sample.encrypted_embedding is None for sample in stored_samples)
        assert (
            db.scalar(
                select(func.count())
                .select_from(EnrollmentSampleImage)
                .where(EnrollmentSampleImage.enrollment_id == enrollment["id"])
            )
            == 0
        )

    profile = client.get(
        "/internal/v1/profiles/nip-198001012005011001", headers=internal_headers()
    )
    assert profile.status_code == 200
    assert profile.json()["status"] == "active"

    client.app.dependency_overrides[require_mobile_subject] = lambda: "nip-198001012005011001"
    try:
        mobile_profile = client.get(
            "/mobile/v1/profile", headers={"Authorization": "Bearer login-token"}
        )
        assert mobile_profile.status_code == 200
        assert mobile_profile.json()["face_trained"] is True

        mobile_verification = client.post(
            "/mobile/v1/verifications",
            headers={
                "Authorization": "Bearer login-token",
                "Idempotency-Key": "mobile-attendance-198001012005011001",
            },
            json={},
        )
        assert mobile_verification.status_code == 201, mobile_verification.text
        assert mobile_verification.json()["subject_id"] == "nip-198001012005011001"
    finally:
        client.app.dependency_overrides.clear()

    verification_start = client.post(
        "/internal/v1/verifications",
        headers={
            **internal_headers(),
            "Idempotency-Key": "attendance-198001012005011001-20261019-am",
        },
        json={"subject_id": "nip-198001012005011001"},
    )
    assert verification_start.status_code == 201, verification_start.text
    verification = verification_start.json()

    result = client.post(
        f"/v1/verifications/{verification['id']}/capture",
        headers={"Authorization": f"Bearer {verification['upload_token']}"},
        data={"liveness_passed": "true"},
        files={"image": ("attendance.jpg", sample_image(), "image/jpeg")},
    )
    assert result.status_code == 200, result.text
    assert result.json()["match"] is True
    assert result.json()["liveness_passed"] is True
    assert result.json()["score"] >= result.json()["threshold"]

    enrollment_report = client.get(
        "/internal/v1/reports/enrollments.csv?status=active",
        headers=internal_headers(),
    )
    assert enrollment_report.status_code == 200
    assert enrollment_report.headers["content-disposition"] == 'attachment; filename="enrollments.csv"'
    enrollment_csv = enrollment_report.content.decode("utf-8-sig")
    assert "enrollment_id,subject_id,status" in enrollment_csv
    assert f"{enrollment['id']},nip-198001012005011001,active" in enrollment_csv
    assert "'=admin-bkd" in enrollment_csv
    assert "upload_token" not in enrollment_csv
    assert "challenge" not in enrollment_csv

    verification_report = client.get(
        "/internal/v1/reports/verifications.csv"
        "?status=matched&subject_id=nip-198001012005011001",
        headers=internal_headers(),
    )
    assert verification_report.status_code == 200
    assert verification_report.headers["content-disposition"] == (
        'attachment; filename="verifications.csv"'
    )
    verification_csv = verification_report.content.decode("utf-8-sig")
    assert "verification_id,subject_id,status" in verification_csv
    assert f"{verification['id']},nip-198001012005011001,matched" in verification_csv


def test_reports_require_internal_auth_and_validate_date_range(client: ASGITestClient):
    unauthorized = client.get("/internal/v1/reports/enrollments.csv")
    assert unauthorized.status_code == 401

    invalid_range = client.get(
        "/internal/v1/reports/verifications.csv"
        "?created_from=2026-10-20T00:00:00Z&created_to=2026-10-19T00:00:00Z",
        headers=internal_headers(),
    )
    assert invalid_range.status_code == 422


def test_rejects_non_image_and_failed_liveness(client: ASGITestClient):
    start = client.post(
        "/internal/v1/enrollments",
        headers=internal_headers(),
        json={"subject_id": "nip-197001012000011001"},
    ).json()
    auth = {"Authorization": f"Bearer {start['upload_token']}"}

    bad = client.post(
        f"/v1/enrollments/{start['id']}/captures",
        headers=auth,
        data={"sample_type": "frontal"},
        files={"image": ("face.txt", b"not-an-image", "text/plain")},
    )
    assert bad.status_code == 415

    sample_types = (
        "frontal",
        "slight_left",
        "slight_right",
        "additional",
        "additional",
        "additional",
    )
    for index, sample_type in enumerate(sample_types):
        response = client.post(
            f"/v1/enrollments/{start['id']}/captures",
            headers=auth,
            data={"sample_type": sample_type},
            files={"image": ("face.jpg", sample_image(offset=index * 7), "image/jpeg")},
        )
        assert response.status_code == 201

    failed = client.post(
        f"/v1/enrollments/{start['id']}/complete",
        headers=auth,
        json={
            "liveness_passed": False,
            "liveness_method": "mlkit_blink_and_turn",
            "completed_challenges": start["challenge"],
            "challenge_duration_ms": 3500,
        },
    )
    assert failed.status_code == 422


def test_mobile_enrollment_uses_authenticated_subject(client: ASGITestClient):
    client.app.dependency_overrides[require_mobile_subject] = lambda: "nip-198001012005011001"
    try:
        profile = client.get("/mobile/v1/profile", headers={"Authorization": "Bearer login-token"})
        assert profile.status_code == 200
        assert profile.json()["status"] == "not_registered"
        assert profile.json()["face_trained"] is False

        blocked_verification = client.post(
            "/mobile/v1/verifications",
            headers={"Authorization": "Bearer login-token"},
            json={},
        )
        assert blocked_verification.status_code == 409

        start = client.post(
            "/mobile/v1/enrollments",
            headers={"Authorization": "Bearer login-token", "Idempotency-Key": "mobile-759-v1"},
            json={"struktur_org": "1,2,949,950,%"},
        )
        assert start.status_code == 201, start.text
        assert start.json()["subject_id"] == "nip-198001012005011001"
        assert set(start.json()["challenge"]) == {"blink", "turn-left", "turn-right"}

        stored = client.get(
            f"/internal/v1/enrollments/{start.json()['id']}", headers=internal_headers()
        )
        assert stored.status_code == 200
        assert stored.json()["struktur_org"] == "1,2,949,950,"

        matching = client.get(
            "/internal/v1/enrollments?struktur_org=1,2,949,%", headers=internal_headers()
        )
        assert matching.status_code == 200
        assert matching.json()["total"] == 1

        excluded = client.get(
            "/internal/v1/enrollments?struktur_org=1,2,999,%", headers=internal_headers()
        )
        assert excluded.status_code == 200
        assert excluded.json()["total"] == 0
    finally:
        client.app.dependency_overrides.clear()


def test_mobile_requires_reenrollment_when_active_model_changes(client: ASGITestClient):
    subject_id = "nip-199001012010011001"
    container = client.app.state.container
    template = BiometricTemplate(
        subject_id=subject_id,
        enrollment_id="legacy-enrollment",
        status="active",
        encrypted_embedding=container.cipher.encrypt_embedding(
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ),
        embedding_dimension=3,
        model_id="legacy-face-model",
        sample_count=6,
    )
    with container.database.session_factory() as db:
        profile = BiometricProfile(subject_id=subject_id, status="active")
        db.add_all([profile, template])
        db.flush()
        profile.active_template_id = template.id
        db.commit()

    client.app.dependency_overrides[require_mobile_subject] = lambda: subject_id
    try:
        profile_response = client.get(
            "/mobile/v1/profile", headers={"Authorization": "Bearer login-token"}
        )
        assert profile_response.status_code == 200
        assert profile_response.json()["face_trained"] is False

        verification = client.post(
            "/mobile/v1/verifications",
            headers={"Authorization": "Bearer login-token"},
            json={},
        )
        assert verification.status_code == 409
        assert "reenrollment" in verification.json()["detail"]
    finally:
        client.app.dependency_overrides.clear()


def test_production_rejects_demo_configuration():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_env="production")
