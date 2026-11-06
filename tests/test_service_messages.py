from sqlalchemy import func, select

from app.db import Database
from app.models import BiometricProfile, BiometricTemplate, EnrollmentSession
from app.services import _face_count_error_detail, delete_templates_for_reenrollment


def test_face_count_message_explains_when_face_is_not_detected():
    message = _face_count_error_detail(0)

    assert "tidak terdeteksi" in message
    assert "cahaya" in message


def test_face_count_message_explains_when_multiple_faces_are_detected():
    message = _face_count_error_detail(2)

    assert "2 wajah" in message
    assert "hanya satu orang" in message


def test_delete_templates_for_reenrollment_removes_embedding_and_closes_open_session(tmp_path):
    database = Database(f"sqlite:///{tmp_path / 'template-reset.db'}")
    database.create_all()
    subject_id = "nip-199412022025041005"

    with database.session_factory() as db:
        profile = BiometricProfile(subject_id=subject_id, status="active")
        db.add(profile)
        db.flush()
        enrollment = EnrollmentSession(
            subject_id=subject_id,
            status="collecting",
            challenge="[]",
            expires_at=profile.created_at,
        )
        template = BiometricTemplate(
            subject_id=subject_id,
            enrollment_id="enr_previous_registration",
            status="active",
            encrypted_embedding=b"encrypted-template",
            embedding_dimension=128,
            model_id="test-model",
            sample_count=6,
        )
        db.add_all([enrollment, template])
        db.flush()
        profile.active_template_id = template.id
        db.commit()

        result, deleted_count, cancelled_count = delete_templates_for_reenrollment(
            db,
            subject_id=subject_id,
            actor="admin-bkd",
            reason="Pegawai mengambil foto ulang",
        )

        assert result.status == "needs_recapture"
        assert result.active_template_id is None
        assert deleted_count == 1
        assert cancelled_count == 1
        assert enrollment.status == "needs_recapture"
        assert (
            db.scalar(
                select(func.count())
                .select_from(BiometricTemplate)
                .where(BiometricTemplate.subject_id == subject_id)
            )
            == 0
        )
