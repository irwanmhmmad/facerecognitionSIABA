from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from cryptography.fernet import Fernet
from pydantic import ValidationError

from app.config import Settings
from app.engines.opencv_engine import OpenCVFaceEngine


class FakeDetector:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, value: tuple[int, int]) -> None:
        self.input_size = value

    def detect(self, _image: np.ndarray) -> tuple[int, np.ndarray | None]:
        return 1, self.faces


class FakeRecognizer:
    def alignCrop(self, image: np.ndarray, _face: np.ndarray) -> np.ndarray:
        return image

    def feature(self, _image: np.ndarray) -> np.ndarray:
        return np.array([[3.0, 4.0, 0.0]], dtype=np.float32)


def build_engine(tmp_path, monkeypatch, faces: np.ndarray | None) -> OpenCVFaceEngine:
    detector_path = tmp_path / "detector.onnx"
    recognizer_path = tmp_path / "recognizer.onnx"
    detector_path.write_bytes(b"detector-model")
    recognizer_path.write_bytes(b"recognizer-model")
    detector = FakeDetector(faces)
    recognizer = FakeRecognizer()
    image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    image = np.stack([image, np.roll(image, 5, axis=0), np.roll(image, 7, axis=1)], axis=-1)
    fake_cv2 = SimpleNamespace(
        IMREAD_COLOR=1,
        imdecode=lambda _encoded, _mode: image,
        FaceDetectorYN=SimpleNamespace(create=lambda *_args: detector),
        FaceRecognizerSF=SimpleNamespace(create=lambda *_args: recognizer),
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    return OpenCVFaceEngine(
        detector_path=detector_path,
        recognizer_path=recognizer_path,
        detector_sha256=hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        recognizer_sha256=hashlib.sha256(recognizer_path.read_bytes()).hexdigest(),
        detection_score_threshold=0.9,
        nms_threshold=0.3,
        min_face_coverage=0.08,
    )


def test_opencv_engine_returns_normalized_sface_embedding(tmp_path, monkeypatch):
    face = np.array(
        [[10, 10, 80, 80, 20, 20, 70, 20, 45, 45, 25, 70, 65, 70, 0.95]],
        dtype=np.float32,
    )
    engine = build_engine(tmp_path, monkeypatch, face)

    result = engine.analyze(b"encoded-image")

    assert result.face_count == 1
    assert result.quality_passed is True
    assert result.embedding is not None
    assert np.allclose(result.embedding, np.array([0.6, 0.8, 0.0], dtype=np.float32))
    assert result.model_id.startswith("opencv-yunet-2023mar-sface-2021dec:")


def test_opencv_engine_rejects_images_without_exactly_one_face(tmp_path, monkeypatch):
    engine = build_engine(tmp_path, monkeypatch, None)

    result = engine.analyze(b"encoded-image")

    assert result.face_count == 0
    assert result.embedding is None
    assert result.reason == "face_count_invalid"


def test_opencv_engine_rejects_unverified_model_file(tmp_path, monkeypatch):
    detector_path = tmp_path / "detector.onnx"
    recognizer_path = tmp_path / "recognizer.onnx"
    detector_path.write_bytes(b"detector-model")
    recognizer_path.write_bytes(b"recognizer-model")

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        OpenCVFaceEngine(
            detector_path=detector_path,
            recognizer_path=recognizer_path,
            detector_sha256="0" * 64,
            recognizer_sha256=hashlib.sha256(recognizer_path.read_bytes()).hexdigest(),
            detection_score_threshold=0.9,
            nms_threshold=0.3,
            min_face_coverage=0.08,
        )


def test_production_requires_explicit_opencv_model_license_confirmation():
    with pytest.raises(ValidationError, match="OPENCV_MODEL_LICENSE_CONFIRMED"):
        Settings(
            _env_file=None,
            app_env="production",
            face_engine="opencv",
            internal_service_key="production-internal-secret",
            upload_token_secret="production-upload-secret-with-enough-entropy",
            biometric_encryption_key=Fernet.generate_key().decode("ascii"),
            opencv_model_license_confirmed=False,
        )
