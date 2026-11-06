from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

import numpy as np

from app.engines.base import FaceAnalysisResult, normalize_embedding


class OpenCVFaceEngine:
    """YuNet detector and SFace recognizer executed by OpenCV DNN."""

    def __init__(
        self,
        *,
        detector_path: Path,
        recognizer_path: Path,
        detector_sha256: str,
        recognizer_sha256: str,
        detection_score_threshold: float,
        nms_threshold: float,
        min_face_coverage: float,
    ) -> None:
        self._detector_path = detector_path.resolve()
        self._recognizer_path = recognizer_path.resolve()
        detector_digest = self._verify_model(self._detector_path, detector_sha256)
        recognizer_digest = self._verify_model(self._recognizer_path, recognizer_sha256)

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV inference is not installed. Install project dependencies with: pip install ."
            ) from exc

        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("Installed OpenCV does not provide FaceDetectorYN and FaceRecognizerSF")

        self._cv2 = cv2
        self._detection_score_threshold = detection_score_threshold
        self._min_face_coverage = min_face_coverage
        self._detector = cv2.FaceDetectorYN.create(
            str(self._detector_path),
            "",
            (320, 320),
            detection_score_threshold,
            nms_threshold,
            5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(self._recognizer_path), "")
        self._model_id = (
            f"opencv-yunet-2023mar-sface-2021dec:"
            f"{detector_digest[:16]}:{recognizer_digest[:16]}"
        )
        # FaceDetectorYN.setInputSize mutates shared native state.
        self._inference_lock = Lock()

    @staticmethod
    def _verify_model(path: Path, expected_sha256: str) -> str:
        if not path.is_file():
            raise RuntimeError(f"OpenCV model file was not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"OpenCV model checksum mismatch for {path.name}: "
                f"expected {expected_sha256.lower()}, got {actual}"
            )
        return actual

    @property
    def model_id(self) -> str:
        return self._model_id

    def analyze(self, image_bytes: bytes) -> FaceAnalysisResult:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = self._cv2.imdecode(encoded, self._cv2.IMREAD_COLOR)
        if image is None:
            return FaceAnalysisResult(0, None, self.model_id, False, {}, "invalid_image")

        height, width = image.shape[:2]
        with self._inference_lock:
            self._detector.setInputSize((width, height))
            _, detected = self._detector.detect(image)
            faces = [] if detected is None else list(detected)
            if len(faces) != 1:
                return FaceAnalysisResult(
                    len(faces),
                    None,
                    self.model_id,
                    False,
                    {"face_count": len(faces), "width": width, "height": height},
                    "face_count_invalid",
                )

            face = np.asarray(faces[0], dtype=np.float32)
            aligned = self._recognizer.alignCrop(image, face)
            raw_embedding = self._recognizer.feature(aligned)

        x, y, face_width, face_height = (float(value) for value in face[:4])
        face_area = max(0.0, face_width) * max(0.0, face_height)
        coverage = face_area / max(1.0, float(width * height))
        detection_score = float(face[-1])
        pixels = np.asarray(image, dtype=np.float32)
        brightness = float(pixels.mean())
        contrast = float(pixels.std())
        quality_passed = (
            detection_score >= self._detection_score_threshold
            and coverage >= self._min_face_coverage
            and 15 <= brightness <= 240
            and contrast >= 5
        )
        quality = {
            "width": width,
            "height": height,
            "detection_score": round(detection_score, 6),
            "face_coverage": round(coverage, 6),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "face_x": round(x, 3),
            "face_y": round(y, 3),
        }
        if raw_embedding is None:
            return FaceAnalysisResult(
                1, None, self.model_id, False, quality, "embedding_unavailable"
            )
        try:
            embedding = normalize_embedding(np.asarray(raw_embedding, dtype=np.float32).reshape(-1))
        except ValueError:
            return FaceAnalysisResult(
                1, None, self.model_id, False, quality, "embedding_unavailable"
            )

        return FaceAnalysisResult(
            face_count=1,
            embedding=embedding,
            model_id=self.model_id,
            quality_passed=quality_passed,
            quality=quality,
            reason=None if quality_passed else "image_quality_failed",
        )
