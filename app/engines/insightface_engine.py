from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from app.engines.base import FaceAnalysisResult, normalize_embedding


class InsightFaceEngine:
    """Adapter for operator-supplied, production-licensed InsightFace-compatible ONNX models."""

    def __init__(self, models_root: str, model_name: str, providers: list[str]) -> None:
        model_dir = Path(models_root).resolve() / "models" / model_name
        model_files = sorted(model_dir.glob("*.onnx"))
        if not model_files:
            raise RuntimeError(
                f"No ONNX models found in {model_dir}. Automatic model downloads are disabled."
            )

        try:
            import cv2
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Install the inference dependencies with: pip install '.[inference]'") from exc

        self._cv2 = cv2
        self._model_id = self._build_model_id(model_name, model_files)
        self._app = FaceAnalysis(name=model_name, root=str(Path(models_root).resolve()), providers=providers)
        self._app.prepare(ctx_id=0, det_size=(640, 640))

    @staticmethod
    def _build_model_id(model_name: str, files: list[Path]) -> str:
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.name.encode("utf-8"))
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        return f"{model_name}:{digest.hexdigest()[:16]}"

    @property
    def model_id(self) -> str:
        return self._model_id

    def analyze(self, image_bytes: bytes) -> FaceAnalysisResult:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = self._cv2.imdecode(encoded, self._cv2.IMREAD_COLOR)
        if image is None:
            return FaceAnalysisResult(0, None, self.model_id, False, {}, "invalid_image")

        faces = self._app.get(image)
        if len(faces) != 1:
            return FaceAnalysisResult(
                len(faces), None, self.model_id, False, {"face_count": len(faces)}, "face_count_invalid"
            )

        face = faces[0]
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                return FaceAnalysisResult(1, None, self.model_id, False, {}, "embedding_unavailable")
            embedding = normalize_embedding(raw_embedding)

        height, width = image.shape[:2]
        bbox = np.asarray(face.bbox, dtype=np.float32)
        face_area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        coverage = face_area / max(1.0, float(width * height))
        detection_score = float(getattr(face, "det_score", 0.0))
        quality_passed = detection_score >= 0.65 and coverage >= 0.08
        quality = {
            "width": width,
            "height": height,
            "detection_score": round(detection_score, 6),
            "face_coverage": round(coverage, 6),
        }
        return FaceAnalysisResult(
            face_count=1,
            embedding=normalize_embedding(embedding),
            model_id=self.model_id,
            quality_passed=quality_passed,
            quality=quality,
            reason=None if quality_passed else "image_quality_failed",
        )
