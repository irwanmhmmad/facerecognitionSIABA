from __future__ import annotations

from app.config import Settings
from app.engines.base import FaceEngine
from app.engines.development import DevelopmentFaceEngine
from app.engines.insightface_engine import InsightFaceEngine
from app.engines.opencv_engine import OpenCVFaceEngine


def create_face_engine(settings: Settings) -> FaceEngine:
    if settings.face_engine == "development":
        return DevelopmentFaceEngine()
    if settings.face_engine == "insightface":
        return InsightFaceEngine(
            models_root=settings.face_models_root,
            model_name=settings.insightface_model_name,
            providers=settings.providers,
        )
    if settings.face_engine == "opencv":
        models_dir = settings.opencv_models_dir
        return OpenCVFaceEngine(
            detector_path=models_dir / settings.opencv_detector_filename,
            recognizer_path=models_dir / settings.opencv_recognizer_filename,
            detector_sha256=settings.opencv_detector_sha256,
            recognizer_sha256=settings.opencv_recognizer_sha256,
            detection_score_threshold=settings.opencv_detection_score_threshold,
            nms_threshold=settings.opencv_nms_threshold,
            min_face_coverage=settings.opencv_min_face_coverage,
        )
    raise ValueError(f"Unsupported FACE_ENGINE: {settings.face_engine}")
