from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.engines.base import FaceAnalysisResult, normalize_embedding


class DevelopmentFaceEngine:
    """Deterministic test adapter. It is deliberately not a production face recognizer."""

    @property
    def model_id(self) -> str:
        return "development-pixel-embedding-v1"

    def analyze(self, image_bytes: bytes) -> FaceAnalysisResult:
        try:
            image = Image.open(BytesIO(image_bytes))
            image.verify()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            return FaceAnalysisResult(
                face_count=0,
                embedding=None,
                model_id=self.model_id,
                quality_passed=False,
                quality={},
                reason=f"invalid_image:{type(exc).__name__}",
            )

        width, height = image.size
        grayscale = ImageOps.grayscale(image)
        pixels = np.asarray(grayscale, dtype=np.float32)
        brightness = float(pixels.mean())
        contrast = float(pixels.std())
        min_side = min(width, height)
        quality_passed = min_side >= 64 and 15 <= brightness <= 240 and contrast >= 5

        reduced = np.asarray(grayscale.resize((24, 24)), dtype=np.float32).reshape(-1)
        reduced = reduced - reduced.mean()
        try:
            embedding = normalize_embedding(reduced)
        except ValueError:
            embedding = None
            quality_passed = False

        quality = {
            "width": width,
            "height": height,
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "development_engine": True,
        }
        return FaceAnalysisResult(
            face_count=1 if embedding is not None else 0,
            embedding=embedding,
            model_id=self.model_id,
            quality_passed=quality_passed,
            quality=quality,
            reason=None if quality_passed else "image_quality_failed",
        )
