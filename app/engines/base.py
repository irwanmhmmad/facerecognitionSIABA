from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class FaceAnalysisResult:
    face_count: int
    embedding: np.ndarray | None
    model_id: str
    quality_passed: bool
    quality: dict[str, float | int | str | bool]
    reason: str | None = None


class FaceEngine(Protocol):
    @property
    def model_id(self) -> str: ...

    def analyze(self, image_bytes: bytes) -> FaceAnalysisResult: ...


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    value = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Embedding has zero norm")
    return value / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Embedding dimensions do not match")
    return float(np.dot(normalize_embedding(left), normalize_embedding(right)))
