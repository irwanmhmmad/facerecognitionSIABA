from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _development_fernet_key() -> str:
    digest = hashlib.sha256(b"absen-face-service-development-only").digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name: str = "ABSEN Face Recognition Service"
    app_env: str = "development"
    database_url: str = "sqlite:///./face_service.db"

    internal_service_key: str = "change-this-in-production"
    upload_token_secret: str = "change-this-in-production-too"
    biometric_encryption_key: str = ""
    token_issuer: str = "absen-face-service"

    upload_token_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    enrollment_ttl_seconds: int = Field(default=1800, ge=300, le=86400)
    verification_ttl_seconds: int = Field(default=300, ge=60, le=1800)
    enrollment_sample_count: int = Field(default=6, ge=3, le=8)
    enrollment_consistency_threshold: float = Field(default=0.35, ge=-1, le=1)
    verification_threshold: float = Field(default=0.55, ge=-1, le=1)
    max_upload_bytes: int = Field(default=2_000_000, ge=50_000, le=10_000_000)

    face_engine: str = "development"
    face_models_root: str = "./models"
    opencv_detector_filename: str = "face_detection_yunet_2023mar.onnx"
    opencv_recognizer_filename: str = "face_recognition_sface_2021dec.onnx"
    opencv_detector_sha256: str = Field(
        default="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    opencv_recognizer_sha256: str = Field(
        default="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        pattern=r"^[0-9a-fA-F]{64}$",
    )
    opencv_detection_score_threshold: float = Field(default=0.75, ge=0, le=1)
    opencv_nms_threshold: float = Field(default=0.3, ge=0, le=1)
    opencv_min_face_coverage: float = Field(default=0.08, ge=0.01, le=1)
    opencv_model_license_confirmed: bool = False
    insightface_model_name: str = "licensed_model"
    face_model_license_confirmed: bool = False
    onnx_providers: str = "CPUExecutionProvider"
    cors_origins: str = ""
    absenweb_base_url: str = "http://127.0.0.1:8000"
    absenweb_profile_path: str = "/api/get-my-profile"
    mobile_auth_timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    offline_signing_private_key_b64: str = ""
    offline_signing_key_id: str = Field(default="face-offline-2026-01", min_length=3, max_length=64)
    offline_package_ttl_seconds: int = Field(default=604_800, ge=300, le=604_800)
    offline_device_public_key_max_header_bytes: int = Field(default=4096, ge=512, le=16384)
    offline_package_rate_limit_per_minute: int = Field(default=10, ge=1, le=600)
    offline_verification_rate_limit_per_minute: int = Field(default=60, ge=1, le=6000)

    @property
    def encryption_key(self) -> str:
        return self.biometric_encryption_key or _development_fernet_key()

    @property
    def providers(self) -> list[str]:
        return [item.strip() for item in self.onnx_providers.split(",") if item.strip()]

    @property
    def opencv_models_dir(self) -> Path:
        return Path(self.face_models_root).resolve() / "opencv"

    @property
    def allowed_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @model_validator(mode="after")
    def validate_production_settings(self) -> Settings:
        if self.app_env.lower() == "production":
            unsafe = {
                "INTERNAL_SERVICE_KEY": self.internal_service_key == "change-this-in-production",
                "UPLOAD_TOKEN_SECRET": self.upload_token_secret == "change-this-in-production-too",
                "BIOMETRIC_ENCRYPTION_KEY": not self.biometric_encryption_key,
                "FACE_ENGINE": self.face_engine == "development",
            }
            invalid = [name for name, failed in unsafe.items() if failed]
            if invalid:
                raise ValueError(f"Unsafe production configuration: {', '.join(invalid)}")

        if self.face_engine == "insightface" and not self.face_model_license_confirmed:
            raise ValueError("FACE_MODEL_LICENSE_CONFIRMED=true is required for the InsightFace adapter")
        if (
            self.app_env.lower() == "production"
            and self.face_engine == "opencv"
            and not self.opencv_model_license_confirmed
        ):
            raise ValueError("OPENCV_MODEL_LICENSE_CONFIRMED=true is required for the OpenCV adapter")
        if (
            self.offline_signing_private_key_b64
            and self.offline_signing_private_key_b64 == self.biometric_encryption_key
        ):
            raise ValueError("Offline signing key must differ from BIOMETRIC_ENCRYPTION_KEY")
        return self
