from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import Settings
from app.container import Container
from app.crypto import TemplateCipher
from app.db import Database
from app.engines.factory import create_face_engine
from app.offline_crypto import OfflinePackageSigner
from app.rate_limit import FixedWindowRateLimiter


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    database = Database(resolved.database_url)
    engine = create_face_engine(resolved)
    cipher = TemplateCipher(resolved.encryption_key)
    offline_signer = (
        OfflinePackageSigner(
            resolved.offline_signing_private_key_b64,
            resolved.offline_signing_key_id,
        )
        if resolved.offline_signing_private_key_b64
        else None
    )
    container = Container(
        settings=resolved,
        database=database,
        engine=engine,
        cipher=cipher,
        offline_signer=offline_signer,
        offline_package_limiter=FixedWindowRateLimiter(resolved.offline_package_rate_limit_per_minute),
        offline_verification_limiter=FixedWindowRateLimiter(
            resolved.offline_verification_rate_limit_per_minute
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        database.create_all()
        yield

    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        description=(
            "Internal biometric enrollment and 1:1 verification service. "
            "The development engine is never suitable for production."
        ),
        lifespan=lifespan,
    )
    app.state.container = container
    if resolved.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=[
                "Authorization",
                "Content-Type",
                "Idempotency-Key",
                "X-Service-Key",
                "X-Device-Id",
                "X-Device-Public-Key",
            ],
        )
    app.include_router(router)
    return app


app = create_app()
