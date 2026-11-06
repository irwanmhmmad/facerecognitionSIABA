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


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()
    database = Database(resolved.database_url)
    engine = create_face_engine(resolved)
    cipher = TemplateCipher(resolved.encryption_key)
    container = Container(settings=resolved, database=database, engine=engine, cipher=cipher)

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
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "X-Service-Key"],
        )
    app.include_router(router)
    return app


app = create_app()
