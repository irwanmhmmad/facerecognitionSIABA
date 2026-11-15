from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.crypto import TemplateCipher
from app.db import Database
from app.engines.base import FaceEngine
from app.offline_crypto import OfflinePackageSigner
from app.rate_limit import FixedWindowRateLimiter


@dataclass(slots=True)
class Container:
    settings: Settings
    database: Database
    engine: FaceEngine
    cipher: TemplateCipher
    offline_signer: OfflinePackageSigner | None
    offline_package_limiter: FixedWindowRateLimiter
    offline_verification_limiter: FixedWindowRateLimiter
