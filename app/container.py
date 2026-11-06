from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.crypto import TemplateCipher
from app.db import Database
from app.engines.base import FaceEngine


@dataclass(slots=True)
class Container:
    settings: Settings
    database: Database
    engine: FaceEngine
    cipher: TemplateCipher
