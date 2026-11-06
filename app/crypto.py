from __future__ import annotations

import numpy as np
from cryptography.fernet import Fernet, InvalidToken


class TemplateCipher:
    def __init__(self, key: str) -> None:
        self._fernet = Fernet(key.encode("ascii"))

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        normalized = np.asarray(embedding, dtype=np.float32)
        return self.encrypt_bytes(normalized.tobytes())

    def encrypt_bytes(self, payload: bytes) -> bytes:
        return self._fernet.encrypt(payload)

    def decrypt_bytes(self, payload: bytes) -> bytes:
        try:
            return self._fernet.decrypt(payload)
        except InvalidToken as exc:
            raise ValueError("Encrypted biometric payload cannot be decrypted") from exc

    def decrypt_embedding(self, payload: bytes, dimension: int) -> np.ndarray:
        raw = self.decrypt_bytes(payload)
        embedding = np.frombuffer(raw, dtype=np.float32)
        if embedding.size != dimension:
            raise ValueError("Biometric template dimension is invalid")
        return embedding.copy()
