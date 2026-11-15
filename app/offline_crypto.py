from __future__ import annotations

import base64
import hashlib
import json
import secrets
from dataclasses import dataclass

import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.engines.base import normalize_embedding


def deterministic_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_device_public_key(value: str, *, max_header_bytes: int) -> tuple[rsa.RSAPublicKey, str]:
    if len(value.encode("ascii", errors="ignore")) > max_header_bytes:
        raise ValueError("Device public key header is too large")
    try:
        der = base64.b64decode(value, validate=True)
        key = serialization.load_der_public_key(der)
    except (ValueError, TypeError) as exc:
        raise ValueError("Device public key is not valid Base64 X.509 DER") from exc
    if not isinstance(key, rsa.RSAPublicKey):
        raise ValueError("Device public key must be RSA")
    if key.key_size < 2048:
        raise ValueError("Device RSA public key must be at least 2048 bits")
    fingerprint = hashlib.sha256(der).hexdigest()
    return key, fingerprint


@dataclass(frozen=True, slots=True)
class EncryptedDeviceEmbedding:
    wrapped_key: str
    iv: str
    ciphertext: str


class OfflinePackageSigner:
    def __init__(self, private_key_b64: str, key_id: str) -> None:
        try:
            encoded = base64.b64decode(private_key_b64, validate=True)
        except ValueError as exc:
            raise ValueError("OFFLINE_SIGNING_PRIVATE_KEY_B64 is not valid Base64") from exc
        try:
            key = serialization.load_pem_private_key(encoded, password=None)
        except (ValueError, TypeError):
            try:
                key = serialization.load_der_private_key(encoded, password=None)
            except (ValueError, TypeError) as exc:
                raise ValueError("OFFLINE_SIGNING_PRIVATE_KEY_B64 is not a valid RSA key") from exc
        if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
            raise ValueError("Offline signing key must be an RSA private key of at least 2048 bits")
        self._key = key
        self.key_id = key_id

    @property
    def public_key_x509_der_b64(self) -> str:
        der = self._key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return base64.b64encode(der).decode("ascii")

    def encrypt_for_device(
        self, embedding: np.ndarray, device_key: rsa.RSAPublicKey
    ) -> EncryptedDeviceEmbedding:
        normalized = normalize_embedding(embedding)
        if normalized.ndim != 1 or not np.all(np.isfinite(normalized)):
            raise ValueError("Embedding must be a finite one-dimensional vector")
        plaintext = deterministic_json_bytes([float(item) for item in normalized.tolist()])
        aes_key = secrets.token_bytes(32)
        iv = secrets.token_bytes(12)
        ciphertext = AESGCM(aes_key).encrypt(iv, plaintext, None)
        wrapped_key = device_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA1()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return EncryptedDeviceEmbedding(
            wrapped_key=base64.b64encode(wrapped_key).decode("ascii"),
            iv=base64.b64encode(iv).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        )

    def sign(self, payload: bytes) -> str:
        signature = self._key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        return base64.b64encode(signature).decode("ascii")
