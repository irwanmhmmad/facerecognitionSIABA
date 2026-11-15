from __future__ import annotations

import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa


def main() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    print(f"OFFLINE_SIGNING_PRIVATE_KEY_B64={base64.b64encode(private_pem).decode('ascii')}")
    print(f"EXPO_PUBLIC_FACE_OFFLINE_PUBLIC_KEY={base64.b64encode(public_der).decode('ascii')}")


if __name__ == "__main__":
    main()
