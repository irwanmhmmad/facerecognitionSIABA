from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib import request as urllib_request

ARTIFACTS = (
    {
        "filename": "face_detection_yunet_2023mar.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "license": "MIT",
    },
    {
        "filename": "face_recognition_sface_2021dec.onnx",
        "url": (
            "https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "license": "Apache-2.0",
    },
)

LICENSES = (
    {
        "filename": "LICENSE-YUNET.txt",
        "url": (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
            "face_detection_yunet/LICENSE"
        ),
    },
    {
        "filename": "LICENSE-SFACE.txt",
        "url": (
            "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"
            "face_recognition_sface/LICENSE"
        ),
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib_request.Request(
        url, headers={"User-Agent": "absen-face-service-model-fetcher"}
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify OpenCV face models")
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    destination = args.models_root.resolve() / "opencv"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_artifacts: list[dict[str, str]] = []

    for artifact in ARTIFACTS:
        path = destination / artifact["filename"]
        if path.exists() and sha256_file(path) != artifact["sha256"] and not args.force:
            raise SystemExit(f"Refusing to overwrite model with unexpected checksum: {path}")
        if not path.exists() or args.force:
            print(f"Downloading {artifact['filename']}...")
            download(artifact["url"], path)
        actual = sha256_file(path)
        if actual != artifact["sha256"]:
            path.unlink(missing_ok=True)
            raise SystemExit(
                f"Checksum mismatch for {artifact['filename']}: "
                f"expected {artifact['sha256']}, got {actual}"
            )
        manifest_artifacts.append({**artifact, "verified_sha256": actual})
        print(f"Verified {artifact['filename']} ({actual})")

    for license_file in LICENSES:
        path = destination / license_file["filename"]
        if not path.exists() or args.force:
            download(license_file["url"], path)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "opencv/opencv_zoo",
        "artifacts": manifest_artifacts,
        "notice": "Review model licenses and training-data provenance before production use.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"OpenCV models are ready in {destination}")


if __name__ == "__main__":
    main()
