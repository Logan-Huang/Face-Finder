"""Fetch the two ONNX models from the OpenCV model zoo.

This is the only network access the app ever makes: a one-time *download* of
public model weights. Photos are never sent anywhere.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Callable

import certifi

_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"

# name -> (filename, url, sha256)
MODELS = {
    # YuNet face detector (MIT license)
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    # SFace face recognizer (Apache 2.0 license)
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def default_model_dir() -> Path:
    env = os.environ.get("FACEFINDER_MODELS")
    return Path(env).expanduser() if env else Path.home() / ".cache" / "face-finder"


def ensure_models(model_dir: str | Path | None = None, log: Callable[[str], None] = print) -> dict[str, Path]:
    """Return {name: path} for every model, downloading any that are missing or corrupt."""
    directory = Path(model_dir) if model_dir else default_model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (filename, url, sha256) in MODELS.items():
        path = directory / filename
        if not path.exists() or _sha256(path) != sha256:
            log(f"Downloading {filename} (one-time setup)...")
            _download(url, path, sha256)
        paths[name] = path
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, sha256: str) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=context, timeout=60) as response, tmp.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
    if _sha256(tmp) != sha256:
        tmp.unlink()
        raise RuntimeError(f"Checksum mismatch for {dest.name}; refusing to use it. Try again later.")
    tmp.replace(dest)
