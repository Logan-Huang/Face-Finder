"""Face detection, embedding and matching, all on-device via OpenCV.

Pipeline per photo:
  1. decode (JPEG/PNG/HEIC/...) and apply the EXIF rotation
  2. YuNet finds every face and its 5 landmarks (on a downscaled copy, for speed)
  3. each face is aligned from the full-resolution photo and SFace turns it into a
     128-number "faceprint" (embedding)
  4. a face matches the person when its cosine similarity to any reference
     faceprint is at or above the threshold
"""

from __future__ import annotations

import base64
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import ensure_models

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

# OpenCV 5 warns that its new graph engine ignores compute targets; that's harmless noise.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Cosine-similarity cut-off recommended for SFace by OpenCV (tuned on the LFW benchmark).
MATCH_THRESHOLD = 0.363

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}

# Faces are detected on a copy no larger than this; 1600px keeps small faces in group shots.
DETECT_MAX_SIDE = 1600

# Phone cameras now reach 200 MP; these are the user's own files, so allow them.
Image.MAX_IMAGE_PIXELS = 400_000_000


@dataclass
class Face:
    box: tuple[float, float, float, float]  # x, y, width, height in pixels
    confidence: float  # detector confidence that this is a face
    embedding: np.ndarray  # unit-length faceprint

    @property
    def area(self) -> float:
        return self.box[2] * self.box[3]


class PhotoError(Exception):
    """A photo that couldn't be read or analysed."""


def load_image(source: str | Path | BinaryIO) -> np.ndarray:
    """Decode a photo into an upright BGR array. Raises PhotoError if it isn't an image."""
    try:
        with Image.open(source) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise PhotoError(f"not a readable image ({exc.__class__.__name__})") from exc


class FaceEngine:
    def __init__(self, model_dir: str | Path | None = None, min_confidence: float = 0.8):
        paths = ensure_models(model_dir)
        self._detector = cv2.FaceDetectorYN.create(
            str(paths["detector"]), "", (320, 320), min_confidence, 0.3, 5000
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(paths["recognizer"]), "")
        # OpenCV networks hold per-call state, so one photo at a time per engine.
        self._lock = threading.Lock()

    def faces(self, img: np.ndarray) -> list[Face]:
        """Find every face in a BGR image and compute its faceprint."""
        try:
            return self._faces(img)
        except cv2.error as exc:
            raise PhotoError("face detection failed on this image") from exc

    def _faces(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
        small = img if scale == 1.0 else cv2.resize(
            img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
        with self._lock:
            self._detector.setInputSize((small.shape[1], small.shape[0]))
            _, rows = self._detector.detect(small)
            if rows is None:
                return []
            found = []
            for row in rows:
                row = row.copy()
                row[:14] /= scale  # box + landmarks back to full-resolution pixels
                aligned = self._recognizer.alignCrop(img, row)
                embedding = self._recognizer.feature(aligned).flatten().astype(np.float32)
                embedding /= np.linalg.norm(embedding) or 1.0
                x, y, bw, bh = (float(v) for v in row[:4])
                found.append(Face((x, y, bw, bh), float(row[14]), embedding))
            return found


def similarity(faces: list[Face], references: np.ndarray) -> list[float]:
    """For each face, its best cosine similarity to any of the reference faceprints."""
    if not faces or len(references) == 0:
        return [0.0] * len(faces)
    sims = np.stack([f.embedding for f in faces]) @ np.asarray(references, np.float32).T
    return [float(s) for s in sims.max(axis=1)]


def reference_face(faces: list[Face]) -> Face | None:
    """The face to learn from in a reference photo: the largest one."""
    return max(faces, key=lambda f: f.area, default=None)


def thumbnail_data_url(img: np.ndarray, max_side: int = 480) -> str:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        raise ValueError("could not encode thumbnail")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


@dataclass(frozen=True)
class PhotoRef:
    """A photo on disk, or one member of a .zip (read in memory, never extracted)."""

    path: Path  # the file, or the .zip holding it
    member: str | None = None

    @property
    def label(self) -> str:
        return str(self.path) if self.member is None else f"{self.path}/{self.member}"


def zip_member_is_photo(name: str) -> bool:
    parts = name.split("/")
    if name.endswith("/") or parts[0] == "__MACOSX" or any(p.startswith(".") for p in parts):
        return False
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def list_photos(paths: list[Path], recursive: bool = False, on_bad_zip=None) -> list[PhotoRef]:
    """Photos from files, folders and .zip archives (also those found inside folders).

    `recursive` only controls folders; inside a zip every folder is searched.
    `on_bad_zip(path, exc)` is called for archives that can't be opened.
    """
    photos: list[PhotoRef] = []

    def add_zip(path: Path) -> None:
        try:
            with zipfile.ZipFile(path) as zf:
                photos.extend(PhotoRef(path, i.filename) for i in zf.infolist() if zip_member_is_photo(i.filename))
        except Exception as exc:
            if on_bad_zip:
                on_bad_zip(path, exc)

    for path in paths:
        path = path.expanduser()
        if path.is_dir():
            children = path.rglob("*") if recursive else path.iterdir()
            for child in sorted(children):
                hidden = any(part.startswith(".") for part in child.relative_to(path).parts)
                if not child.is_file() or hidden:
                    continue
                if child.suffix.lower() == ".zip":
                    add_zip(child)
                elif child.suffix.lower() in IMAGE_EXTENSIONS:
                    photos.append(PhotoRef(child))
        elif path.is_file():
            if path.suffix.lower() == ".zip":
                add_zip(path)
            else:
                photos.append(PhotoRef(path))
    return photos
