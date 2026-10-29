"""Local web UI. The server listens on 127.0.0.1 only, so photos never leave this computer."""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import threading
import webbrowser
import zipfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
from waitress import create_server

from .engine import (
    MATCH_THRESHOLD, Face, FaceEngine, PhotoError, load_image, similarity, thumbnail_data_url,
)

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}
MAX_REFERENCE_FACES = 12  # a crowd shot offers its 12 largest faces to choose from


def create_app(engine: FaceEngine) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = None  # local only; folders of photos can be large

    @app.before_request
    def only_this_computer():
        # Blocks DNS-rebinding tricks where a website poses as 127.0.0.1 to reach this app.
        if request.host.rsplit(":", 1)[0] not in LOCAL_HOSTS:
            abort(403)

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/config")
    def config():
        return jsonify(threshold=MATCH_THRESHOLD)

    @app.post("/api/reference")
    def reference():
        """Every face in each example photo, largest first, so the page can let you pick the right one."""
        refs, errors = [], []
        for upload in request.files.getlist("files"):
            try:
                img = load_image(upload.stream)
                faces = engine.faces(img)
            except PhotoError as exc:
                errors.append(f"{upload.filename}: {exc}")
                continue
            if not faces:
                errors.append(f"{upload.filename}: no face found")
                continue
            faces = sorted(faces, key=lambda f: f.area, reverse=True)[:MAX_REFERENCE_FACES]
            refs.append({
                "name": upload.filename,
                "thumb": thumbnail_data_url(img),
                "aspect": _aspect(img),
                "faces": [
                    {"box": _box(f, img), "embedding": f.embedding.tolist(), "crop": _face_crop(img, f)} for f in faces
                ],
            })
        return jsonify(refs=refs, errors=errors)

    @app.post("/api/scan")
    def scan():
        """Score every face in one photo against each person's example faceprints."""
        upload = request.files["file"]
        people = [np.asarray(refs, np.float32) for refs in json.loads(request.form["people"])]
        try:
            img = load_image(upload.stream)
            faces = engine.faces(img)
        except PhotoError as exc:
            return jsonify(name=upload.filename, error=str(exc))
        per_person = [similarity(faces, refs) for refs in people]
        # No thumbnail here: with tens of thousands of photos the page would run out of memory.
        # It asks /api/thumb for previews of just the photos on screen instead.
        return jsonify(
            name=upload.filename,
            aspect=_aspect(img),
            faces=[
                {"box": _box(f, img), "scores": [round(scores[i], 4) for scores in per_person]}
                for i, f in enumerate(faces)
            ],
        )

    @app.post("/api/thumb")
    def thumb():
        """A small upright JPEG preview of one photo."""
        try:
            data = _thumbnail_jpeg(request.files["file"].stream)
        except Exception:
            return jsonify(error="not a readable image"), 422
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/zip")
    def make_zip():
        """Bundle the photos the browser sends back (the matches) into one download, a folder per person."""
        out = tempfile.SpooledTemporaryFile(max_size=64 << 20)
        used: set[str] = set()
        folders = request.form.getlist("folders")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for i, upload in enumerate(request.files.getlist("files")):
                folder = _safe_folder(folders[i]) if i < len(folders) else ""
                name = _unique_name(Path(upload.filename or "photo").name, used, folder)
                with archive.open(name, "w", force_zip64=True) as dest:
                    shutil.copyfileobj(upload.stream, dest)
        out.seek(0)
        return send_file(out, mimetype="application/zip", as_attachment=True, download_name="matches.zip")

    return app


def _box(face: Face, img: np.ndarray) -> list[float]:
    """Face box as fractions of the image size, so the page can draw it over any thumbnail size."""
    h, w = img.shape[:2]
    x, y, bw, bh = face.box
    return [x / w, y / h, bw / w, bh / h]


def _thumbnail_jpeg(stream, max_side: int = 480) -> bytes:
    with Image.open(stream) as img:
        img.draft("RGB", (max_side, max_side))  # JPEGs decode straight at reduced size: much faster
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((max_side, max_side))
        out = io.BytesIO()
        img.save(out, "JPEG", quality=82)
        return out.getvalue()


def _aspect(img: np.ndarray) -> float:
    return img.shape[1] / img.shape[0]


def _face_crop(img: np.ndarray, face: Face, size: int = 96) -> str:
    """A square close-up of one face, for picking who's who in a group photo."""
    h, w = img.shape[:2]
    x, y, bw, bh = face.box
    side = max(1, int(min(max(bw, bh) * 1.5, w, h)))
    left = int(min(max(x + bw / 2 - side / 2, 0), w - side))
    top = int(min(max(y + bh / 2 - side / 2, 0), h - side))
    crop = cv2.resize(img[top:top + side, left:left + side], (size, size), interpolation=cv2.INTER_AREA)
    return thumbnail_data_url(crop, size)


def _safe_folder(name: str) -> str:
    """A person's name made safe to use as a folder name inside the zip."""
    return re.sub(r"[^\w .()-]+", "", name).strip(" .")[:60] or "Person"


def _unique_name(name: str, used: set[str], folder: str = "") -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    prefix = f"{folder}/" if folder else ""
    candidate, n = name, 1
    while (prefix + candidate).lower() in used:
        candidate = f"{stem}_{n}{dot}{ext}"
        n += 1
    used.add((prefix + candidate).lower())
    return prefix + candidate


def serve(port: int = 8765, open_browser: bool = True) -> None:
    app = create_app(FaceEngine())
    # Waitress keeps connections open between requests. Flask's built-in server opens a new one per
    # photo, and tens of thousands of photos use up the computer's supply of network ports.
    try:
        server = create_server(app, host="127.0.0.1", port=port, threads=8)
    except OSError:
        print(f"Port {port} is already in use. Is Face Finder already open? Try --port {port + 1}.")
        return
    url = f"http://127.0.0.1:{port}/"
    print(f"Face Finder is running at {url}  (only reachable from this computer; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.run()
    except KeyboardInterrupt:
        print("Stopped.")
