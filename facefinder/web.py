"""Local web UI. The server listens on 127.0.0.1 only, so photos never leave this computer."""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import threading
import webbrowser
import zipfile
from pathlib import Path

import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
from waitress import create_server

from .engine import (
    MATCH_THRESHOLD, Face, FaceEngine, PhotoError, load_image, reference_face, similarity, thumbnail_data_url,
)

STATIC = Path(__file__).parent / "static"
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


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
        """Learn the person's face from one or more reference photos."""
        refs, embeddings, errors = [], [], []
        for upload in request.files.getlist("files"):
            try:
                img = load_image(upload.stream)
                faces = engine.faces(img)
            except PhotoError as exc:
                errors.append(f"{upload.filename}: {exc}")
                continue
            face = reference_face(faces)
            if face is None:
                errors.append(f"{upload.filename}: no face found")
                continue
            embeddings.append(face.embedding.tolist())
            refs.append({
                "name": upload.filename,
                "thumb": thumbnail_data_url(img),
                "aspect": _aspect(img),
                "faces": [_box(face, img)],
                "note": f"{len(faces)} faces found; using the largest" if len(faces) > 1 else None,
            })
        return jsonify(refs=refs, embeddings=embeddings, errors=errors)

    @app.post("/api/scan")
    def scan():
        """Score every face in one photo against the reference faceprints."""
        upload = request.files["file"]
        references = np.asarray(json.loads(request.form["embeddings"]), np.float32)
        try:
            img = load_image(upload.stream)
            faces = engine.faces(img)
        except PhotoError as exc:
            return jsonify(name=upload.filename, error=str(exc))
        scores = similarity(faces, references)
        # No thumbnail here: with tens of thousands of photos the page would run out of memory.
        # It asks /api/thumb for previews of just the photos on screen instead.
        return jsonify(
            name=upload.filename,
            aspect=_aspect(img),
            faces=[{"box": _box(f, img), "score": s} for f, s in zip(faces, scores)],
            best=max(scores, default=None),
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
        """Bundle the photos the browser sends back (the matches) into one download."""
        out = tempfile.SpooledTemporaryFile(max_size=64 << 20)
        used: set[str] = set()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
            for upload in request.files.getlist("files"):
                name = _unique_name(Path(upload.filename or "photo").name, used)
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


def _unique_name(name: str, used: set[str]) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    candidate, n = name, 1
    while candidate.lower() in used:
        candidate = f"{stem}_{n}{dot}{ext}"
        n += 1
    used.add(candidate.lower())
    return candidate


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
