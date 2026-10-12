import io
import json
import zipfile

import numpy as np
from PIL import Image

from facefinder.engine import Face, PhotoError
from facefinder.web import create_app


class FakeEngine:
    """Stands in for the real models: a photo's colour decides whose 'face' it holds."""

    def faces(self, img):
        b, g, r = (int(c) for c in img[0, 0])
        if r > 200 and g > 200 and b > 200:
            raise PhotoError("face detection failed on this image")
        if r > 200:
            return [Face((10, 10, 20, 20), 0.99, np.array([1, 0], np.float32))]
        if g > 200:
            return [Face((0, 0, 5, 5), 0.99, np.array([0, 1], np.float32)),
                    Face((20, 5, 30, 30), 0.99, np.array([0.6, 0.8], np.float32))]
        return []


def _jpeg(color, size=(64, 48)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    buf.seek(0)
    return buf


def _client():
    return create_app(FakeEngine()).test_client()


def test_reference_learns_largest_face_and_reports_problems():
    resp = _client().post("/api/reference", data={"files": [
        (_jpeg("red"), "me.jpg"), (_jpeg("lime"), "group.jpg"), (_jpeg("blue"), "nobody.jpg"),
        (io.BytesIO(b"junk"), "broken.jpg"), (_jpeg("white"), "weird.jpg"),
    ]}).get_json()
    assert [r["name"] for r in resp["refs"]] == ["me.jpg", "group.jpg"]
    assert resp["embeddings"] == [[1.0, 0.0], [0.6000000238418579, 0.800000011920929]]
    assert resp["refs"][1]["note"] == "2 faces found; using the largest"
    assert resp["refs"][0]["faces"][0] == [10 / 64, 10 / 48, 20 / 64, 20 / 48]
    assert resp["refs"][0]["aspect"] == 64 / 48
    assert resp["refs"][0]["thumb"].startswith("data:image/jpeg;base64,")
    assert resp["errors"] == [
        "nobody.jpg: no face found",
        "broken.jpg: not a readable image (UnidentifiedImageError)",
        "weird.jpg: face detection failed on this image",
    ]


def test_scan_scores_every_face():
    client = _client()
    embeddings = json.dumps([[1.0, 0.0]])
    hit = client.post("/api/scan", data={"file": (_jpeg("red"), "a.jpg"), "embeddings": embeddings}).get_json()
    assert hit["best"] == 1.0 and len(hit["faces"]) == 1
    group = client.post("/api/scan", data={"file": (_jpeg("lime"), "b.jpg"), "embeddings": embeddings}).get_json()
    assert [round(f["score"], 3) for f in group["faces"]] == [0.0, 0.6]
    empty = client.post("/api/scan", data={"file": (_jpeg("blue"), "c.jpg"), "embeddings": embeddings}).get_json()
    assert empty["faces"] == [] and empty["best"] is None
    bad = client.post("/api/scan", data={"file": (io.BytesIO(b"x"), "d.jpg"), "embeddings": embeddings}).get_json()
    assert "error" in bad
    failed = client.post("/api/scan", data={"file": (_jpeg("white"), "e.jpg"), "embeddings": embeddings}).get_json()
    assert failed["error"] == "face detection failed on this image"


def test_rejects_requests_addressed_to_other_hostnames():
    client = _client()
    assert client.get("/", headers={"Host": "127.0.0.1:8765"}).status_code == 200
    assert client.get("/", headers={"Host": "attacker.example:8765"}).status_code == 403


def test_zip_bundles_files_with_unique_names():
    resp = _client().post("/api/zip", data={"files": [
        (io.BytesIO(b"one"), "trip/IMG_1.jpg"), (io.BytesIO(b"two"), "party/IMG_1.jpg"),
    ]})
    assert resp.mimetype == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.data)) as z:
        assert z.namelist() == ["IMG_1.jpg", "IMG_1_1.jpg"]
        assert z.read("IMG_1_1.jpg") == b"two"


def test_page_is_self_contained():
    html = _client().get("/").get_data(as_text=True)
    assert "PQ Face Finder" in html
    assert "http://" not in html and "https://" not in html  # no CDNs, fonts or trackers
