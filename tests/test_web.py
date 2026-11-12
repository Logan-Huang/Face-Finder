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


def test_reference_offers_every_face_largest_first_and_reports_problems():
    resp = _client().post("/api/reference", data={"files": [
        (_jpeg("red"), "me.jpg"), (_jpeg("lime"), "group.jpg"), (_jpeg("blue"), "nobody.jpg"),
        (io.BytesIO(b"junk"), "broken.jpg"), (_jpeg("white"), "weird.jpg"),
    ]}).get_json()
    me, group = resp["refs"]
    assert [me["name"], group["name"]] == ["me.jpg", "group.jpg"]
    assert me["faces"][0]["box"] == [10 / 64, 10 / 48, 20 / 64, 20 / 48]
    assert me["faces"][0]["embedding"] == [1.0, 0.0]
    assert me["aspect"] == 64 / 48 and me["thumb"].startswith("data:image/jpeg;base64,")
    # Both faces of the group photo are offered, largest first, each with a close-up to click on.
    assert [f["embedding"] for f in group["faces"]] == [[0.6000000238418579, 0.800000011920929], [0.0, 1.0]]
    assert all(f["crop"].startswith("data:image/jpeg;base64,") for f in group["faces"])
    assert resp["errors"] == [
        "nobody.jpg: no face found",
        "broken.jpg: not a readable image (UnidentifiedImageError)",
        "weird.jpg: face detection failed on this image",
    ]


def test_scan_scores_every_face_against_every_person():
    client = _client()
    people = json.dumps([[[1.0, 0.0]], [[0.0, 1.0]]])  # person 1 looks like "red", person 2 like the small "lime" face
    hit = client.post("/api/scan", data={"file": (_jpeg("red"), "a.jpg"), "people": people}).get_json()
    assert hit["faces"][0]["scores"] == [1.0, 0.0]
    group = client.post("/api/scan", data={"file": (_jpeg("lime"), "b.jpg"), "people": people}).get_json()
    assert [f["scores"] for f in group["faces"]] == [[0.0, 1.0], [0.6, 0.8]]
    empty = client.post("/api/scan", data={"file": (_jpeg("blue"), "c.jpg"), "people": people}).get_json()
    assert empty["faces"] == []
    bad = client.post("/api/scan", data={"file": (io.BytesIO(b"x"), "d.jpg"), "people": people}).get_json()
    assert "error" in bad
    failed = client.post("/api/scan", data={"file": (_jpeg("white"), "e.jpg"), "people": people}).get_json()
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
    assert "Face Finder" in html
    assert "http://" not in html and "https://" not in html  # no CDNs, fonts or trackers


def test_thumb_is_small_upright_jpeg():
    img = Image.new("RGB", (1200, 800), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    client = _client()
    resp = client.post("/api/thumb", data={"file": (buf, "big.jpg")})
    assert resp.mimetype == "image/jpeg"
    assert Image.open(io.BytesIO(resp.data)).size == (320, 480)  # rotated upright, longest side 480
    bad = client.post("/api/thumb", data={"file": (io.BytesIO(b"x"), "bad.jpg")})
    assert bad.status_code == 422


def test_scan_results_carry_no_thumbnail():
    people = json.dumps([[[1.0, 0.0]]])
    resp = _client().post("/api/scan", data={"file": (_jpeg("red"), "a.jpg"), "people": people}).get_json()
    assert "thumb" not in resp


def test_zip_puts_each_person_in_their_own_folder():
    resp = _client().post("/api/zip", data={
        "files": [(io.BytesIO(b"a"), "IMG_1.jpg"), (io.BytesIO(b"b"), "IMG_1.jpg"), (io.BytesIO(b"c"), "IMG_2.jpg")],
        "folders": ["Kanye", "Obama", "../../etc"],
    })
    with zipfile.ZipFile(io.BytesIO(resp.data)) as z:
        assert z.namelist() == ["Kanye/IMG_1.jpg", "Obama/IMG_1.jpg", "etc/IMG_2.jpg"]
        assert z.read("Obama/IMG_1.jpg") == b"b"
