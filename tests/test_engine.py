import io

import numpy as np
import pytest
from PIL import Image

from facefinder.cli import _unique_path
from facefinder.engine import Face, PhotoError, iter_image_files, load_image, reference_face, similarity
from facefinder.web import _unique_name


def _face(vec, box=(0, 0, 10, 10)):
    v = np.asarray(vec, np.float32)
    return Face(box, 0.9, v / np.linalg.norm(v))


def test_similarity_takes_best_reference():
    faces = [_face([1, 0, 0]), _face([0, 1, 0]), _face([0, 0, 1])]
    refs = np.stack([_face([1, 0, 0]).embedding, _face([0, 1, 1]).embedding])
    scores = similarity(faces, refs)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(np.sqrt(0.5))
    assert scores[2] == pytest.approx(np.sqrt(0.5))


def test_similarity_handles_empty_inputs():
    assert similarity([], np.zeros((1, 3))) == []
    assert similarity([_face([1, 0])], np.zeros((0, 2))) == [0.0]


def test_reference_face_is_largest():
    small, big = _face([1, 0], (0, 0, 10, 10)), _face([0, 1], (50, 50, 40, 30))
    assert reference_face([small, big]) is big
    assert reference_face([]) is None


def test_load_image_applies_exif_rotation():
    img = Image.new("RGB", (40, 20), "red")
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways; viewers rotate it 90 degrees
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    buf.seek(0)
    assert load_image(buf).shape == (40, 20, 3)


def test_load_image_handles_alpha_and_grayscale():
    for mode in ("RGBA", "L", "P"):
        buf = io.BytesIO()
        Image.new(mode, (8, 6)).save(buf, "PNG")
        buf.seek(0)
        assert load_image(buf).shape == (6, 8, 3)


def test_load_image_rejects_non_images():
    with pytest.raises(PhotoError):
        load_image(io.BytesIO(b"definitely not a photo"))


def test_load_image_reads_heic():
    pytest.importorskip("pillow_heif")
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), "blue").save(buf, "HEIF")
    buf.seek(0)
    assert load_image(buf).shape == (24, 32, 3)


def test_iter_image_files_skips_hidden_and_non_images(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.HEIC").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / ".hidden.jpg").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.png").touch()
    (tmp_path / ".thumbs").mkdir()
    (tmp_path / ".thumbs" / "d.jpg").touch()
    flat = {p.name for p in iter_image_files([tmp_path])}
    deep = {p.name for p in iter_image_files([tmp_path], recursive=True)}
    assert flat == {"a.jpg", "b.HEIC"}
    assert deep == {"a.jpg", "b.HEIC", "c.png"}


def test_unique_names():
    used: set[str] = set()
    assert [_unique_name(n, used) for n in ["a.jpg", "A.JPG", "a.jpg", "noext"]] == ["a.jpg", "A_1.JPG", "a_2.jpg", "noext"]


def test_unique_path(tmp_path):
    (tmp_path / "x.jpg").touch()
    assert _unique_path(tmp_path / "x.jpg").name == "x_1.jpg"
    assert _unique_path(tmp_path / "y.jpg").name == "y.jpg"
