import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from facefinder import cli
from facefinder.engine import Face


class FakeEngine:
    """A photo's colour decides whose 'face' it holds: red is the target person, green someone else."""

    def faces(self, img):
        b, g, r = (int(c) for c in img[0, 0])
        if r > 200:
            return [Face((10, 10, 20, 20), 0.99, np.array([1, 0], np.float32))]
        if g > 200:
            return [Face((0, 0, 5, 5), 0.99, np.array([0, 1], np.float32))]
        return []


def _jpeg(color, size=(32, 24)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr(cli, "FaceEngine", FakeEngine)


def _run(tmp_path, *args):
    (tmp_path / "ref.jpg").write_bytes(_jpeg("red"))
    return cli.main(["find", "--ref", str(tmp_path / "ref.jpg"), *map(str, args)])


def _make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_zip_input_skips_junk_and_labels_members(tmp_path, capsys):
    z = tmp_path / "trip.zip"
    _make_zip(z, {
        "a/red.jpg": _jpeg("red"), "a/b/green.jpg": _jpeg("lime"), "blue.png": _jpeg("blue"),
        "__MACOSX/a/._red.jpg": b"junk", ".hidden/red.jpg": _jpeg("red"), "a/notes.txt": b"hi",
        "a/": b"",
    })
    assert _run(tmp_path, z, "--csv", tmp_path / "out.csv") == 0
    out = capsys.readouterr()
    assert f"{z}/a/red.jpg" in out.out and "green" not in out.out
    assert "1 of 3 photo(s)" in out.err
    assert {r["photo"]: r["result"] for r in _rows(tmp_path / "out.csv")} == {
        f"{z}/a/red.jpg": "match", f"{z}/a/b/green.jpg": "no match", f"{z}/blue.png": "no match"}


def test_zip_inside_searched_folder(tmp_path, capsys):
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    _make_zip(folder / "sub" / "deep.zip", {"x.jpg": _jpeg("red")})
    _make_zip(folder / "top.zip", {"y.jpg": _jpeg("red")})
    (folder / "loose.jpg").write_bytes(_jpeg("red"))
    _run(tmp_path, folder)
    assert capsys.readouterr().out.count("MATCH") == 2
    _run(tmp_path, folder, "-r")
    assert capsys.readouterr().out.count("MATCH") == 3


def test_copy_to_from_zip_is_byte_identical_and_unique(tmp_path):
    red = _jpeg("red")
    z = tmp_path / "p.zip"
    _make_zip(z, {"one/a.jpg": red, "two/a.jpg": red + b"\0"})
    dest = tmp_path / "found"
    _run(tmp_path, z, "--copy-to", dest)
    assert sorted(p.name for p in dest.iterdir()) == ["a.jpg", "a_1.jpg"]
    assert (dest / "a.jpg").read_bytes() == red


def test_copy_to_regular_file(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "me.jpg").write_bytes(_jpeg("red"))
    _run(tmp_path, folder, "--copy-to", tmp_path / "found")
    assert (tmp_path / "found" / "me.jpg").exists()


def test_resume_skips_done_and_appends_without_second_header(tmp_path, capsys):
    folder = tmp_path / "photos"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(_jpeg("red"))
    (folder / "b.jpg").write_bytes(_jpeg("lime"))
    csv_path = tmp_path / "out.csv"
    _run(tmp_path, folder, "--csv", csv_path)
    (folder / "c.jpg").write_bytes(_jpeg("red"))
    capsys.readouterr()
    assert _run(tmp_path, folder, "--csv", csv_path, "--resume") == 0
    out = capsys.readouterr()
    assert "Resuming: 2 photo(s) already done" in out.err
    assert out.out.count("MATCH") == 1 and "c.jpg" in out.out
    assert [r["photo"].rsplit("/", 1)[1] for r in _rows(csv_path)] == ["a.jpg", "b.jpg", "c.jpg"]
    assert csv_path.read_text().count("photo,result") == 1


def test_resume_requires_csv(tmp_path, capsys):
    assert _run(tmp_path, tmp_path, "--resume") == 2
    assert "--resume needs --csv" in capsys.readouterr().err


def test_corrupt_zip_does_not_abort(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip")
    good = tmp_path / "good.jpg"
    good.write_bytes(_jpeg("red"))
    assert _run(tmp_path, bad, good) == 0
    out = capsys.readouterr()
    assert "bad.zip" in out.err and "good.jpg" in out.out


def test_encrypted_member_is_reported(tmp_path, capsys):
    z = tmp_path / "p.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("secret.jpg", _jpeg("red"))
        zf.getinfo("secret.jpg").flag_bits |= 0x1  # recorded in the central directory on close
        zf.writestr("ok.jpg", _jpeg("red"))
    assert _run(tmp_path, z, "--csv", tmp_path / "out.csv") == 0
    rows = {r["photo"].rsplit("/", 1)[1]: r for r in _rows(tmp_path / "out.csv")}
    assert rows["secret.jpg"]["result"] == "error" and rows["secret.jpg"]["error"] == "password-protected"
    assert rows["ok.jpg"]["result"] == "match"


def test_ctrl_c_stops_cleanly(tmp_path, capsys, monkeypatch):
    folder = tmp_path / "photos"
    folder.mkdir()
    for n in "abc":
        (folder / f"{n}.jpg").write_bytes(_jpeg("red"))
    calls = []

    class Interrupting(FakeEngine):
        def faces(self, img):
            calls.append(1)
            if len(calls) == 3:  # references took one call; stop during the 2nd photo
                raise KeyboardInterrupt
            return super().faces(img)

    monkeypatch.setattr(cli, "FaceEngine", Interrupting)
    assert _run(tmp_path, folder, "--csv", tmp_path / "out.csv") == 130
    err = capsys.readouterr().err
    assert "Stopped after 1 of 3 photos." in err and "--resume" in err
    assert len(_rows(tmp_path / "out.csv")) == 1
