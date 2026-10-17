"""Command line: `facefinder find ...` to scan folders, `facefinder web` for the drag-and-drop UI."""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from .engine import (
    MATCH_THRESHOLD, FaceEngine, PhotoError, PhotoRef, list_photos, load_image, reference_face, similarity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facefinder",
        description="Find which photos contain a given person. Everything runs on this computer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find", help="scan photos/folders for the person shown in --ref")
    find.add_argument("photos", nargs="+", type=Path,
                      help="photo files, folders and/or .zip files to search (zips are read without unpacking)")
    find.add_argument("--ref", action="append", required=True, type=Path, metavar="PHOTO",
                      help="a clear photo of the person; repeat --ref to add more angles (recommended)")
    find.add_argument("-r", "--recursive", action="store_true", help="also search sub-folders")
    find.add_argument("-t", "--threshold", type=float, default=MATCH_THRESHOLD,
                      help=f"match cut-off, higher = stricter (default {MATCH_THRESHOLD})")
    find.add_argument("--copy-to", type=Path, help="copy each matching photo into this folder as soon as it is found")
    find.add_argument("--csv", type=Path, help="write every photo's score to this CSV file as the run goes")
    find.add_argument("--resume", action="store_true",
                      help="continue an interrupted run: skip photos already listed in --csv and append to it")

    web = sub.add_parser("web", help="open the drag-and-drop web page (served only to this computer)")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-browser", action="store_true", help="don't open a browser tab automatically")

    args = parser.parse_args(argv)
    if args.command == "web":
        from .web import serve

        serve(port=args.port, open_browser=not args.no_browser)
        return 0
    return _find(args)


CSV_HEADER = ["photo", "result", "best_similarity", "faces", "error"]
PROGRESS_EVERY = 0.5  # seconds between progress updates


def _find(args: argparse.Namespace) -> int:
    if args.resume and not args.csv:
        print("--resume needs --csv to know which photos are already done.", file=sys.stderr)
        return 2
    engine = FaceEngine()

    references = []
    for path in args.ref:
        try:
            face = reference_face(engine.faces(load_image(path)))
        except PhotoError as exc:
            print(f"  skipping reference {path}: {exc}", file=sys.stderr)
            continue
        if face is None:
            print(f"  skipping reference {path}: no face found", file=sys.stderr)
            continue
        references.append(face.embedding)
    if not references:
        print("No usable face in the reference photo(s).", file=sys.stderr)
        return 1
    references = np.stack(references)

    def bad_zip(path: Path, exc: Exception) -> None:
        print(f"  skipping {path}: not a readable zip ({exc.__class__.__name__})", file=sys.stderr)

    photos = list_photos(args.photos, args.recursive, bad_zip)
    if not photos:
        print("No photos found to search.", file=sys.stderr)
        return 1

    done: set[str] = set()
    if args.resume and args.csv.exists():
        with args.csv.open(newline="") as f:
            done = {row["photo"] for row in csv.DictReader(f)}
        print(f"Resuming: {len(done):,} photo(s) already done", file=sys.stderr)
        photos = [p for p in photos if p.label not in done]
    total = len(photos)
    print(f"Searching {total:,} photo(s) using {len(references)} reference face(s)...", file=sys.stderr)

    out = writer = None
    if args.csv:
        append = args.resume and args.csv.exists() and args.csv.stat().st_size > 0
        out = args.csv.open("a" if append else "w", newline="")
        writer = csv.writer(out)
        if not append:
            writer.writerow(CSV_HEADER)
            out.flush()
    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)

    progress = _Progress(total)
    zips = _ZipReader()
    matches = processed = 0
    try:
        for ref in photos:
            label = ref.label
            data = None
            try:
                if ref.member is None:
                    img = load_image(ref.path)
                else:
                    data = zips.read(ref)
                    img = load_image(io.BytesIO(data))
                faces = engine.faces(img)
                row = None
            except PhotoError as exc:
                row = (label, "error", "", 0, str(exc))
            if row is None:
                scores = similarity(faces, references)
                best = max(scores, default=0.0)
                matched = bool(faces) and best >= args.threshold
                row = (label, "match" if matched else "no match", f"{best:.3f}" if faces else "", len(faces), "")
                if matched:
                    matches += 1
                    progress.clear()
                    print(f"MATCH  {best:.3f}  {label}", flush=True)
                    if args.copy_to:
                        _copy_match(ref, data, args.copy_to, zips)
            if writer:
                writer.writerow(row)
                out.flush()
            processed += 1
            progress.update(processed, matches)
    except KeyboardInterrupt:
        progress.clear()
        print(f"Stopped after {processed:,} of {total:,} photos.", file=sys.stderr)
        print("Run the same command with --resume to continue." if args.csv
              else "Tip: add --csv results.csv next time so an interrupted run can be resumed.", file=sys.stderr)
        return 130
    finally:
        zips.close()
        if out:
            out.close()

    progress.clear()
    print(f"{matches:,} of {total:,} photo(s) contain the person.", file=sys.stderr)
    if args.csv:
        print(f"Scores written to {args.csv}", file=sys.stderr)
    if args.copy_to and matches:
        print(f"Copied {matches:,} photo(s) to {args.copy_to}", file=sys.stderr)
    return 0


class _ZipReader:
    """Reads zip members into memory, keeping only the current archive open."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._zf: zipfile.ZipFile | None = None

    def read(self, ref: PhotoRef) -> bytes:
        try:
            if self._path != ref.path:
                self.close()
                self._zf, self._path = zipfile.ZipFile(ref.path), ref.path
            info = self._zf.getinfo(ref.member)
            if info.flag_bits & 0x1:
                raise PhotoError("password-protected")
            return self._zf.read(info)
        except PhotoError:
            raise
        except Exception as exc:
            raise PhotoError(f"unreadable zip entry ({exc.__class__.__name__})") from exc

    def date_time(self, ref: PhotoRef) -> tuple[int, ...]:
        """Modified time of a member of the archive that was just read."""
        return self._zf.getinfo(ref.member).date_time

    def close(self) -> None:
        if self._zf:
            self._zf.close()
        self._zf = self._path = None


class _Progress:
    """One in-place status line on stderr, refreshed at most every PROGRESS_EVERY seconds."""

    def __init__(self, total: int) -> None:
        self.total, self._start, self._last, self._width = total, time.monotonic(), 0.0, 0

    def update(self, done: int, matches: int) -> None:
        now = time.monotonic()
        if now - self._last < PROGRESS_EVERY and done < self.total:
            return
        self._last = now
        text = f"[{done:,}/{self.total:,}] {matches:,} match{'es' if matches != 1 else ''}"
        remaining = (now - self._start) / done * (self.total - done)
        if done < self.total:
            text += f" \u00b7 {_eta(remaining)} left"
        self._width = max(self._width, len(text))
        print("\r" + text.ljust(self._width), end="", file=sys.stderr, flush=True)

    def clear(self) -> None:
        if self._width:
            print("\r" + " " * self._width + "\r", end="", file=sys.stderr, flush=True)
            self._width = 0


def _eta(seconds: float) -> str:
    minutes = round(seconds / 60)
    if seconds < 60:
        return "under a minute"
    if minutes < 60:
        return f"about {minutes} min"
    return f"about {minutes // 60} h {minutes % 60:02d} min"


def _copy_match(ref: PhotoRef, data: bytes | None, folder: Path, zips: _ZipReader) -> None:
    if ref.member is None:
        shutil.copy2(ref.path, _unique_path(folder / ref.path.name))
        return
    target = _unique_path(folder / PurePosixPath(ref.member).name)
    target.write_bytes(data)
    try:
        mtime = time.mktime((*zips.date_time(ref), 0, 0, -1))
        os.utime(target, (mtime, mtime))
    except Exception:
        pass  # the copy is what matters; keep the default timestamp


def _unique_path(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


if __name__ == "__main__":
    sys.exit(main())
