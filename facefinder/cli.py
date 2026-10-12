"""Command line: `facefinder find ...` to scan folders, `facefinder web` for the drag-and-drop UI."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

from .engine import (
    MATCH_THRESHOLD, FaceEngine, PhotoError, iter_image_files, load_image, reference_face, similarity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="facefinder",
        description="Find which photos contain a given person. Everything runs on this computer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    find = sub.add_parser("find", help="scan photos/folders for the person shown in --ref")
    find.add_argument("photos", nargs="+", type=Path, help="photo files and/or folders to search")
    find.add_argument("--ref", action="append", required=True, type=Path, metavar="PHOTO",
                      help="a clear photo of the person; repeat --ref to add more angles (recommended)")
    find.add_argument("-r", "--recursive", action="store_true", help="also search sub-folders")
    find.add_argument("-t", "--threshold", type=float, default=MATCH_THRESHOLD,
                      help=f"match cut-off, higher = stricter (default {MATCH_THRESHOLD})")
    find.add_argument("--copy-to", type=Path, help="copy every matching photo into this folder")
    find.add_argument("--csv", type=Path, help="write every photo's score to this CSV file")

    web = sub.add_parser("web", help="open the drag-and-drop web page (served only to this computer)")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--no-browser", action="store_true", help="don't open a browser tab automatically")

    args = parser.parse_args(argv)
    if args.command == "web":
        from .web import serve

        serve(port=args.port, open_browser=not args.no_browser)
        return 0
    return _find(args)


def _find(args: argparse.Namespace) -> int:
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

    photos = list(iter_image_files(args.photos, args.recursive))
    if not photos:
        print("No photos found to search.", file=sys.stderr)
        return 1
    print(f"Searching {len(photos)} photo(s) using {len(references)} reference face(s)...", file=sys.stderr)

    rows, matches = [], []
    for i, path in enumerate(photos, 1):
        try:
            faces = engine.faces(load_image(path))
        except PhotoError as exc:
            rows.append((path, "error", "", 0, str(exc)))
            continue
        scores = similarity(faces, references)
        best = max(scores, default=0.0)
        matched = bool(faces) and best >= args.threshold
        rows.append((path, "match" if matched else "no match", f"{best:.3f}" if faces else "", len(faces), ""))
        if matched:
            matches.append(path)
            print(f"MATCH  {best:.3f}  {path}")
        print(f"  [{i}/{len(photos)}]", end="\r", file=sys.stderr, flush=True)

    print(f"\n{len(matches)} of {len(photos)} photo(s) contain the person.", file=sys.stderr)

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["photo", "result", "best_similarity", "faces", "error"])
            writer.writerows(rows)
        print(f"Scores written to {args.csv}", file=sys.stderr)

    if args.copy_to and matches:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for path in matches:
            shutil.copy2(path, _unique_path(args.copy_to / path.name))
        print(f"Copied {len(matches)} photo(s) to {args.copy_to}", file=sys.stderr)
    return 0


def _unique_path(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


if __name__ == "__main__":
    sys.exit(main())
