# PQ Face Finder

Find which of your photos contain a given person. Give it a few photos of the person, point it at a
pile of photos (or whole folders), and it shows you every photo they appear in. It can also hand
you the matches as a `.zip` or copy them into a folder.

**Everything runs on your computer.** Photos are never uploaded to the internet, to an AI company or
anywhere else. The only network access is a one-time download of two small public model files
(~39 MB) from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo).

## Setup

You need [uv](https://docs.astral.sh/uv/) (`brew install uv`). It installs the right Python and
dependencies automatically on first run.

```sh
cd PQ-Face-Finder
uv sync
```

## Use it in the browser (drag and drop)

```sh
uv run facefinder web
```

This opens `http://127.0.0.1:8765` in your browser. The page is served by a program running on your
Mac and is only reachable from your Mac. "Uploading" a photo there just hands it to that local
program.

1. **Who are you looking for?** Drop 1–5 clear photos of the person. A few different angles and
   lighting conditions make it more reliable. If a reference photo has several people, the largest
   face is used (the page tells you when this happens).
2. **Which photos should I search?** Drop photos or whole folders (JPEG, PNG, HEIC, WebP, TIFF...).
3. Click **Find this person**. Results come in as each photo is checked. Matching faces get a green
   box.
4. Use the **Looser ↔ Stricter** slider to adjust the cut-off without rescanning, and
   **Download matches (.zip)** to collect the matching photos.

Press `Ctrl+C` in the terminal to stop it.

## Use it from the terminal (big folders)

```sh
# Search a folder (add -r to include sub-folders) and copy the matches somewhere
uv run facefinder find ~/Pictures/Trip -r --ref me1.jpg --ref me2.jpg --copy-to ~/Pictures/Found

# Also save every photo's score to a spreadsheet
uv run facefinder find ~/Pictures/Trip --ref me1.jpg --csv scores.csv
```

Repeat `--ref` for each reference photo. `-t/--threshold` changes the cut-off (see below).

## How it works

| Step | What happens | Model |
| --- | --- | --- |
| 1 | Decode the photo and turn it upright using its EXIF rotation | Pillow + pillow-heif |
| 2 | Find every face and its eyes, nose and mouth corners | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) |
| 3 | Straighten each face and turn it into a 128-number "faceprint" | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0) |
| 4 | Compare each faceprint with the person's (cosine similarity, -1 to 1) | NumPy |

A photo **matches** when any face in it scores at or above **0.363** against any reference photo.
OpenCV recommends that cut-off for SFace. In a small test with public photos, different people
scored 0.09–0.26 and the same person scored 0.66–1.0. Slide the cut-off up if you see false matches and down if the
tool misses the person.

Speed: about 55 ms per 12-megapixel photo on an Apple M-series Mac (~1,000 photos/minute from the
terminal).

**Limits:** like all face recognition, it struggles with faces that are tiny, heavily turned away,
blurred, covered (sunglasses, masks) or in very dim light. Young children's faces change quickly, so
use recent reference photos. Treat results as a strong filter rather than proof.

## Why not Laya?

`laya-0.1.1.zip` in this folder is [Laya](https://huggingface.co/convaiinnovations/laya), a
Jev-compatible "System 1" decision model. It is a **text-only** model (a ModernBERT text encoder):
you give it text or JSON and typed questions, and it returns calibrated choices, scores and yes/no
probabilities. It has no way to look at pixels, so it can't tell who is in a photo. That job needs a
vision model like the YuNet + SFace pair above. Laya would only become useful as an extra step if
you later want text-based sorting, for example routing photos by caption, filename or album notes.

## Development

```sh
uv run pytest -q   # tests use generated images and a stand-in engine, so no photos are needed
```

Models are cached in `~/.cache/pq-face-finder` (override with `FACEFINDER_MODELS=/some/dir`) and
checked against SHA-256 hashes before use.
