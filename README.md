# Face Finder

Find which of your photos contain a given person. Give it a few photos of the person, point it at a
pile of photos (or whole folders), and it shows you every photo they appear in. It can also hand
you the matches as a `.zip` or copy them into a folder.

**Everything runs on your own computer.** Photos are never uploaded to the internet, to an AI
company or anywhere else. The only network access is a one-time download of two small public model
files (~39 MB) from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo).

Works on macOS, Windows and Linux.

## Quick start

1. **Install [uv](https://docs.astral.sh/uv/)** (one time). It sets up the right Python version and
   everything else automatically.
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv`)
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. **Download Face Finder and start it:**
   ```sh
   git clone https://github.com/Logan-Huang/Face-Finder.git
   cd Face-Finder
   uv run facefinder web
   ```
   The first run installs dependencies and downloads the models, so it takes a minute. After that it
   starts in seconds. A browser tab opens at `http://127.0.0.1:8765`.

No git? Click **Code → Download ZIP** on GitHub, unzip it, and run `uv run facefinder web` inside
the folder.

<details>
<summary>Without uv (plain Python 3.10+)</summary>

```sh
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install .
facefinder web
```
</details>

## Use it in the browser (drag and drop)

The page is served by a program running on your computer and is only reachable from your computer.
"Uploading" a photo there just hands it to that local program.

1. **Who are you looking for?** Drop 1–5 clear photos of the person (see tips below).
2. **Which photos should I search?** Drop photos or whole folders (JPEG, PNG, HEIC, WebP, TIFF...).
3. Click **Find this person**. Results come in as each photo is checked. Every face in a photo is
   checked, so group shots work, and the matching face gets a green box.
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

## Tips for good results

There's no training step. The reference photos are just examples to compare against.

- **1 photo works; 3–5 varied photos work best.** Each face is compared with every example and the
  closest one counts, so different angles and lighting catch more photos. More than about 5 rarely
  helps.
- Use clear, well-lit photos where the face is reasonably large, without sunglasses or masks.
- Use photos that look like the person in the photos you're searching. Use recent ones for kids,
  and include with and without glasses if they wear them.
- Prefer solo photos. If a reference has several people, the largest face is used (the page tells
  you when this happens).
- Missing some photos? Add one of the missed ones as another example. Getting wrong people? Slide
  toward **Stricter**.

## How it works

| Step | What happens | Model |
| --- | --- | --- |
| 1 | Decode the photo and turn it upright using its EXIF rotation | Pillow + pillow-heif |
| 2 | Find every face and its eyes, nose and mouth corners | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) |
| 3 | Straighten each face and turn it into a 128-number "faceprint" | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0) |
| 4 | Compare each faceprint with the person's (cosine similarity, -1 to 1) | NumPy |

A photo **matches** when any face in it scores at or above **0.363** against any reference photo.
OpenCV recommends that cut-off for SFace. In a small test with public photos, different people
scored 0.09–0.26 and the same person scored 0.66–1.0.

Speed: about 55 ms per 12-megapixel photo on an Apple M-series Mac (~1,000 photos/minute from the
terminal).

**Limits:** like all face recognition, it struggles with faces that are tiny (back rows of big group
photos), heavily turned away, blurred, covered or in very dim light. Treat results as a strong
filter rather than proof.

**Why not a text model like [Laya](https://huggingface.co/convaiinnovations/laya)?** Laya and similar
decision models only read text and JSON. They can't look at pixels, so they can't tell who is in a
photo. That job needs vision models like the YuNet + SFace pair above.

## Development

```sh
uv run pytest -q   # tests use generated images and a stand-in engine, so no photos are needed
```

Models are cached in `~/.cache/face-finder` (override with `FACEFINDER_MODELS=/some/dir`) and
checked against SHA-256 hashes before use.

## License

[MIT](LICENSE)
