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

1. **Who are you looking for?** Name the person and drop 1–5 clear photos of them (see tips
   below). To look for several people at once, click **+ Add another person** for each one. If an
   example photo has more than one face, tap the right person's face. Group photos are fine, and the
   same photo can be an example for several people, with a different face picked for each.
2. **Which photos should I search?** Drop photos, whole folders or `.zip` files (JPEG, PNG, HEIC,
   WebP, TIFF...). Zips are opened right in your browser, so there's no need to unzip them first.
3. Click **Find this person** (or **Find these people**). Results come in as each photo is checked.
   Every face in a photo is checked, so group shots work. Each person gets their own tab and colour,
   and a photo with two of them shows up in both tabs.
4. Use the **Looser ↔ Stricter** slider to adjust the cut-off without rescanning. Collect the
   matching photos with **Download matches (.zip)**, or in Chrome and Edge, **Save matches to a
   folder**, which writes them straight into a folder you pick (best for thousands of matches).
   When you search for several people, each one gets their own folder.

Press `Ctrl+C` in the terminal to stop it.

## Use it from the terminal

```sh
# Search a folder (add -r to include sub-folders) and copy the matches somewhere
uv run facefinder find ~/Pictures/Trip -r --ref me1.jpg --ref me2.jpg --copy-to ~/Pictures/Found

# Zips work too, without unzipping. --csv saves every photo's score as it goes
uv run facefinder find Photos.zip --ref me1.jpg --csv scores.csv --copy-to Found

# Stopped halfway (Ctrl+C, closed the terminal)? Add --resume to the same command to continue
uv run facefinder find Photos.zip --ref me1.jpg --csv scores.csv --copy-to Found --resume
```

Repeat `--ref` for each reference photo. `-t/--threshold` changes the cut-off (see below). The
terminal looks for one person at a time and uses the largest face in each `--ref` photo; use the
browser to search for several people at once or to pick a face in a group photo.

## Big libraries (tens of thousands of photos)

Both ways handle huge collections. They were tested with a single 70,000-photo `.zip`.

- **Browser:** memory stays small no matter how many photos there are, because previews are only
  made for the photos on screen and results are shown 200 at a time. Keep the tab open while it
  searches; the page asks before you close it mid-search.
- **Terminal:** best for very large jobs. It copies matches as it finds them, shows how long is
  left, and with `--csv` it can `--resume` where it stopped.

Time depends on photo size: about 70 ms per 12-megapixel photo on an Apple M-series Mac, so
70,000 phone photos take roughly 1–1.5 hours.

## Tips for good results

There's no training step. The reference photos are just examples to compare against.

- **1 photo works. Variety matters more than quantity.** Each face is compared with every example
  and the closest one counts, so photos from different angles and lighting catch more. In testing,
  3 similar front-facing photos found only 1% more than 1 photo did.
- Use clear, well-lit photos where the face is reasonably large, without sunglasses or masks.
- Use photos that look like the person in the photos you're searching. Use recent ones for kids,
  and include with and without glasses if they wear them.
- In a group photo, check which face is picked. If the person already has another example, the
  face most like it is picked for you; otherwise the biggest face is guessed and the page asks you
  to confirm.
- Missing some photos? Add one of the missed ones as another example. Getting wrong people? Slide
  toward **Stricter**.

## How it works

| Step | What happens | Model |
| --- | --- | --- |
| 1 | Decode the photo and turn it upright using its EXIF rotation | Pillow + pillow-heif |
| 2 | Find every face and its eyes, nose and mouth corners | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT) |
| 3 | Straighten each face and turn it into a 128-number "faceprint" | [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0) |
| 4 | Compare each faceprint with the person's (cosine similarity, -1 to 1) | NumPy |

A photo **matches** when any face in it scores at or above **0.40** against any of the person's
example photos. When you look for several people, each face counts only for the person it looks
most like, so one face can't match two people.

OpenCV suggests 0.363 for SFace. On our labelled test set, 0.40 cut wrong matches from 18 to 4
while finding 3% fewer photos. See [Measuring accuracy](#measuring-accuracy).

**Limits:** like all face recognition, it struggles with faces that are tiny (back rows of big group
photos), heavily turned away, blurred, covered or in very dim light. Treat results as a strong
filter rather than proof.

## Measuring accuracy

`scripts/make_test_photos.py` builds a labelled test set in `test-photos/` (git ignores it; about
1 GB of downloads) from Wikimedia Commons and the Labeled Faces in the Wild research dataset, and
scores a run against it. Its first lines show the exact commands.

Latest results at the default cut-off (0.40), searching 3,171 photos with 3 example photos each:

| Person | Photos of them | Found | Wrong matches among the other photos |
| --- | --- | --- | --- |
| Barack Obama | 1,000 | 759 (76%) | 4 of 2,171 |
| Kanye West | 619 | 81 (13%) | 12 of 2,552 |

The two sets measure different things. Obama's is mostly event and press photos, close to a
typical photo library; most of the photos it missed show him facing away, in profile or far away.
Kanye's is mostly concert photos: in a sample of 150, 56% had no detectable face at all (stage
lighting, distance, masks), so it shows where face recognition stops working rather than
typical accuracy. Searching for both at once gave nearly the same results (11 wrong Kanye matches
instead of 12); a face only ever counts for the person it looks most like.

## Development

```sh
uv run pytest -q   # tests use generated images and a stand-in engine, so no photos are needed
```

Models are cached in `~/.cache/face-finder` (override with `FACEFINDER_MODELS=/some/dir`) and
checked against SHA-256 hashes before use.

## License

[MIT](LICENSE)
