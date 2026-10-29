"""Build a labelled test set for measuring Face Finder's accuracy, then score runs against it.

    uv run --with pyarrow python scripts/make_test_photos.py build      # ~1 GB of downloads
    for who in obama kanye; do
      uv run facefinder find test-photos/search --csv results_$who.csv \\
          --ref test-photos/reference/$who/1.jpg --ref test-photos/reference/$who/2.jpg \\
          --ref test-photos/reference/$who/3.jpg
    done
    uv run --with pyarrow python scripts/make_test_photos.py score obama=results_obama.csv kanye=results_kanye.csv

The set (written to test-photos/, which git ignores):
  reference/<person>/1.jpg 2.jpg 3.jpg   3 clear photos of each target person, used as the examples
  reference/credits.csv                  Commons page and license of each reference photo
  search/       shuffled photos named photo_0001.jpg..., so names don't give the answer away:
                  up to 1,000 Wikimedia Commons photos per target person (tagged "depicts: <person>";
                    a photo tagged with several targets counts for each of them)
                  1,300 photos of other people from Labeled Faces in the Wild (2002-2004 news photos)
                  ~100 "hard" photos each of Michelle Obama, Joe Biden, Kim Kardashian and Jay-Z that
                    aren't tagged with any target and whose description doesn't mention one
                  (Kanye has few "depicts" tags, so his are topped up from Commons' "Kanye West" category
                    tree, counting only photos whose caption names him: the tree also holds e.g. sneakers.)
  answers.csv   photo, obama, kanye (yes/no each), source, page, license

Labels come from Commons' "depicts" tags and captions, so a few are imperfect (e.g. a person far away
or facing away). Reference photos must be captioned as that person alone, since a caption like
"Kanye West and Jay-Z performing" over one visible face could be either. Copies of a person's
reference photos are left out so the test can't be won by duplicates.
To add a person, add one line to TARGETS (or HARD_NEGATIVES); the answers.csv columns follow.

`score` prints one table per person. A "wrong match" there is any photo whose answer for that person
is "no" (so a Kanye photo matched as Obama counts against Obama), followed by where those came from.
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-photos"
CACHE = Path.home() / ".cache" / "face-finder" / "test-photos"
USER_AGENT = "FaceFinderTestSet/0.1 (https://github.com/Logan-Huang/Face-Finder)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
LFW_PARQUET = "https://huggingface.co/datasets/bitmind/lfw/resolve/main/data/train-00000-of-00001.parquet"

# One line per person: key -> (Wikidata id, max photos, regex for their name in a title/description,
# regex for their name in an LFW file name, extra Commons search that also finds photos of them).
# Wikidata ids checked with wbgetentities. Few Kanye photos have a "depicts" tag (~70), so his
# are topped up from the "Kanye West" category tree; Obama has plenty of tagged photos.
TARGETS = {
    "obama": ("Q76", 1000, r"barack|president obama|presidente obama|obamas", r"obama", ""),
    "kanye": ("Q15935", 1000, r"kanye|kanyé", r"kanye", 'deepcategory:"Kanye West"'),
}
# People who look/sound similar or appear alongside the targets: key -> (Wikidata id, photos wanted)
HARD_NEGATIVES = {
    "michelle-obama": ("Q13133", 100),
    "joe-biden": ("Q6279", 100),
    "kim-kardashian": ("Q186304", 100),
    "jay-z": ("Q62766", 100),
}
OLD_HARD_FOLDER = {"michelle-obama", "joe-biden"}  # earlier downloads live in cache/hard
N_OTHER = 1300
# Things tagged "depicts Obama" that aren't photos of him.
NOT_A_PHOTO = re.compile(
    r"wax|tussaud|mural|poster|painting|drawing|caricature|statue|sculpture|signature|cartoon|logo|costume|"
    r"impersonat|mask|sticker|button|graffiti|doll|figurine|cake|\bart\b|artwork|illustration|sketch|banner|"
    r"billboard|mosaic|stamp|coin|tattoo|puppet|toy|lego|effigy|cutout|cardboard|mannequin|screenshot|"
    r"magazine|cover|t-shirt|mug|collage|montage|composite|bust|portrait of .* by|\.svg|\.png|\.gif|\.tif",
    re.I,
)
# A caption naming someone else too ("Kanye West and Jay-Z performing"): with one visible face, it
# could be either of them, so such photos are never used as examples.
NAMES_SOMEONE_ELSE = re.compile(
    r"&|\+|\band\b|\bwith\b|\bfeat|\bft\.|\bjoin|jay-?z|kardashian|biden|michelle", re.I)
DUPLICATE = 0.9  # similarity above this to a reference means it's the same photo (or a crop of it)
random.seed(42)


def get(url: str, tries: int = 5) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def commons(**params) -> dict:
    return json.loads(get(COMMONS + "?" + urllib.parse.urlencode({"format": "json", **params})))


def search_commons(query: str, limit: int) -> list[str]:
    titles, offset = [], 0
    while len(titles) < limit:
        for attempt in range(5):  # the API sometimes answers with an error (rate limits) instead of results
            data = commons(action="query", list="search", srsearch=query, srnamespace=6, srlimit=500, sroffset=offset)
            if "query" in data:
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Commons search failed: {data}")
        titles += [r["title"] for r in data["query"]["search"]]
        if "continue" not in data:
            break
        offset = data["continue"]["sroffset"]
    return titles


def image_info(titles: list[str]) -> list[dict]:
    """Thumbnail URL (1280 px wide) and credit for each JPEG file title."""
    infos = []
    for i in range(0, len(titles), 50):
        data = commons(action="query", titles="|".join(titles[i:i + 50]), prop="imageinfo",
                       iiprop="url|mime|size|extmetadata", iiextmetadatafilter="LicenseShortName|ImageDescription",
                       iiurlwidth=1280)
        for page in data["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if not info or info["mime"] != "image/jpeg" or info["width"] < 640 or info["height"] < 480:
                continue
            meta = info.get("extmetadata", {})
            infos.append({
                "title": page["title"],
                "url": info.get("thumburl") or info["url"],
                "page": info["descriptionurl"],
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "description": meta.get("ImageDescription", {}).get("value", ""),
            })
    return infos


def download_all(infos: list[dict], folder: Path) -> list[tuple[dict, Path]]:
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(info):
        path = folder / (re.sub(r"[^\w.-]+", "_", info["title"].removeprefix("File:"))[:120])
        if not path.exists():
            try:
                path.write_bytes(get(info["url"]))
            except Exception:
                return None
        return info, path

    with ThreadPoolExecutor(4) as pool:  # be gentle with Wikimedia
        return [r for r in pool.map(fetch, infos) if r]


def tagged_titles(include: str, exclude: list[str] = ()) -> list[str]:
    query = f"haswbstatement:P180={include} " + "".join(f"-haswbstatement:P180={q} " for q in exclude) + "filetype:bitmap"
    return [t for t in search_commons(query, 3000) if not NOT_A_PHOTO.search(t)]


def build() -> None:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pyarrow.parquet as pq

    from facefinder.engine import FaceEngine, PhotoError, load_image, similarity

    engine = FaceEngine()
    cache: dict[Path, list | None] = {}

    def faces_of(path):
        if path not in cache:
            try:
                cache[path] = engine.faces(load_image(path))
            except PhotoError:
                cache[path] = None
        return cache[path]

    qids = {k: v[0] for k, v in TARGETS.items()}
    mentions = re.compile("|".join(v[2] for v in TARGETS.values()), re.I)

    # Commons photos of each target person; a photo tagged with several gets every label.
    tags: dict[str, set[str]] = {}
    candidates: dict[str, list[tuple[dict, Path]]] = {}
    depicts: dict[str, set[str]] = {}
    for key, (qid, want, _, _, extra) in TARGETS.items():
        print(f"Finding photos of {key} on Wikimedia Commons...")
        tags[key] = set(tagged_titles(qid))
        depicts[key] = set(tags[key])
        n_tagged = len(tags[key])
        if extra:
            tags[key] |= {t for t in search_commons(extra + " filetype:bitmap", 3000) if not NOT_A_PHOTO.search(t)}
        titles = sorted(tags[key])
        random.shuffle(titles)
        candidates[key] = download_all(image_info(titles[: int(want * 1.6)]), CACHE / key)
        print(f"  {n_tagged} tagged, {len(tags[key])} with the extra search, downloaded {len(candidates[key])} candidates")

    all_tagged = set().union(*tags.values())
    text = lambda info: info["title"] + " " + re.sub(r"<[^>]+>", " ", info.get("description", ""))
    names = {k: re.compile(v[2], re.I) for k, v in TARGETS.items()}

    def people_in(info) -> set[str] | None:
        """Who's in a photo: a "depicts" tag counts; being in a category only counts if the caption
        names them too. None if it was found for someone the caption doesn't name (label unsure)."""
        who = {k for k in TARGETS if info["title"] in depicts[k]
               or (info["title"] in tags[k] and names[k].search(text(info)))}
        unsure = {k for k in TARGETS if info["title"] in tags[k]} - who
        return None if unsure else who

    # References: clear solo shots (one large face) of the person most of these photos agree on.
    # Requiring agreement guards against a mis-tagged photo of someone else becoming an example.
    refs: dict[str, list[tuple[dict, Path]]] = {}
    ref_vecs: dict[str, np.ndarray] = {}
    for key in TARGETS:
        solo = []  # photos with exactly one face: (info, path, face width, confidence, embedding)
        for info, path in candidates[key][:500]:
            if sum(info["title"] in tags[k] for k in TARGETS) > 1:
                continue  # tagged with more than one target: not a safe example
            if people_in(info) != {key} or NAMES_SOMEONE_ELSE.search(text(info)):
                continue  # the caption doesn't say it's just them
            faces = faces_of(path)
            if faces and len(faces) == 1 and faces[0].confidence >= 0.8:
                solo.append((info, path, faces[0].box[2], faces[0].confidence, faces[0].embedding))
        if len(solo) < 3:
            sys.exit(f"Couldn't find 3 clear photos of {key} to use as references.")
        vecs = np.stack([x[4] for x in solo])
        sims = vecs @ vecs.T
        support = (sims > 0.5).sum(axis=1)  # how many other solo photos agree it's the same person
        chosen: list[int] = []
        # Prefer large, confident faces; relax only if a person has too few (e.g. mostly stage shots).
        for min_width, min_conf in ((150, 0.9), (100, 0.9), (70, 0.85), (0, 0.8)):
            chosen = []
            for i in np.argsort(-support):
                if solo[i][2] < min_width or solo[i][3] < min_conf or support[i] < 5:
                    continue
                if all(0.5 < sims[i, j] < DUPLICATE for j in chosen):  # same person, but a different photo
                    chosen.append(int(i))
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                print(f"  {key}: reference faces at least {min_width}px wide, confidence {min_conf}")
                break
        if len(chosen) < 3:
            sys.exit(f"Couldn't find 3 clear photos of {key} to use as references.")
        refs[key] = [(solo[i][0], solo[i][1]) for i in chosen]
        ref_vecs[key] = np.stack([solo[i][4] for i in chosen])
        print(f"  {key} references:", "; ".join(info["title"] for info, _ in refs[key]))
    ref_paths = {p for rs in refs.values() for _, p in rs}

    def is_copy(faces):
        return bool(faces) and any(max(similarity(faces, v)) >= DUPLICATE for v in ref_vecs.values())

    # Search photos: (path, set of people in it, source, page, license)
    entries, seen, dropped = [], set(), Counter()
    for key in TARGETS:
        kept = 0
        for info, path in candidates[key]:
            if kept == TARGETS[key][1]:
                break
            if path in ref_paths or info["title"] in seen:
                continue
            who = people_in(info)
            if not who:
                dropped["unsure label"] += 1  # in the category, but the caption doesn't name them
                continue
            faces = faces_of(path)
            if faces is None:
                continue
            if is_copy(faces):
                dropped[key] += 1  # a copy or crop of a reference photo
                continue
            seen.add(info["title"])
            entries.append((path, who, "commons-" + "+".join(sorted(who)), info["page"], info["license"]))
            kept += 1
        print(f"  {key}: kept {kept} (left out {dropped[key]} copies of reference photos, "
              f"{dropped['unsure label']} so far whose caption doesn't name them)")

    # Hard negatives: tagged with a look-alike/companion but with no target, and nothing in the
    # description saying a target is there anyway.
    print("Finding hard negatives on Wikimedia Commons...")
    hard_counts = {}
    for key, (qid, want) in HARD_NEGATIVES.items():
        found = [t for t in tagged_titles(qid, list(qids.values())) if not mentions.search(t) and t not in all_tagged and not re.search(r"family", t, re.I)]
        random.shuffle(found)
        infos = [i for i in image_info(found[: want + 40]) if not mentions.search(i["description"])]
        folder = CACHE / ("hard" if key in OLD_HARD_FOLDER else f"hard-{key}")
        hard = download_all(infos, folder)
        random.shuffle(hard)
        n = 0
        for info, path in hard:
            if n == want:
                break
            faces = faces_of(path)
            if faces is None or is_copy(faces) or info["title"] in seen:
                continue
            seen.add(info["title"])
            entries.append((path, set(), f"commons-hard:{key}", info["page"], info["license"]))
            n += 1
        hard_counts[key] = n
        print(f"  {key}: {n}")

    print("Downloading Labeled Faces in the Wild (other people)...")
    parquet = CACHE / "lfw.parquet"
    if not parquet.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(get(LFW_PARQUET))
    lfw_skip = re.compile("|".join(v[3] for v in TARGETS.values()), re.I)
    by_person: dict[str, list[dict]] = {}
    for row in pq.read_table(parquet).to_pylist():
        person = row["filename"].rsplit("_", 1)[0]
        if not lfw_skip.search(person):
            by_person.setdefault(person, []).append(row)
    people = list(by_person)
    random.shuffle(people)
    lfw_dir = CACHE / "lfw"
    lfw_dir.mkdir(exist_ok=True)
    n_lfw = 0
    for person in people:  # at most 2 photos per person, for variety
        for row in by_person[person][:2]:
            if n_lfw == N_OTHER:
                break
            path = lfw_dir / row["filename"]
            if not path.exists():
                path.write_bytes(row["image"]["bytes"])
            entries.append((path, set(), "lfw:" + person, "", "LFW (research dataset)"))
            n_lfw += 1

    # Write the folder.
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "reference").mkdir(parents=True)
    (OUT / "search").mkdir()
    with (OUT / "reference" / "credits.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "page", "license"])
        for key in TARGETS:
            (OUT / "reference" / key).mkdir()
            for n, (info, path) in enumerate(refs[key], 1):
                shutil.copy(path, OUT / "reference" / key / f"{n}.jpg")
                w.writerow([f"{key}/{n}.jpg", info["title"], info["page"], info["license"]])
    random.shuffle(entries)
    with (OUT / "answers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", *TARGETS, "source", "page", "license"])
        for n, (path, who, source, page, lic) in enumerate(entries, 1):
            name = f"photo_{n:04d}.jpg"
            shutil.copy(path, OUT / "search" / name)
            w.writerow([name, *("yes" if k in who else "no" for k in TARGETS), source, page, lic])
    counts = Counter(src.split(":")[0] if src.startswith("lfw") else src for _, _, src, _, _ in entries)
    print(f"\nWrote {OUT}: {len(entries)} photos to search.")
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n}")


def score(args: list[str]) -> None:
    from facefinder.engine import MATCH_THRESHOLD

    answers = {r["photo"]: r for r in csv.DictReader((OUT / "answers.csv").open())}
    cutoffs = sorted({0.25, 0.30, 0.33, 0.363, 0.40, 0.45, 0.50, MATCH_THRESHOLD})
    for arg in args:
        person, _, results_csv = arg.partition("=")
        if person not in TARGETS or not results_csv:
            sys.exit(f"Use person=results.csv, with person one of: {', '.join(TARGETS)}")
        best = {Path(r["photo"]).name: float(r["best_similarity"] or -1) for r in csv.DictReader(open(results_csv))}
        truth = {n: a[person] == "yes" for n, a in answers.items()}
        pos = sum(truth.values())
        print(f"\n== {person} ({pos} photos contain them, {len(truth) - pos} don't) ==")
        print(f"{'cut-off':>8} {'found':>12} {'missed':>7} {'wrong matches':>14} {'precision':>10}")
        for t in cutoffs:
            tp = sum(1 for n, y in truth.items() if y and best.get(n, -1) >= t)
            fp = sum(1 for n, y in truth.items() if not y and best.get(n, -1) >= t)
            mark = "  <- default" if t == MATCH_THRESHOLD else ""
            print(f"{t:>8.3f} {tp:>5}/{pos:<6} {pos - tp:>7} {fp:>14} {tp / max(tp + fp, 1):>10.1%}{mark}")
        wrong = Counter()
        totals = Counter()
        for n, y in truth.items():
            src = answers[n]["source"]
            src = "lfw" if src.startswith("lfw") else src
            if not y:
                totals[src] += 1
                if best.get(n, -1) >= MATCH_THRESHOLD:
                    wrong[src] += 1
        print(f"Wrong matches at the default cut-off {MATCH_THRESHOLD}, by source:")
        for src in sorted(totals, key=lambda s: -wrong[s]):
            print(f"  {src:<28} {wrong[src]:>4} of {totals[src]}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        build()
    elif len(sys.argv) >= 3 and sys.argv[1] == "score":
        score(sys.argv[2:])
    else:
        sys.exit("Usage: make_test_photos.py build | score person=results.csv [person=results.csv ...]")
