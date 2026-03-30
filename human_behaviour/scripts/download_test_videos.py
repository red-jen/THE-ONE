"""Download free protest / crowd videos from Pexels API for pipeline testing.

All clips are royalty-free under the Pexels license.

Usage:
  set PEXELS_API_KEY=your_key_here
  python scripts/download_test_videos.py

Get a free API key at: https://www.pexels.com/api/new/
"""

import os, sys, pathlib, json, urllib.request, urllib.error

DEST = pathlib.Path(__file__).resolve().parent.parent / "testvideos"
DEST.mkdir(exist_ok=True)

API_KEY = os.environ.get("PEXELS_API_KEY", "")

PEXELS_VIDEO_IDS = [
    4623570,   # protesters on the street
    4614125,   # protest rally in the city
    4614200,   # people marching on the street
    4614191,   # people at a protest
    4614143,   # crowd of people walking on the street
    4614132,   # people standing on the street
    7607953,   # people holding banner on protest
    6804114,   # people rallying on streets
]

HEADERS = {"Authorization": API_KEY}


def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def best_file(video_files, max_h=1080):
    """Pick the best quality file at or below max_h."""
    candidates = [f for f in video_files if (f.get("height") or 0) <= max_h and f.get("link")]
    if not candidates:
        candidates = video_files
    return max(candidates, key=lambda f: f.get("height") or 0)


def download_video(video_id):
    out = DEST / f"protest_{video_id}.mp4"
    if out.exists():
        print(f"  [skip] {out.name} already exists")
        return True

    print(f"  [fetch metadata] video {video_id} ...")
    try:
        data = fetch_json(f"https://api.pexels.com/videos/videos/{video_id}")
    except urllib.error.HTTPError as e:
        print(f"  [fail] API error {e.code} for video {video_id}")
        return False

    files = data.get("video_files", [])
    if not files:
        print(f"  [fail] no files for video {video_id}")
        return False

    chosen = best_file(files)
    link = chosen["link"]
    h = chosen.get("height", "?")
    w = chosen.get("width", "?")
    print(f"  [download] {w}x{h}  ->  {out.name}")

    try:
        urllib.request.urlretrieve(link, str(out))
    except Exception as e:
        print(f"  [fail] download error: {e}")
        if out.exists():
            out.unlink()
        return False

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"  [ok] {size_mb:.1f} MB saved")
    return True


def search_and_download(query="protest march crowd", per_page=5):
    """Fallback: search Pexels and download top results."""
    print(f"\n--- Searching Pexels for '{query}' ---")
    url = f"https://api.pexels.com/videos/search?query={query}&per_page={per_page}"
    try:
        data = fetch_json(url)
    except urllib.error.HTTPError as e:
        print(f"  [fail] search error {e.code}")
        return 0

    count = 0
    for vid in data.get("videos", []):
        vid_id = vid["id"]
        out = DEST / f"protest_{vid_id}.mp4"
        if out.exists():
            print(f"  [skip] {out.name}")
            count += 1
            continue

        files = vid.get("video_files", [])
        if not files:
            continue
        chosen = best_file(files)
        link = chosen["link"]
        print(f"  [download] video {vid_id} ({chosen.get('width','?')}x{chosen.get('height','?')})")
        try:
            urllib.request.urlretrieve(link, str(out))
            size_mb = out.stat().st_size / (1024 * 1024)
            print(f"  [ok] {size_mb:.1f} MB")
            count += 1
        except Exception as e:
            print(f"  [fail] {e}")
            if out.exists():
                out.unlink()
    return count


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set PEXELS_API_KEY environment variable first.")
        print("Get a free key at: https://www.pexels.com/api/new/")
        print()
        print('  On PowerShell:  $env:PEXELS_API_KEY = "your_key"')
        print('  On bash/zsh:    export PEXELS_API_KEY="your_key"')
        sys.exit(1)

    print(f"Destination: {DEST}\n")

    ok, fail = 0, 0
    for vid_id in PEXELS_VIDEO_IDS:
        if download_video(vid_id):
            ok += 1
        else:
            fail += 1

    extra = search_and_download("protest demonstration people", per_page=4)
    ok += extra

    print(f"\nDone: {ok} videos ready, {fail} failed.")
    print(f"Videos saved in: {DEST}")
