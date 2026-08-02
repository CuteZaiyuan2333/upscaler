#!/usr/bin/env python3
"""Download 1024x1024 JPG images from Poly Haven: albedo maps + preview renders.

Poly Haven publishes free (CC0) textures and models. Every asset has an albedo
(color) map published as a "1k" JPG, which is exactly 1024x1024 pixels. Each
asset also has a preview render (the material shown on a sphere); we fetch it
at the largest native size and resize/convert it to a 1024x1024 JPG.

Only the albedo map is downloaded -- no PBR maps (roughness, AO, normal, ...).

Downloads stream in as metadata arrives, so files start appearing within a
few seconds. Already-downloaded files are skipped (resume-safe).

Usage:
    python download_polyhaven.py --out images
    python download_polyhaven.py --dry-run                # count without downloading
    python download_polyhaven.py --no-previews            # skip sphere previews
    python download_polyhaven.py --maps rough             # override map filter
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.polyhaven.com"
USER_AGENT = "piccollector/1.0 (download script; contact: none)"
ASSET_TYPES = ("textures", "models")
# Non-map keys found in the /files tree (3D scene formats etc.)
SKIP_KEYS = {"blend", "gltf", "fbx", "usd"}
DEFAULT_MAPS = {"diffuse"}  # albedo only
PREVIEW_SIZE = 1024

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Running under pythonw.exe (double-clicked) has no console streams; keep alive.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


def get_json(url, retries=6):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 60))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(min(2 ** attempt, 60))
    raise RuntimeError(f"giving up on {url}")


def collect_map_jobs(asset_id, map_filter):
    """Return [(asset_id, 'map', url, size, md5)] for the filtered 1k jpg maps."""
    jobs = []
    try:
        data = get_json(f"{API}/files/{asset_id}")
    except Exception:
        print(f"  [meta] FAILED {asset_id}", file=sys.stderr)
        return jobs
    for map_name, res_map in data.items():
        if map_name in SKIP_KEYS or map_name.startswith("_"):
            continue
        if not isinstance(res_map, dict):
            continue
        if map_filter and map_name.lower() not in map_filter:
            continue
        res = res_map.get("1k")
        if not isinstance(res, dict):
            continue
        f = res.get("jpg")
        if not isinstance(f, dict) or "url" not in f:
            continue
        jobs.append((asset_id, "map", f["url"], f.get("size"), f.get("md5")))
    return jobs


def preview_job(asset_id):
    url = f"https://cdn.polyhaven.com/asset_img/thumbs/{asset_id}.png?width={PREVIEW_SIZE}&height={PREVIEW_SIZE}"
    return (asset_id, "preview", url, None, None)


def _out_name(job):
    asset_id, kind, url, size, md5 = job
    if kind == "preview":
        return f"{asset_id}_preview.jpg"
    return os.path.basename(urllib.parse.urlparse(url).path)


def download_one(job, out_dir, verify_md5):
    asset_id, kind, url, size, md5 = job
    name = _out_name(job)
    path = os.path.join(out_dir, name)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return "skip", name

    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
            if kind == "preview":
                if not HAS_PIL:
                    raise RuntimeError("Pillow required for previews (pip install Pillow)")
                with Image.open(tmp) as img:
                    img = img.convert("RGB").resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
                    img.save(tmp, "JPEG", quality=92)
            elif verify_md5 and md5:
                if hashlib.md5(open(tmp, "rb").read()).hexdigest() != md5:
                    os.remove(tmp)
                    raise IOError("md5 mismatch")
            os.replace(tmp, path)
            return "downloaded", name
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, RuntimeError) as e:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt == 5:
                print(f"  [dl] FAILED {name}: {e}", file=sys.stderr)
                return "failed", name
            time.sleep(min(2 ** attempt, 60))
    return "failed", name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="images", help="output directory (default: ./images)")
    ap.add_argument("--workers", type=int, default=6, help="download workers (default: 6)")
    ap.add_argument("--meta-workers", type=int, default=12, help="metadata fetch workers (default: 12)")
    ap.add_argument("--maps", nargs="*", default=None,
                    help="map filter override (default: albedo only, i.e. diffuse)")
    ap.add_argument("--no-previews", action="store_true", help="skip sphere preview images")
    ap.add_argument("--limit", type=int, default=None, help="only scan first N assets (testing)")
    ap.add_argument("--dry-run", action="store_true", help="scan and count only, no downloads")
    ap.add_argument("--no-md5", action="store_true", help="skip md5 verification")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    map_filter = {m.lower() for m in args.maps} if args.maps else set(DEFAULT_MAPS)
    verify_md5 = not args.no_md5
    do_previews = not args.no_previews
    if do_previews and not HAS_PIL:
        print("WARNING: Pillow not installed; skipping sphere previews.", file=sys.stderr)
        do_previews = False

    print("Scanning Poly Haven for albedo + preview images (downloads start immediately)...")
    t0 = time.time()

    asset_ids = []
    for t in ASSET_TYPES:
        data = get_json(f"{API}/assets?t={t}")
        asset_ids.extend(data.keys())
    if args.limit:
        asset_ids = asset_ids[:args.limit]

    if args.dry_run:
        counts = {"map": 0, "preview": 0}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.meta_workers) as ex:
            futs = [ex.submit(collect_map_jobs, aid, map_filter) for aid in asset_ids]
            for fut in concurrent.futures.as_completed(futs):
                counts["map"] += len(fut.result())
        if do_previews:
            counts["preview"] = len(asset_ids)
        for k, n in counts.items():
            print(f"  {k:8s} {n}")
        print(f"total: {sum(counts.values())} files")
        return

    meta_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.meta_workers)
    dl_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)

    # Kick off preview downloads immediately (no metadata needed).
    dl_futs = set()
    if do_previews:
        dl_futs.update(dl_pool.submit(download_one, preview_job(aid), args.out, verify_md5)
                       for aid in asset_ids)

    # Stream: as each asset's metadata arrives, enqueue its albedo downloads.
    meta_futs = {meta_pool.submit(collect_map_jobs, aid, map_filter): aid for aid in asset_ids}

    scanned = 0
    submitted = len(dl_futs)
    done_count = 0
    stats = {"downloaded": 0, "skip": 0, "failed": 0}
    total_assets = len(asset_ids)

    while meta_futs or dl_futs:
        if meta_futs:
            try:
                done_m, meta_futs = concurrent.futures.wait(
                    meta_futs, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            except concurrent.futures.TimeoutError:
                done_m = set()
            for fut in done_m:
                scanned += 1
                for job in fut.result():
                    submitted += 1
                    dl_futs.add(dl_pool.submit(download_one, job, args.out, verify_md5))
                if scanned % 100 == 0 or scanned == total_assets:
                    print(f"  [scan] {scanned}/{total_assets} assets, {submitted} files queued")
        if dl_futs:
            try:
                done_d, dl_futs = concurrent.futures.wait(
                    dl_futs, timeout=0.5, return_when=concurrent.futures.FIRST_COMPLETED)
            except concurrent.futures.TimeoutError:
                done_d = set()
            for fut in done_d:
                status, name = fut.result()
                stats[status] += 1
                done_count += 1
                if done_count % 100 == 0 or done_count == submitted and submitted:
                    print(f"  [dl] {done_count}/{submitted} downloaded={stats['downloaded']} "
                          f"skipped={stats['skip']} failed={stats['failed']}")

    meta_pool.shutdown(wait=False)
    dl_pool.shutdown(wait=False)
    print(f"Done in {time.time()-t0:.0f}s. downloaded={stats['downloaded']} "
          f"skipped={stats['skip']} failed={stats['failed']}")
    print(f"Saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
