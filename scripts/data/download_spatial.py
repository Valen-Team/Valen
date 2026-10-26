"""Download an explicit, pinned spatial-data manifest with resumable transfers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def download_file(entry):
    destination = Path(entry["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get("bytes")
    if destination.exists() and (expected is None or destination.stat().st_size == expected):
        return {"file": str(destination), "status": "existing", "bytes": destination.stat().st_size}
    temporary = destination.with_suffix(destination.suffix + ".partial")
    last_error = None
    for attempt in range(6):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"User-Agent": "Valen-spatial-data/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urllib.request.urlopen(urllib.request.Request(entry["url"], headers=headers), timeout=90) as response:
                append = offset and response.status == 206
                length = response.headers.get("Content-Length")
                total = (offset if append else 0) + int(length) if length else expected
                with temporary.open("ab" if append else "wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
            if total is not None and temporary.stat().st_size != total:
                raise IOError("Incomplete HTTP response")
            if expected is not None and temporary.stat().st_size != expected:
                raise IOError("Downloaded size differs from source manifest")
            hasher = hashlib.sha256()
            with temporary.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
            if entry.get("sha256") and digest != entry["sha256"]:
                raise IOError("Download checksum mismatch")
            temporary.replace(destination)
            return {"file": str(destination), "status": "downloaded", "bytes": destination.stat().st_size,
                    "sha256": digest}
        except (OSError, ValueError) as exc:
            last_error = str(exc)
            if attempt < 5:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"Download failed for {destination.name}: {last_error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    path = Path(args.manifest)
    entries = json.loads(path.read_text())["files"]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_file, entry): entry for entry in entries}
        for future in as_completed(pending):
            try:
                result = future.result()
            except Exception as exc:
                result = {"file": pending[future]["destination"], "status": "failed", "error": str(exc)}
                failures.append(result)
            print(json.dumps(result), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} transfer(s) failed; rerun the manifest to resume")


if __name__ == "__main__":
    main()
