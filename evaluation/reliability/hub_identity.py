"""Record the Hugging Face commit and per-file SHA-256 of the Valen-Decision-1.5 repositories.

Weights and other LFS files use the SHA-256 the Hub stores for them; every other
file is downloaded at the resolved commit and hashed. banking77.py --study
decision-1.5 compares each hash a run recorded with this file. Standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

MODELS = ("Valen-Decision-1.5-0.8B", "Valen-Decision-1.5-2B", "Valen-Decision-1.5-4B")
HUB = "https://huggingface.co"


def fetch(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def identity(model, revision="main"):
    repo = "Valen-Team/" + model
    commit = json.loads(fetch(f"{HUB}/api/models/{repo}/revision/{revision}"))["sha"]
    files = {}
    for entry in json.loads(fetch(f"{HUB}/api/models/{repo}/tree/{commit}?recursive=true")):
        if entry["type"] != "file":
            continue
        if entry.get("lfs"):
            files[entry["path"]] = entry["lfs"]["oid"]
        else:
            data = fetch(f"{HUB}/{repo}/resolve/{commit}/{entry['path']}")
            if len(data) != entry["size"]:
                raise ValueError("Size mismatch: " + entry["path"])
            files[entry["path"]] = hashlib.sha256(data).hexdigest()
    return {"repo_id": repo, "revision": commit, "files_sha256": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--revision", action="append", default=[], metavar="MODEL=COMMIT")
    args = parser.parse_args(argv)
    revisions = dict(item.split("=", 1) for item in args.revision)
    result = {model: identity(model, revisions.get(model, "main")) for model in MODELS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
