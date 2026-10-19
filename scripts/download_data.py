"""Download the pinned upstream bytes. Existing originals are verified, never overwritten."""

import hashlib

import requests

from src.data import ROOT, manifest, verify_file


def main():
    source = manifest()
    for entry in source["files"]:
        target = ROOT / entry["local_path"]
        if target.exists():
            verify_file(target, entry["sha256"])
            print(f"Verified {entry['local_path']}")
            continue
        url = ("https://raw.githubusercontent.com/afrisenti-semeval/afrisent-semeval-2023/"
               f"{source['revision']}/{entry['source_path']}")
        for attempt in range(1, 4):
            try:
                with requests.get(url, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    content = response.content
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                print(f"Retrying {entry['source_path']} (attempt {attempt + 1}/3)")
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Unexpected upstream content: {entry['source_path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        print(f"Downloaded {entry['local_path']}")


if __name__ == "__main__":
    main()
