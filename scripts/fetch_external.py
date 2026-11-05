"""Fetch the exact external Metrica files recorded in docs/external_data_manifest.json."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/external/metrica_sample"))
    args = parser.parse_args()
    manifest_path = Path(__file__).resolve().parents[1] / "docs/external_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Manifest filename must be a basename")
        target = args.output_dir / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            print("Verified", name)
            continue
        game = name.split("_")[2]
        url = ("https://raw.githubusercontent.com/metrica-sports/sample-data/"
               + manifest["revision"] + "/data/Sample_Game_" + game + "/" + name)
        temporary = target.with_suffix(".tmp")
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("Checksum mismatch: " + name)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print("Downloaded and verified", name)


if __name__ == "__main__":
    main()
