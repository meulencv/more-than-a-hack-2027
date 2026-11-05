"""Comprueba los datos locales sin leer etiquetas de posesión.

Ejecutar desde cualquier directorio: python scripts/check_data.py
No modifica archivos. ffprobe es opcional para comprobar el contenedor de vídeo.
"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from games import SOURCES, asset_paths, source_frames
from starter.load_smart_tagging import load, check_time_base


def check_source(source):
    paths = asset_paths(source=source)
    expected = source_frames(source)
    result = {"files": {}, "checks": {}}
    for name, path in paths.items():
        p = Path(path)
        result["files"][name] = {
            "path": str(p), "exists": p.is_file(),
            "bytes": p.stat().st_size if p.is_file() else None,
        }
    checks = result["checks"]
    if Path(paths["atd_meta"]).is_file():
        ET.parse(paths["atd_meta"])
        checks["metadata_xml"] = True
    if Path(paths["atd_raw"]).is_file():
        count, ordered = 0, True
        with open(paths["atd_raw"]) as handle:
            for count, line in enumerate(handle, 1):
                ordered = ordered and int(line.split(":", 1)[0]) == count
        checks["tracking_frames"] = count == expected and ordered
        result["tracking_records"] = count
    if Path(paths["homography"]).is_file():
        count, ordered = 0, True
        with open(paths["homography"], newline="") as handle:
            for count, row in enumerate(csv.DictReader(handle), 1):
                ordered = ordered and int(row["frame"]) == count - 1
        checks["homography_frames"] = count == expected and ordered
        result["homography_records"] = count
    if source == "tactical" and Path(paths["smart_tagging"]).is_file():
        events = load()
        result["smart_tagging_events"] = len(events)
        checks["tagging_time_base"] = check_time_base(events)["agrees"]
    if Path(paths["video"]).is_file() and shutil.which("ffprobe"):
        probe = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,width,height,r_frame_rate",
            "-of", "json", paths["video"],
        ], check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)["streams"][0]
        result["video"] = stream
        checks["video_frames"] = int(stream["nb_frames"]) == expected
    result["complete"] = all(f["exists"] for f in result["files"].values())
    result["ok"] = result["complete"] and all(checks.values())
    return result


if __name__ == "__main__":
    report = {source: check_source(source) for source in SOURCES}
    print(json.dumps(report, indent=2))
    sys.exit(0 if all(item["ok"] for item in report.values()) else 1)
