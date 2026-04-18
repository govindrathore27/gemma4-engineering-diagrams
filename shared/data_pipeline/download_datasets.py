import subprocess
from pathlib import Path
import requests

DATASETS = {
    "pid2graph": {
        "type": "zenodo",
        "record_id": "14803338",
        "dest": "project1_pid_copilot/data/pid2graph",
    },
    "pid_symbols": {
        "type": "kaggle_dataset",
        "ref": "hristohristov21/pid-symbols",
        "dest": "project1_pid_copilot/data/pid_symbols",
    },
    "cghd": {
        "type": "zenodo",
        "record_id": "14042961",
        "dest": "project2_circuit_tutor/data/cghd",
    },
    "as1100": {
        "type": "huggingface",
        "ref": "jcrzd/engineering-drawings-as1100",
        "dest": "project3_drawing_tutor/data/as1100",
    },
    "dimensioning": {
        "type": "kaggle_dataset",
        "ref": "alihhhjj/30-engineering-drawing-dimensioning",
        "dest": "project3_drawing_tutor/data/dimensioning",
    },
}


def download_zenodo(record_id: str, dest: str) -> None:
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    url = f"https://zenodo.org/api/records/{record_id}/files"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    for entry in resp.json()["entries"]:
        out = dest_path / entry["key"]
        if out.exists():
            continue
        r = requests.get(entry["links"]["content"], stream=True, timeout=60)
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_kaggle_dataset(ref: str, dest: str) -> None:
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", ref, "-p", dest, "--unzip"],
        check=True,
    )


def download_huggingface(ref: str, dest: str) -> None:
    from datasets import load_dataset
    Path(dest).mkdir(parents=True, exist_ok=True)
    ds = load_dataset(ref)
    ds.save_to_disk(dest)


def download_all(names: list[str] | None = None) -> None:
    if names is not None:
        unknown = set(names) - DATASETS.keys()
        if unknown:
            raise ValueError(f"Unknown dataset names: {unknown}. Available: {list(DATASETS.keys())}")
    targets = {k: v for k, v in DATASETS.items() if names is None or k in names}
    handlers = {
        "zenodo": lambda cfg: download_zenodo(cfg["record_id"], cfg["dest"]),
        "kaggle_dataset": lambda cfg: download_kaggle_dataset(cfg["ref"], cfg["dest"]),
        "huggingface": lambda cfg: download_huggingface(cfg["ref"], cfg["dest"]),
    }
    for name, cfg in targets.items():
        print(f"Downloading {name}...")
        try:
            handlers[cfg["type"]](cfg)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")


if __name__ == "__main__":
    import sys
    download_all(sys.argv[1:] or None)
