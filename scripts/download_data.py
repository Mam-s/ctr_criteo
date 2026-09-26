"""Download a small subset of the Criteo 1TB Click Logs dataset.

Source: https://huggingface.co/datasets/criteo/CriteoClickLogs (CC BY-NC-SA 4.0)

    train -> days 1 and 2
    val   -> day 3
    test  -> day 4

Files land in data/raw/data/day=YYYY-MM-DD/.

Usage:
    python scripts/download_data.py
"""

from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "criteo/CriteoClickLogs"
# Pinned commit so every run gets exactly the same files
REVISION = "e11d69ae913b16cdd7387706a2133def0fdc6ced"

SPLITS = {
    "train": ["2015-02-15", "2015-02-16"],
    "val": ["2015-02-17"],
    "test": ["2015-02-18"],
}
# A full day is ~250 files (~12 GB). Set to None to download whole days.
FILES_PER_DAY = 1

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    api = HfApi()
    for split, days in SPLITS.items():
        for day in days:
            files = sorted(
                f.path
                for f in api.list_repo_tree(
                    REPO_ID, path_in_repo=f"data/day={day}", repo_type="dataset", revision=REVISION
                )
                if f.path.endswith(".parquet")
            )[:FILES_PER_DAY]

            print(f"{split}: {day} ({len(files)} file(s))")
            for path in files:
                # Skips files if they are downloaded already
                hf_hub_download(REPO_ID, path, repo_type="dataset", revision=REVISION, local_dir=RAW_DIR)


if __name__ == "__main__":
    main()
