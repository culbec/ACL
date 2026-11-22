"""UCI Adult: download and load raw data. Cleaning/encoding live in `preprocess.py`.

Run `uv run project-acl-data --help` for the CLI.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from project_acl.log import get_logger

log = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
NAMES_FILE = "adult.names"
SPLITS = {"train": "adult.data", "test": "adult.test"}
TARGET = "income"  # adult.names lists the target values but gives the column no name
POSITIVE = ">50K"


def download(
    raw_dir: Path = RAW_DIR,
    base_url: str = BASE_URL,
    force: bool = False,
    timeout: float = 60,
    retries: int = 3,
) -> None:
    """Fetch the schema + both split files. Writes via `.part` so a failed run leaves no corrupt file.

    Raises RuntimeError listing every file that still failed after `retries` attempts.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    with Progress(
        TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn()
    ) as progress:
        for name in [NAMES_FILE, *SPLITS.values()]:  # adult.names gives the column names
            path = raw_dir / name
            if path.exists() and not force:
                log.info("skip %s (present)", name)
                continue
            for attempt in range(1, retries + 1):
                try:
                    _fetch(base_url + name, path, timeout, progress)
                    break
                except (requests.RequestException, OSError) as e:
                    log.warning("%s: attempt %d/%d failed: %s", name, attempt, retries, e)
            else:
                failed.append(name)
    if failed:
        raise RuntimeError(f"download failed for {failed}; check the URL / network and rerun")


def _fetch(url: str, path: Path, timeout: float, progress: Progress) -> None:
    """Stream one file to `<path>.part`, then rename it, so `path` exists only when complete."""
    part = path.with_name(path.name + ".part")
    task = progress.add_task(path.name, total=None)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            size = r.headers.get("content-length")
            progress.update(task, total=int(size) if size else None)
            with part.open("wb") as f:
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    f.write(chunk)
                    progress.advance(task, len(chunk))
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
        progress.remove_task(task)


def columns(raw_dir: Path = RAW_DIR, target: str = TARGET) -> list[str]:
    """Feature names parsed from `adult.names` (lines `name: values.`), plus the target column last."""
    text = (raw_dir / NAMES_FILE).read_text()
    feats = re.findall(r"^([\w-]+): .+\.$", text, flags=re.MULTILINE)
    if not feats:
        raise ValueError(f"no attribute lines found in {raw_dir / NAMES_FILE}")
    return [*feats, target]


def load(raw_dir: Path = RAW_DIR, target: str = TARGET, positive: str = POSITIVE) -> pd.DataFrame:
    """Both official files in one frame; `split` column marks train/test, target is 0/1.

    Raw values only: missing values (`?`) are NaN, no rows dropped. Cleaning happens in the EDA.
    """
    names = columns(raw_dir, target)
    parts = []
    for split, name in SPLITS.items():
        df = pd.read_csv(
            raw_dir / name, names=names, skipinitialspace=True, na_values="?",
            skiprows=1 if split == "test" else 0,  # adult.test starts with a junk header line
        )
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df[target] = (df[target].str.rstrip(".") == positive).astype(int)  # test labels end in '.', e.g. '>50K.'
    log.info("loaded %d rows (%s)", len(df), df["split"].value_counts().to_dict())
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Download and load the UCI Adult dataset.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--target", default=TARGET, help="name given to the target column")
    p.add_argument("--positive", default=POSITIVE, help="target value mapped to 1")
    p.add_argument("--force", action="store_true", help="re-download files already present")
    p.add_argument("--timeout", type=float, default=60, help="seconds per request")
    p.add_argument("--retries", type=int, default=3)
    a = p.parse_args()

    try:
        download(a.raw_dir, a.base_url, a.force, a.timeout, a.retries)
        df = load(a.raw_dir, a.target, a.positive)
    except (RuntimeError, OSError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)
    log.info("shape %s, positive rate %.3f", df.shape, df[a.target].mean())
    log.info("missing: %s", df.isna().sum()[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
