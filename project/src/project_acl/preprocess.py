"""Preprocessing between raw data and the EDA notebook: clean, save, encode, split.

Run `uv run project-acl-preprocess --help` for the CLI.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from project_acl.data import DATA_DIR, POSITIVE, RAW_DIR, TARGET, load
from project_acl.log import get_logger

log = get_logger(__name__)

PROCESSED = DATA_DIR / "processed" / "adult.csv"


def clean(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    """Optionally drop rows with missing values (`workclass`, `occupation`, `native-country`)."""
    if dropna:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        log.info("dropped %d rows with missing values", before - len(df))
    return df


def save(df: pd.DataFrame, path: Path = PROCESSED) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("saved %s %s", path, df.shape)


def read(path: Path = PROCESSED) -> pd.DataFrame:
    """Processed frame, as read by the EDA notebook and later phases."""
    return pd.read_csv(path)


def encode(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """One-hot categoricals (`col=value`). Missing values -> all-zero dummies.

    Returns X, y and a map encoded column -> source feature.
    """
    feats = df.drop(columns=[target, "split"])
    cats = feats.select_dtypes(exclude="number").columns.tolist()
    X = pd.get_dummies(feats, columns=cats, prefix_sep="=", dtype=int)
    source = {c: c.split("=")[0] for c in X.columns}
    return X, df[target], source


def split(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame):
    """Official UCI split (no randomness). Returns X_train, X_test, y_train, y_test."""
    test = (df["split"] == "test").to_numpy()
    return X[~test], X[test], y[~test], y[test]


def main() -> None:
    p = argparse.ArgumentParser(description="Clean raw UCI Adult data and save it for the EDA.")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out", type=Path, default=PROCESSED)
    p.add_argument("--target", default=TARGET, help="name given to the target column")
    p.add_argument("--positive", default=POSITIVE, help="target value mapped to 1")
    p.add_argument("--dropna", action="store_true", help="drop rows with missing values")
    a = p.parse_args()

    try:
        df = clean(load(a.raw_dir, a.target, a.positive), a.dropna)
        save(df, a.out)
    except (OSError, ValueError) as e:
        log.error("%s (run project-acl-data first?)", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
