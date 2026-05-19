"""Dataset download utilities.

Supports:
  1. KaggleHub (preferred) – downloads via kagglehub with caching
  2. Kaggle API – uses the official kaggle CLI
  3. Manual path – user supplies a local file path

Supported datasets:
  - LendingClub Loan Data (primary)
  - Home Credit Default Risk (advanced)
  - Give Me Some Credit (quick validation)
  - German Credit Dataset (sanity check)
  - FICO HELOC (XAI benchmark)
"""

import os
import zipfile
from pathlib import Path

import pandas as pd


DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = {
    "lendingclub": {
        "kaggle_path": "wordsforthewise/lending-club",
        "files": ["accepted_2007_to_2018Q4.csv.gz", "rejected_2007_to_2018Q4.csv.gz"],
        "description": "LendingClub 2007-2018 accepted/rejected loans",
    },
    "home_credit": {
        "kaggle_path": "home-credit-default-risk",
        "files": [
            "application_train.csv",
            "application_test.csv",
            "bureau.csv",
            "bureau_balance.csv",
            "credit_card_balance.csv",
            "installments_payments.csv",
            "previous_application.csv",
            "POS_CASH_balance.csv",
        ],
        "description": "Home Credit Default Risk (7 tables)",
    },
    "give_me_some_credit": {
        "kaggle_path": "GiveMeSomeCredit",
        "files": ["cs-training.csv"],
        "description": "Give Me Some Credit (150K loans)",
    },
}


def download_kagglehub(dataset_path: str, save_dir: Path) -> Path:
    """Download a dataset via kagglehub (no API key needed for public datasets)."""
    import kagglehub

    download_path = kagglehub.dataset_download(dataset_path)
    download_path = Path(download_path)
    target = save_dir / dataset_path.replace("/", "_")
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        for f in download_path.iterdir():
            dest = target / f.name
            if not dest.exists():
                f.rename(dest)
    return target


def download_kaggle_api(dataset_path: str, save_dir: Path) -> Path:
    """Download a dataset via the kaggle CLI (requires `kaggle` pip package + API key)."""
    import subprocess

    target = save_dir / dataset_path.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_path, "-p", str(target)],
        check=True,
    )
    for archive in target.glob("*.zip"):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    return target


def download_manual(source_path: str, save_dir: Path, dataset_name: str) -> Path:
    """Register a manually placed dataset file or directory."""
    source = Path(source_path)
    target = save_dir / dataset_name
    if not source.exists():
        raise FileNotFoundError(f"Manual source not found: {source_path}")
    if source.is_dir():
        return source
    if source.suffix in (".csv", ".gz"):
        df = pd.read_csv(source, compression="infer")
        target.mkdir(parents=True, exist_ok=True)
        df.to_csv(target / source.name, index=False)
        return target
    return source


def get_dataset(
    name: str,
    method: str = "kagglehub",
    manual_path: str | None = None,
    force: bool = False,
) -> Path:
    """Download or locate a dataset. Returns the path to the dataset directory.

    Args:
        name: One of DATASETS keys ("lendingclub", "home_credit", "give_me_some_credit").
        method: "kagglehub" (default), "kaggle_api", or "manual".
        manual_path: Required when method="manual" – path to local CSV/directory.
        force: If True, re-download even if data already exists.

    Returns:
        Path to the dataset directory.
    """
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}"
        )

    meta = DATASETS[name]
    target = DATA_RAW / meta["kaggle_path"].replace("/", "_")

    if target.exists() and any(target.iterdir()) and not force:
        print(f"Dataset '{name}' already exists at {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)

    if method == "kagglehub":
        return download_kagglehub(meta["kaggle_path"], DATA_RAW)
    elif method == "kaggle_api":
        return download_kaggle_api(meta["kaggle_path"], DATA_RAW)
    elif method == "manual":
        if not manual_path:
            raise ValueError("manual_path is required for method='manual'")
        return download_manual(manual_path, DATA_RAW, name)
    else:
        raise ValueError(f"Unknown method '{method}'")
