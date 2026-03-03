#!/usr/bin/env python3
"""Download the SimTK IMU dataset (study20-latest.tar.gz, ~5.5GB)."""

import os
import re
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
_URL = "https://datashare.simtk.org/apps/browse/download/sendRelease.php"
_FILENAME = "study20-latest.tar.gz"
_GROUP_ID = "2164"
_STUDY_ID = "20"
_MANUAL_URL = "https://simtk.org/plugins/datashare/index.php?group_id=2164"


def _get_download_token():
    """Fetch a fresh download token from SimTK."""
    resp = requests.post(
        "https://simtk.org/plugins/datashare/view.php",
        data={"id": _GROUP_ID, "pluginname": "datashare", "studyid": _STUDY_ID},
    )
    resp.raise_for_status()
    match = re.search(r'<iframe[^>]+src="([^"]+)"', resp.text)
    if not match:
        raise RuntimeError("Could not find datashare iframe in response")
    iframe_url = match.group(1)
    params = parse_qs(urlparse(iframe_url).query)
    if "token" not in params:
        raise RuntimeError("Token not found in iframe URL")
    return params["token"][0]


def download_and_extract_simtk_dataset():
    """Download, extract, and clean up the SimTK IMU dataset (~5.5GB)."""
    data_dir = ROOT / "data"
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Data directory already exists: {data_dir}")
        print("Skipping download. Delete the data/ directory to re-download.")
        return

    try:
        print("Fetching download token...")
        token = _get_download_token()
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(f"Please download the dataset manually from:\n  {_MANUAL_URL}")
        print(f"Extract the archive and rename the 'files' directory to 'data/'.")
        sys.exit(1)

    try:
        params = {"groupid": _GROUP_ID, "userid": "0", "studyid": _STUDY_ID, "token": token}
        with requests.get(_URL, params=params, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(_FILENAME, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print(f"Please download the dataset manually from:\n  {_MANUAL_URL}")
        print(f"Extract the archive and rename the 'files' directory to 'data/'.")
        sys.exit(1)

    print(f"Extracting {_FILENAME}...")
    with tarfile.open(_FILENAME, "r:gz") as tar:
        tar.extractall(filter="data")
    os.rename("files", str(data_dir))
    os.remove(_FILENAME)
    print("Done.")


if __name__ == "__main__":
    download_and_extract_simtk_dataset()
