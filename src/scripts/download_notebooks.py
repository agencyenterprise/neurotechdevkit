"""Script to download and extract NDK notebooks."""

import argparse
import os
import zipfile
from typing import Optional

import pooch

NOTEBOOKS_URL = (
    "https://agencyenterprise.github.io"
    "/neurotechdevkit/generated/gallery/gallery_jupyter.zip"
)


def _prepare_destination_folder(user_input_destination_path: Optional[str]) -> str:
    if user_input_destination_path is None:
        return "./"

    if os.path.exists(user_input_destination_path):
        if not os.path.isdir(user_input_destination_path):
            raise Exception("Destination path is not a directory.")
    else:
        os.makedirs(user_input_destination_path)
    return user_input_destination_path


def _download_and_extract(zip_url: str, destination_folder: str):
    downloaded_file_path = pooch.retrieve(
        url=zip_url, known_hash=None, fname="gallery_jupyter.zip", progressbar=True
    )
    with zipfile.ZipFile(downloaded_file_path, "r") as zip_ref:
        zip_ref.extractall(destination_folder)
    os.remove(downloaded_file_path)


def run():
    """Download and extract notebooks."""
    parser = argparse.ArgumentParser(description="Download and extract notebooks.")
    parser.add_argument(
        "--destination-path",
        type=str,
        default=None,
        help="The destination folder for the downloaded notebooks.",
    )
    args = parser.parse_args()
    destination_path = _prepare_destination_folder(args.destination_path)
    _download_and_extract(NOTEBOOKS_URL, destination_path)


if __name__ == "__main__":
    run()
