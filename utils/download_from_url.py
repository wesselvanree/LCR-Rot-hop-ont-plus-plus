import os

import requests
from os.path import isfile

from tqdm import tqdm


def download_from_url(url: str, path: str):
    """Download the file if it does not exist, and returns the path."""
    if isfile(path):
        return path

    print(f"Downloading {url} to {path}")
    response = requests.get(url, stream=True)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

    return path
