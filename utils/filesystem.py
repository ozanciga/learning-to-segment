import os
import shutil
import numpy as np
from pathlib import Path


def make_folder(pth, purge=False):
    if purge and os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth, exist_ok=True)


def fetch_metadata(pth):
    if os.path.exists(pth):
        return np.load(pth, allow_pickle=True).flatten()[0]
    return {}


def fix_path(pth):
    base_path = Path(__file__).parent
    return (base_path / pth).resolve().as_posix()
