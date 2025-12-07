# data_loader.py
"""
Download dataset using kagglehub.
This script checks ./data/ and downloads if missing.
"""

import os
from pathlib import Path
import kagglehub

DATA_DIR = Path("data")
KAGGLE_DATASET = "luiscrmartins/surveillance-images-for-person-detection"

def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Checking dataset in {DATA_DIR} ...")
    # kagglehub.dataset_download returns local path (per your environment)
    path = kagglehub.dataset_download(KAGGLE_DATASET, path=str(DATA_DIR))
    print("Downloaded dataset to:", path)

if __name__ == "__main__":
    download_dataset()
