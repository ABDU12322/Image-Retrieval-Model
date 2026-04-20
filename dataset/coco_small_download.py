import os
import requests
from tqdm import tqdm

SAVE_DIR = "coco_small"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
ANNO_DIR = os.path.join(SAVE_DIR, "annotations")

# Val2017 has exactly 5,000 images + its annotation file
IMAGE_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNO_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

import zipfile

CHUNK_SIZE = 8 * 1024 * 1024

def download(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    filename = url.split("/")[-1]
    path = os.path.join(save_dir, filename)

    if os.path.exists(path):
        print(f"Already exists: {filename}")
        return path

    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))

    with open(path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as bar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))

    return path

def extract(path, extract_to):
    print(f"Extracting {os.path.basename(path)} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_to)
    os.remove(path)

zip_path = download(IMAGE_URL, IMAGE_DIR)
extract(zip_path, IMAGE_DIR)

zip_path = download(ANNO_URL, ANNO_DIR)
extract(zip_path, ANNO_DIR)

print("Done. 5,000 val2017 images downloaded.")