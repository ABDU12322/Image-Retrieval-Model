import os
import zipfile
import requests
from tqdm import tqdm

SAVE_DIR = "coco"
CHUNK_SIZE = 8 * 1024 * 1024

URLS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/test2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]

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

for url in URLS:
    zip_path = download(url, SAVE_DIR)
    extract(zip_path, SAVE_DIR)

print("Done.")