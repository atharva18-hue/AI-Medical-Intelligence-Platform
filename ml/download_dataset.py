"""
Download Chest X-Ray Pneumonia dataset.

Tries Kaggle API first (if ~/.kaggle/kaggle.json exists),
otherwise downloads from Hugging Face mirror of the same dataset.
"""

import os
import sys
import zipfile
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "chest_xray")


def download_from_kaggle():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        return False

    print("Downloading from Kaggle...")
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

    import subprocess
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", "paultimothymooney/chest-xray-pneumonia",
        "-p", os.path.join(BASE_DIR, "data"),
        "--unzip",
    ])
    return os.path.exists(os.path.join(DATA_DIR, "train"))


def download_from_huggingface():
    print("Downloading from Hugging Face (hf-vision/chest-xray-pneumonia)...")
    print("This is ~1.2 GB, may take a few minutes...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    ds = load_dataset("hf-vision/chest-xray-pneumonia")

    # map HF split names to our folder names
    split_map = {"train": "train", "validation": "val", "test": "test"}

    for hf_split, folder_name in split_map.items():
        if hf_split not in ds:
            continue
        split_data = ds[hf_split]
        print(f"  Saving {hf_split} split ({len(split_data)} images)...")

        for i, sample in enumerate(split_data):
            label_idx = sample["label"]
            label_name = split_data.features["label"].names[label_idx]
            out_dir = os.path.join(DATA_DIR, folder_name, label_name)
            os.makedirs(out_dir, exist_ok=True)

            img = sample["image"]
            img_path = os.path.join(out_dir, f"{label_name.lower()}_{i:05d}.jpeg")
            img.save(img_path)

            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(split_data)} done")

    return True


def main():
    if os.path.exists(os.path.join(DATA_DIR, "train", "NORMAL")):
        print(f"Dataset already exists at {DATA_DIR}")
        train_normal = len(os.listdir(os.path.join(DATA_DIR, "train", "NORMAL")))
        train_pneumonia = len(os.listdir(os.path.join(DATA_DIR, "train", "PNEUMONIA")))
        print(f"  train: {train_normal} NORMAL, {train_pneumonia} PNEUMONIA")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if download_from_kaggle():
        print("Downloaded via Kaggle")
    else:
        print("Kaggle credentials not found, using Hugging Face...")
        download_from_huggingface()

    print(f"\nDataset ready at: {DATA_DIR}")
    print("Now run: python ml/train.py --epochs 10")


if __name__ == "__main__":
    main()
