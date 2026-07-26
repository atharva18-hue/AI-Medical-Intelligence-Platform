"""
Creates sample chest x-ray like images for demo/training when
full Kaggle dataset is not available.
These are synthetic grayscale patterns - NOT real medical images.
For actual project evaluation use the real Kaggle dataset.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample_images")


def make_xray_like(label, idx):
    """Generate a fake x-ray looking image"""
    img = Image.new("RGB", (224, 224), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)

    # lung shaped regions
    draw.ellipse([30, 60, 100, 180], fill=(60, 60, 70), outline=(80, 80, 90))
    draw.ellipse([124, 60, 194, 180], fill=(60, 60, 70), outline=(80, 80, 90))

    if label == "PNEUMONIA":
        # add some white patches for opacity
        for _ in range(5):
            x = np.random.randint(40, 180)
            y = np.random.randint(80, 160)
            r = np.random.randint(10, 30)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(140, 140, 150))

    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


def main():
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(SAMPLE_DIR, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(20):  # 20 images per class
            img = make_xray_like(label, i)
            img.save(os.path.join(folder, f"{label.lower()}_{i:03d}.png"))
        print(f"Created 20 sample images in {folder}")

    print("\nDone! Now run: python ml/train.py --demo")


if __name__ == "__main__":
    main()
