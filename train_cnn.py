import os
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATASET_ROOT = "data/raw"
SAMPLE_IMAGES_PER_CLASS = 1
TARGET_SIZE = 256
# ==========================================


# ---------- Utility ----------
def resize_and_pad(img, size=256, pad_color=0):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    return cv2.copyMakeBorder(
        img_resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=pad_color
    )


# ================= 1. EDA =================
print("\n=== Sprint-1: Dataset Overview ===")

class_counts = defaultdict(int)
image_sizes = []

for main_cat in ["Seat Cover", "Toilets"]:
    main_path = os.path.join(DATASET_ROOT, main_cat)
    if not os.path.exists(main_path):
        continue

    for cls in os.listdir(main_path):
        cls_path = os.path.join(main_path, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        class_counts[cls] += len(images)

        for img_name in images:
            img = Image.open(os.path.join(cls_path, img_name))
            image_sizes.append(img.size)

# Safety check
if not image_sizes:
    raise RuntimeError("No images found under data/raw")

print("\nImages per class:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")

widths, heights = zip(*image_sizes)
print(f"\nImage width range: {min(widths)} – {max(widths)}")
print(f"Image height range: {min(heights)} – {max(heights)}")


# ================= 2. SAMPLE VISUALIZATION =================
print("\nShowing sample images...")

plt.figure(figsize=(8, 8))
plot_idx = 1

for main_cat in ["Seat Cover", "Toilets"]:
    main_path = os.path.join(DATASET_ROOT, main_cat)
    if not os.path.exists(main_path):
        continue

    for cls in os.listdir(main_path):
        cls_path = os.path.join(main_path, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if images:
            img_path = os.path.join(cls_path, random.choice(images))
            img = Image.open(img_path)

            plt.subplot(4, 3, plot_idx)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
            plot_idx += 1

plt.tight_layout()
plt.show()


# ================= 3. PREPROCESSING DEMO =================
print("\nPreprocessing demo: resize + pad")

# Pick one random image
sample_img_path = None
for root, _, files in os.walk(DATASET_ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            sample_img_path = os.path.join(root, f)
            break
    if sample_img_path:
        break

img = cv2.imread(sample_img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_padded = resize_and_pad(img, TARGET_SIZE)
img_padded_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_padded_rgb)
plt.title("Resized & Padded")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nSprint-1 Python script executed successfully.")
