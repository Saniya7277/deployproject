import os
from PIL import Image
import imagehash
import numpy as np

DATASET_FOLDER = r"C:\Users\saniy\Downloads\EEG seizure and non-seizure image dataset"
OUTPUT_FILE = os.path.join(DATASET_FOLDER, "hashes.npy")  # Save hashes here

hashes = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file)
            try:
                img = Image.open(path).convert("L")  # Grayscale
                h = imagehash.phash(img)             # Perceptual hash
                hashes.append(str(h))
            except Exception as e:
                print(f"Skipping {path}: {e}")

hashes = np.array(hashes)
np.save(OUTPUT_FILE, hashes)
print(f"✅ Hashes saved to {OUTPUT_FILE}, total images: {len(hashes)}")
