import os
import numpy as np
from PIL import Image
import imagehash

# ================= CONFIGURE =================
DATASET_FOLDER = r"C:\Users\saniy\Downloads\EEG seizure and non-seizure image dataset"
OUTPUT_FILE = os.path.join(DATASET_FOLDER, "hashes.npy")
IMAGE_SIZE = (128, 128)  # Resize all images to 128x128

# ================= GENERATE HASHES =================
hashes = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path).convert('L')
                img = img.resize(IMAGE_SIZE)
                phash = str(imagehash.phash(img))
                hashes.append(phash)
            except Exception as e:
                print(f"Skipping {file_path}, error: {e}")

hashes = np.array(hashes)
np.save(OUTPUT_FILE, hashes)

print(f"✅ Perceptual hashes saved to {OUTPUT_FILE}")
print(f"Total images processed: {len(hashes)}")
