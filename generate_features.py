import os
import numpy as np
from PIL import Image

# ================= CONFIGURE =================
DATASET_FOLDER = r"C:\Users\saniy\Downloads\EEG seizure and non-seizure image dataset"
OUTPUT_FILE = os.path.join(DATASET_FOLDER, "features.npy")
IMAGE_SIZE = (128, 128)  # Resize all images to 128x128

# ================= COLLECT FEATURES =================
features = []

for root, dirs, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path).convert('L')  # Grayscale
                img = img.resize(IMAGE_SIZE)
                arr = np.array(img).flatten() / 255.0  # Normalize
                features.append(arr)
            except Exception as e:
                print(f"Skipping {file_path}, error: {e}")

features = np.array(features)
np.save(OUTPUT_FILE, features)
print(f"✅ Features saved to {OUTPUT_FILE}")
print(f"Total images processed: {len(features)}")
