import os
import numpy as np
from models.deepphys.dataset_loader import RPPGDataset

DATA_PATH = "/content/drive/MyDrive/FYP/data/data/videos"
SAVE_PATH = "/content/drive/MyDrive/FYP/processed"
os.makedirs(SAVE_PATH, exist_ok=True)
dataset = RPPGDataset(DATA_PATH)
print("Total videos:", len(dataset))

for i in range(len(dataset)):
    print(f"\nProcessing video {i+1}/{len(dataset)}")

    try:
        appearance, motion, signal = dataset[i]
        save_file = os.path.join(SAVE_PATH, f"sample_{i}.npz")
        np.savez_compressed(
            save_file,
            appearance=appearance,
            motion=motion,
            signal=signal
        )

    except Exception as e:
        print("Error:", e)
        continue

print("\nPreprocessing DONE!")