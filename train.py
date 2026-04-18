import torch
from torch.utils.data import DataLoader
from models.deepphys.dataset_loader import RPPGDataset
from models.deepphys.model import DeepPhysModel
import torch.nn as nn

# 🔥 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔥 DATA PATH (COLAB)
DATA_PATH = "/content/drive/MyDrive/data"

# Dataset
dataset = RPPGDataset(DATA_PATH)

# 🔥 IMPORTANT: keep batch_size = 1 for now
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
model = DeepPhysModel().to(device)
model.train()

# Optimizer + Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 🔥 TRAIN LOOP (SAFE VERSION)
for epoch in range(2):
    print(f"\n===== Epoch {epoch} =====")

    for batch_idx, (appearance, motion, signal) in enumerate(loader):

        print(f"\nProcessing video batch {batch_idx+1}")

        # remove batch dimension
        motion = motion.squeeze(0)   # (W, T, H, W, C)
        signal = signal.squeeze(0)   # (W, T)
        appearance = appearance.squeeze(0) # (W, T, H, W, C)

        # 🔥 LIMIT WINDOWS (VERY IMPORTANT)
        max_windows = min(2, len(motion))

        for i in range(max_windows):

            print(f"  Window {i+1}/{max_windows}")

            # move small chunk to GPU
            a = appearance[i].float().to(device)
            m = motion[i].float().to(device)
            s = signal[i].float().to(device)
            output = model(a, m)

            try:
                # TEMP: motion used as appearance too
                output = model(m, m)

                loss = criterion(output, s)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except Exception as e:
                print("Error in window:", e)
                continue

    print(f"\nEpoch {epoch} completed | Loss: {loss.item()}")