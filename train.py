import torch
from torch.utils.data import DataLoader
from models.deepphys.dataset_loader import RPPGDataset
from models.deepphys.model import DeepPhysModel
import torch.nn as nn

# Load dataset
dataset = RPPGDataset("data/videos")

# DataLoader
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
model = DeepPhysModel()

# Optimizer + Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(2):
    for motion, signal in loader:

        motion = motion.squeeze(0)   # (W, T, H, W, C)
        signal = signal.squeeze(0)

        motion = motion.float()
        signal = signal.float()

        max_windows = min(3, len(motion))   # 🔥 VERY SAFE LIMIT

        for i in range(max_windows):

            m = motion[i]   # (T, H, W, C)
            s = signal[i]   # (T,)

            try:
                output = model(m, m)
                loss = criterion(output, s)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            except Exception as e:
                print("Error in window:", e)
                continue

    print(f"Epoch {epoch}, Loss: {loss.item()}")