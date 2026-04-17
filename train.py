import torch
from torch.utils.data import DataLoader
from models.deepphys.dataset_loader import RPPGDataset
from models.deepphys.model import DeepPhysModel
import torch.nn as nn

# 🔥 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
DATA_PATH = "/content/drive/MyDrive/data"  # change in Colab
dataset = RPPGDataset(DATA_PATH)

loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
model = DeepPhysModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 🔥 TRAIN LOOP (SAFE VERSION)
for epoch in range(2):
    for motion, signal in loader:

        motion = motion.squeeze(0).float().to(device)   # (W, T, H, W, C)
        signal = signal.squeeze(0).float().to(device)   # (W, T)

        # 🔥 LIMIT WINDOWS (VERY IMPORTANT)
        max_windows = min(3, len(motion))

        for i in range(max_windows):

            m = motion[i]   # (T, H, W, C)
            s = signal[i]   # (T,)

            output = model(m, m)   # TEMP (will improve later)

            loss = criterion(output, s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")