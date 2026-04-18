import torch
from torch.utils.data import DataLoader
from models.deepphys.dataset_loader import RPPGDataset
from models.deepphys.model import DeepPhysModel
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PATH
DATA_PATH = "/content/drive/MyDrive/data"

# Dataset
dataset = RPPGDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
model = DeepPhysModel().to(device)
model.train()

#LOWER LR(IMPORTANT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

#  LOOP
for epoch in range(3):
    print(f"\n===== Epoch {epoch} =====")

    epoch_loss = 0
    count = 0

    for batch_idx, (appearance, motion, signal) in enumerate(loader):

        print(f"\nProcessing video batch {batch_idx+1}")

        appearance = appearance.squeeze(0)
        motion = motion.squeeze(0)
        signal = signal.squeeze(0)

        max_windows = min(3, len(motion))

        for i in range(max_windows):

            print(f"  Window {i+1}/{max_windows}")

            a = appearance[i].float().to(device)
            m = motion[i].float().to(device)
            s = signal[i].float().to(device)

            #NORMALIZE SIGNAL 
            s = (s - s.mean()) / (s.std() + 1e-6)

            try:
               
                output = model(a, m)

                loss = criterion(output, s)

                optimizer.zero_grad()
                loss.backward()

                #GRADIENT CLIPPING
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                epoch_loss += loss.item()
                count += 1

            except Exception as e:
                print("Error:", e)
                continue

    print(f"\nEpoch {epoch} Avg Loss: {epoch_loss / count:.4f}")