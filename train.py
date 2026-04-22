import torch
from torch.utils.data import DataLoader
from models.deepphys.dataset_loader import RPPGDataset
from models.deepphys.model import DeepPhysModel
import torch.nn as nn
from torch.utils.data import random_split
from dataset_fast import RPPGFastDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PATH
DATA_PATH = "/content/drive/MyDrive/FYP/processed"

# Dataset
dataset = RPPGFastDataset(DATA_PATH)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
# Model
model = DeepPhysModel().to(device)
model.train()

#LOWER LR(IMPORTANT)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()
best_val_loss = float("inf")

#  LOOP
EPOCHS = 5

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch} =====")

    model.train()

    epoch_loss = 0
    count = 0

    # TRAINING LOOP 
    for batch_idx, (appearance, motion, signal) in enumerate(train_loader):

        print(f"\nProcessing video batch {batch_idx+1}")

        appearance = appearance.squeeze(0)
        motion = motion.squeeze(0)
        signal = signal.squeeze(0)

        max_windows = min(10, len(motion))

        a_batch = appearance[:max_windows].float().to(device)
        m_batch = motion[:max_windows].float().to(device)
        s_batch = signal[:max_windows].float().to(device)

        B, T, H, W, C = a_batch.shape

        a_batch = a_batch.view(B * T, H, W, C)
        m_batch = m_batch.view(B * T, H, W, C)
        s_batch = s_batch.view(B * T)

        

        try:
            output = model(a_batch, m_batch)

            loss = criterion(output, s_batch)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        except Exception as e:
            print("Error:", e)
            continue

    print(f"\nEpoch {epoch} Avg Loss: {epoch_loss / count:.4f}")

    # VALIDATION
    model.eval()
    val_loss = 0
    val_count = 0

    with torch.no_grad():
        for appearance, motion, signal in val_loader:

            appearance = appearance.squeeze(0)
            motion = motion.squeeze(0)
            signal = signal.squeeze(0)

            max_windows = min(3, len(motion))

            for i in range(max_windows):

                a = appearance[i].float().to(device)
                m = motion[i].float().to(device)
                s = signal[i].float().to(device)
                output = model(a, m)
                loss = criterion(output, s)

                val_loss += loss.item()
                val_count += 1

    val_loss = val_loss / val_count
    print(f"Validation Loss: {val_loss:.4f}")

    #  BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

    model.train()