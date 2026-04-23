import torch
from torch.utils.data import DataLoader, random_split
#from models.deepphys.model import DeepPhysModel
from dataset_fast import RPPGFastDataset
import torch.nn as nn
from models.deepphys.model import DeepPhysLSTMAttention

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA PATH
DATA_PATH = "/content/drive/MyDrive/ML/FYP/processed"

# DATASET
dataset = RPPGFastDataset(DATA_PATH)

# Train / Val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
#model = DeepPhysModel().to(device)
model = DeepPhysLSTMAttention().to(device)
# OPTIMIZER + LOSS
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.MSELoss()

best_val_loss = float("inf")


EPOCHS = 15



for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch} =====")

    model.train()
    epoch_loss = 0
    count = 0

    for batch_idx, (appearance, motion, signal) in enumerate(train_loader):

        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx+1}")

        # remove batch dim
        appearance = appearance.squeeze(0)
        motion = motion.squeeze(0)
        signal = signal.squeeze(0)

        max_windows = min(15, len(motion))

        #LOOP THROUGH WINDOWS
        for i in range(max_windows):

            a = appearance[i].float().to(device)   # (T, H, W, C)
            m = motion[i].float().to(device)
            s = signal[i].float().to(device)       # (T,)

            try:
                output = model(a, m)

                loss = criterion(output, s)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                count += 1

            except Exception as e:
                print("Error:", e)
                continue

    print(f"Epoch {epoch} Avg Loss: {epoch_loss / count:.4f}")


    model.eval()
    val_loss = 0
    val_count = 0

    with torch.no_grad():
        for appearance, motion, signal in val_loader:

            appearance = appearance.squeeze(0)
            motion = motion.squeeze(0)
            signal = signal.squeeze(0)

            max_windows = min(10, len(motion))

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

    # SAVE BEST MODEL
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved!")

    model.train()