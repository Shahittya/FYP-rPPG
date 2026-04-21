import os
import numpy as np
from torch.utils.data import Dataset

class RPPGFastDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f)
                      for f in os.listdir(data_dir)
                      if f.endswith(".npz")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        appearance = data["appearance"]
        motion = data["motion"]
        signal = data["signal"]

        return appearance, motion, signal