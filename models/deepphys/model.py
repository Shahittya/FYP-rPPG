import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPhysLSTM(nn.Module):
    def __init__(self):
        super(DeepPhysLSTM, self).__init__()

        # Motion stream
        self.motion_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Appearance stream
        self.appearance_stream = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM (temporal modeling)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=128, 
            num_layers=1,
            batch_first=True
        )

        # Final regression
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, appearance, motion):
        """
        appearance: (T, H, W, C)
        motion:     (T, H, W, C)
        """

        # Convert to (T, C, H, W)
        appearance = appearance.permute(0, 3, 1, 2)
        motion = motion.permute(0, 3, 1, 2)

        # CNN feature extraction
        m = self.motion_stream(motion)
        a = self.appearance_stream(appearance)

        # Element-wise fusion
        x = m * a  # (T, 64, H, W)

        # Global pooling → (T, 64)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Add batch dimension → (1, T, 64)
        x = x.unsqueeze(0)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (1, T, 128)

        # Remove batch → (T, 128)
        lstm_out = lstm_out.squeeze(0)

        # Final prediction → (T,)
        out = self.fc(lstm_out).squeeze(-1)

        return out