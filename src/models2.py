# src/model2.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepVoiceClassifier(nn.Module):
    def __init__(
        self,
        n_mels=80,
        lstm_hidden=128,
        num_classes=2
    ):
        super().__init__()

        # -------- CNN Encoder --------
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # -------- BiLSTM --------
        self.lstm = nn.LSTM(
            input_size=(n_mels // 4) * 32,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        # -------- Classifier --------
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        """
        x: (B, 1, n_mels, T)
        """

        # CNN
        x = self.conv(x)
        # (B, C, n_mels//4, T//4)

        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, C * F) # (B, T, feature)

        # LSTM
        out, _ = self.lstm(x)

        # 마지막 timestep
        out = out[:, -1, :]

        # Classifier
        logits = self.fc(out)
        return logits
