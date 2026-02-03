# src/dataset.py
import os
import random
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        """
        mode: 'train' or 'test'
        """
        self.cfg = cfg
        self.mode = mode
        self.samples = []

        if mode == "train":
            self._load_train_data()
        elif mode == "test":
            self._load_test_data()
        else:
            raise ValueError("mode must be 'train' or 'test'")

    def _load_train_data(self):
        root = self.cfg["data"]["train_clean_path"]
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(".flac"):
                    self.samples.append(os.path.join(r, f))

    def _load_test_data(self):
        clean_root = self.cfg["data"]["test_clean_path"]
        anomaly_root = self.cfg["data"]["test_anomaly_path"]

        for r, _, files in os.walk(clean_root):
            for f in files:
                if f.endswith(".flac"):
                    self.samples.append((os.path.join(r, f), 0))

        for r, _, files in os.walk(anomaly_root):
            for f in files:
                if f.endswith(".wav"):
                    self.samples.append((os.path.join(r, f), 1))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _crop_or_pad(self, mel):
        """
        mel: (n_mels, T)
        return: (n_mels, fixed_T)
        """
        target_T = self.cfg["data"]["segment_frames"]
        T = mel.shape[1]

        if T > target_T:
            start = random.randint(0, T - target_T)
            mel = mel[:, start:start + target_T]
        else:
            pad_width = target_T - T
            mel = np.pad(
                mel,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant"
            )
        return mel

    def __getitem__(self, idx):
        # idx번째 샘플 반환
        if self.mode == "train":
            path = self.samples[idx]
            label = None
        else:
            path, label = self.samples[idx]

        # 음성 로드
        wav, sr = librosa.load(
            path,
            sr=self.cfg["data"]["sample_rate"]
        )

        # mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
            n_mels=self.cfg["data"]["n_mels"]
        )
        mel = librosa.power_to_db(mel)

        # segment
        mel = self._crop_or_pad(mel)

        feature = torch.tensor(mel, dtype=torch.float32)

        if self.mode == "train":
            return feature
        else:
            return feature, torch.tensor(label)
