# src/dataset.py
import os
import random
import librosa
import torch
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
        #cfg는 파이썬 딕셔너리. config.yaml의 내용을 딕셔너리로 생성
        root = self.cfg["data"]["train_clean_path"]
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".flac"):
                    self.samples.append(r + "/" + f)

    def _load_test_data(self):
        clean_root = self.cfg["data"]["test_clean_path"]
        anomaly_root = self.cfg["data"]["test_anomaly_path"]

        for r, _, files in os.walk(clean_root):
            for f in files:
                if f.endswith(".flac"):
                    self.samples.append((r + "/" + f, 0))

        for r, _, files in os.walk(anomaly_root):
            for f in files:
                if f.endswith(".wav"):
                    self.samples.append((r + "/" + f, 1))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "train":
            path = self.samples[idx]
            label = None
        else:
            path, label = self.samples[idx]

        wav, sr = librosa.load(
            path,
            sr=self.cfg["data"]["sample_rate"]
        )

        # feature extraction (on-the-fly)
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=sr,
        )