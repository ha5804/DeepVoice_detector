# src/dataset.py
import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset


class SegmentVoiceDataset(Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        self.sr = cfg["data"]["sample_rate"]
        self.segment_frames = cfg["data"]["segment_frames"]
        self.hop = 512
        self.max_files = cfg["data"].get("max_samples", None)

        # ---------- containers ----------
        self.items = []
        self.wav_cache = {}   # 🔥 wav 캐시

        if mode == "train":
            self._prepare_train(cfg["data"]["train_path"])
        else:
            self._prepare_test(cfg["data"]["test_path"])

        print(f"[Dataset] {mode} segments: {len(self.items)}")

    # ==================================================
    # Train: 정상 flac만 + 파일 단위 샘플링
    # ==================================================
    def _prepare_train(self, root):
        # 1️⃣ flac 파일 수집
        flac_files = []
        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(".flac"):
                    flac_files.append(os.path.join(r, f))

        # 2️⃣ 파일 단위 샘플링 (핵심)
        if self.max_files is not None and len(flac_files) > self.max_files:
            flac_files = np.random.choice(
                flac_files, size=self.max_files, replace=False
            )

        print(f"[Dataset] train wav files used: {len(flac_files)}")

        # 3️⃣ wav 로드 + segment 생성
        wav_id = 0
        for path in flac_files:
            wav, _ = librosa.load(path, sr=self.sr)
            self.wav_cache[path] = wav  # 🔥 캐시

            rms = librosa.feature.rms(y=wav, hop_length=self.hop)[0]
            n_frames = len(rms)

            # wav당 최대 5 segment (노트북 안전)
            n_segments = min(5, max(1, n_frames // self.segment_frames))

            for i in range(n_segments):
                self.items.append({
                    "path": path,
                    "label": 0,
                    "start": i * self.segment_frames,
                    "wav_id": wav_id
                })

            wav_id += 1

    # ==================================================
    # Test: flac = 정상, wav = 딥보이스
    # ==================================================
    def _prepare_test(self, root):
        wav_id = 0

        for r, _, files in os.walk(root):
            for f in files:
                if f.endswith(".flac"):
                    label = 0
                elif f.endswith(".wav"):
                    label = 1
                else:
                    continue

                path = os.path.join(r, f)
                wav, _ = librosa.load(path, sr=self.sr)
                self.wav_cache[path] = wav  # 🔥 캐시

                rms = librosa.feature.rms(y=wav, hop_length=self.hop)[0]
                n_frames = len(rms)

                # test는 파일당 최대 3 segment
                n_segments = min(3, max(1, n_frames // self.segment_frames))

                for i in range(n_segments):
                    self.items.append({
                        "path": path,
                        "label": label,
                        "start": i * self.segment_frames,
                        "wav_id": wav_id
                    })

                wav_id += 1

    # ==================================================
    # 🔥 빠른 Prosody Feature (pyin 제거)
    # ==================================================
    def extract_feature(self, wav):
        rms = librosa.feature.rms(y=wav)[0]
        zcr = librosa.feature.zero_crossing_rate(wav)[0]

        T = min(len(rms), len(zcr))
        rms, zcr = rms[:T], zcr[:T]

        feat = np.stack([
            rms,
            np.diff(rms, prepend=rms[0]),
            zcr,
            np.diff(zcr, prepend=zcr[0])
        ], axis=0)

        return torch.tensor(feat, dtype=torch.float32)

    # ==================================================
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        wav = self.wav_cache[item["path"]]   # 🔥 캐시 사용

        start = item["start"] * self.hop
        end = start + self.segment_frames * self.hop
        segment = wav[start:end]

        if len(segment) < self.segment_frames * self.hop:
            segment = np.pad(
                segment,
                (0, self.segment_frames * self.hop - len(segment))
            )

        feat = self.extract_feature(segment)

        if self.mode == "train":
            return feat
        else:
            return feat, item["label"], item["wav_id"]


