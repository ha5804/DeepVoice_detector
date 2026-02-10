import torch
from torch.utils.data import Dataset

class VoiceDataset(Dataset):
    def __init__(self, file_list, preprocessor, feature_fn, segment_len, mode):
        self.items = []
        self.wav_cache = {}
        self.feature_fn = feature_fn
        self.segment_len = segment_len
        self.mode = mode

        for wid, item in enumerate(file_list):
            if mode == "anomaly_train":
                path, label = item, 0
            else:
                path, label = item

            wav = preprocessor.preprocess(path)
            self.wav_cache[path] = wav

            n_seg = len(wav) // segment_len
            for i in range(n_seg):
                self.items.append((path, label, i, wid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, seg_idx, wid = self.items[idx]
        wav = self.wav_cache[path]

        seg = wav[
            seg_idx*self.segment_len:(seg_idx+1)*self.segment_len
        ]

        feat = self.feature_fn(seg)

        if self.mode == "anomaly_train":
            return feat
        elif "test" in self.mode:
            return feat, label, wid
        else:
            return feat, label




