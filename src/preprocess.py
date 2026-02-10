import librosa
import numpy as np

class AudioPreprocessor:
    def __init__(self, sr):
        self.sr = sr

    def preprocess(self, path):
        wav, _ = librosa.load(path, sr=self.sr)
        return wav / (np.max(np.abs(wav)) + 1e-9)
