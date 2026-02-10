import yaml
import librosa
import numpy as np

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def extract_logmel(wav, sr, n_mels):
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=n_mels
    )
    return librosa.power_to_db(mel)


