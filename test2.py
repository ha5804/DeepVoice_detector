import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
import time
import numpy as np
from src.data_split import get_data
from src.dataset import VoiceDataset
from src.preprocess import AudioPreprocessor
from src.utils import load_config, extract_logmel
from src.models2 import DeepVoiceClassifier
from eval import plot_roc

cfg = load_config("config/config.yaml")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_, _, test = get_data("classifier", cfg["data"]["root"])
prep = AudioPreprocessor(cfg["data"]["sample_rate"])

dataset = VoiceDataset(
    test,
    prep,
    lambda x: torch.tensor(
        extract_logmel(x, cfg["data"]["sample_rate"], cfg["data"]["n_mels"]),
        dtype=torch.float32
    ).unsqueeze(0),
    cfg["data"]["segment_frames"] * cfg["data"]["hop_length"],
    "classifier_test"
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = DeepVoiceClassifier(
    cfg["data"]["n_mels"],
    cfg["model"]["hidden_dim"]
).to(device)

model.load_state_dict(torch.load(cfg["training"]["classifier_save_path"], map_location=device))
model.eval()

wav_scores = defaultdict(list)
wav_labels = {}

start_test = time.time()

with torch.no_grad():
    for x, label, wid in loader:
        x = x.to(device)
        prob = torch.softmax(model(x), dim=1)[0, 1].item()
        wav_scores[wid.item()].append(prob)
        wav_labels[wid.item()] = label.item()

total_test_time = time.time() - start_test
print("Total classifier test time:", total_test_time)

np.save("clf_test_time.npy", np.array([total_test_time]))


final_scores, final_labels = [], []
for wid, scores in wav_scores.items():
    final_scores.append(np.mean(sorted(scores, reverse=True)[:5]))
    final_labels.append(wav_labels[wid])

auc = plot_roc(np.array(final_scores), np.array(final_labels))
print("AUC:", auc)
