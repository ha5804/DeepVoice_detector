import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from src.data_split import get_data
from src.dataset import VoiceDataset
from src.preprocess import AudioPreprocessor
from src.utils import load_config, extract_logmel
from src.models2 import DeepVoiceClassifier

cfg = load_config("config/config.yaml")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train, _, _ = get_data("classifier", cfg["data"]["root"])
prep = AudioPreprocessor(cfg["data"]["sample_rate"])

dataset = VoiceDataset(
    train,
    prep,
    lambda x: torch.tensor(
        extract_logmel(x, cfg["data"]["sample_rate"], cfg["data"]["n_mels"]),
        dtype=torch.float32
    ).unsqueeze(0),
    cfg["data"]["segment_frames"] * cfg["data"]["hop_length"],
    "classifier"
)

loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

model = DeepVoiceClassifier(
    cfg["data"]["n_mels"],
    cfg["model"]["hidden_dim"]
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
loss_fn = nn.CrossEntropyLoss()

epoch_times = []

for e in range(cfg["training"]["epochs"]):
    start = time.time()
    correct = total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    elapsed = time.time() - start
    epoch_times.append(elapsed)

    acc = correct / total if total > 0 else 0.0
    print(f"[Epoch {e+1}] acc={acc:.4f}, time={elapsed:.2f}s")


np.save("clf_train_times.npy", np.array(epoch_times))

os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), cfg["training"]["classifier_save_path"])
