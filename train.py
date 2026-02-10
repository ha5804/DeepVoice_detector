import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import numpy as np
from src.data_split import get_data
from src.dataset import VoiceDataset
from src.preprocess import AudioPreprocessor
from src.utils import load_config, extract_logmel
from src.models import LSTMAutoEncoder

cfg = load_config("config/config.yaml")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_files, _ = get_data("anomaly", cfg["data"]["root"])
prep = AudioPreprocessor(cfg["data"]["sample_rate"])

dataset = VoiceDataset(
    train_files,
    prep,
    lambda x: torch.tensor(
        extract_logmel(x, cfg["data"]["sample_rate"], cfg["data"]["n_mels"]),
        dtype=torch.float32
    ),
    cfg["data"]["segment_frames"] * cfg["data"]["hop_length"],
    "anomaly_train"
)

loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

model = LSTMAutoEncoder(
    cfg["data"]["n_mels"],
    cfg["model"]["hidden_dim"],
    cfg["model"]["latent_dim"],
    cfg["model"]["num_layers"]
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
loss_fn = nn.MSELoss()
epoch_times = []

for e in range(cfg["training"]["epochs"]):
    start = time.time()

    for x in loader:
        x = x.to(device)
        loss = loss_fn(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()

    elapsed = time.time() - start
    epoch_times.append(elapsed)

    print(f"[Epoch {e+1}] loss={loss.item():.4f}, time={elapsed:.2f}s")

total_train_time = sum(epoch_times)
print("Total AE train time:", total_train_time)

np.save("ae_train_times.npy", np.array(epoch_times))
