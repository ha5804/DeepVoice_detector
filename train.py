# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import SegmentVoiceDataset
from src.models import LSTMAutoEncoder
from src.utils import load_config
import os

def train():
    # ---------- config ----------
    cfg = load_config("config/config.yaml")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)


    # ---------- dataset / dataloader ----------
    train_dataset = SegmentVoiceDataset(cfg, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True
    )

    # ---------- model ----------
    model = LSTMAutoEncoder(
        n_mels=cfg["data"]["n_mels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        num_layers=cfg["model"]["num_layers"]
    ).to(device)

    # ---------- loss / optimizer ----------
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    # ---------- training loop ----------
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{cfg['training']['epochs']}]",
            total=len(train_loader)
        )
        for x in train_loader:
            x = x.to(device)

            x_hat = model(x)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}] loss: {epoch_loss:.6f}")

    save_path = cfg["training"]["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    # ---------- save ----------
    torch.save(model.state_dict(), cfg["training"]["save_path"])
    print("Model saved.")

if __name__ == "__main__":
    train()
