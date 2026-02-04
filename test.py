# test.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from src.dataset import SegmentVoiceDataset
from src.models import LSTMAutoEncoder
from src.utils import load_config
from eval import plot_roc


def test():
    # ---------- config ----------
    cfg = load_config("config/config.yaml")

    # ---------- device ----------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # ---------- dataset / dataloader ----------
    test_dataset = SegmentVoiceDataset(cfg, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    # ---------- model ----------
    model = LSTMAutoEncoder(
        n_mels=cfg["data"]["n_mels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        num_layers=cfg["model"]["num_layers"]
    ).to(device)

    model.load_state_dict(
        torch.load(cfg["training"]["save_path"], map_location=device)
    )
    model.eval()

    # ---------- loss ----------
    criterion = nn.MSELoss(reduction="none")

    # ---------- containers ----------
    wav_scores = defaultdict(list)   # wav_id -> list of segment scores
    wav_labels = {}                  # wav_id -> label (0/1)

    # ---------- inference ----------
    with torch.no_grad():
        pbar = tqdm(
            test_loader,
            desc="Testing (segment-level)",
            total=len(test_loader)
        )

        for x, label, wav_id in pbar:
            x = x.to(device)

            x_hat = model(x)

            # reconstruction error (segment-level)
            loss = criterion(x_hat, x)
            score = loss.mean().item()

            wid = wav_id.item()
            wav_scores[wid].append(score)
            wav_labels[wid] = label.item()

            pbar.set_postfix(score=score)

    # ---------- wav-level aggregation ----------
    final_scores = []
    final_labels = []

    for wid, scores in wav_scores.items():
        # top-k mean (k=5)
        scores = sorted(scores, reverse=True)
        k = min(5, len(scores))
        wav_score = np.mean(scores[:k])

        final_scores.append(wav_score)
        final_labels.append(wav_labels[wid])

    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)

    # ---------- stats ----------
    print("\nTest finished (wav-level).")
    print("Total wavs:", len(final_scores))
    print("Score stats:")
    print("  mean:", final_scores.mean())
    print("  std :", final_scores.std())
    print("Normal mean:", final_scores[final_labels == 0].mean())
    print("Fake mean  :", final_scores[final_labels == 1].mean())

    # ---------- ROC ----------
    auc_score = plot_roc(final_scores, final_labels)
    print("AUC:", auc_score)

    return final_scores, final_labels


if __name__ == "__main__":
    test()

