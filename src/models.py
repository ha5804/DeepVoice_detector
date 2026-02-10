# src/models.py
import torch
import torch.nn as nn

class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        n_mels=80,
        hidden_dim=128,
        latent_dim=32,
        num_layers=2
    ):
        super().__init__()

        # -------- Encoder --------
        self.encoder = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # -------- Decoder --------
        self.from_latent = nn.Linear(latent_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=n_mels,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        """
        x: (B, n_mels, T)
        return: reconstructed x (B, n_mels, T)
        """

        # (B, n_mels, T) → (B, T, n_mels)
        x = x.permute(0, 2, 1)

        # -------- Encoder --------
        enc_out, (h_n, _) = self.encoder(x)
        # h_n: (num_layers, B, hidden_dim)

        h_last = h_n[-1]                 # (B, hidden_dim)
        z = self.to_latent(h_last)       # (B, latent_dim)

        # -------- Decoder --------
        h_dec = self.from_latent(z)      # (B, hidden_dim)

        # repeat for each timestep
        T = x.size(1)
        h_dec_seq = h_dec.unsqueeze(1).repeat(1, T, 1)

        recon, _ = self.decoder(h_dec_seq)
        # recon: (B, T, n_mels)

        # (B, T, n_mels) → (B, n_mels, T)
        recon = recon.permute(0, 2, 1)

        return recon
