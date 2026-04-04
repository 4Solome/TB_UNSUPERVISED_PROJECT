import torch
import torch.nn as nn

class TTVAE(nn.Module):
    def __init__(
        self,
        D_in,
        latent_dim=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ff_dim=128,
        dropout=0.05
    ):
        super().__init__()

        self.feature_weight = nn.Parameter(torch.randn(D_in, d_model) * 0.02)
        self.feature_bias   = nn.Parameter(torch.zeros(D_in, d_model))
        self.pos = nn.Parameter(torch.randn(1, D_in, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mu = nn.Linear(d_model, latent_dim)
        self.logvar = nn.Linear(d_model, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, D_in),
            nn.Sigmoid()
        )

    def embed(self, x):
        t = (
            x.unsqueeze(-1)
            * self.feature_weight.unsqueeze(0)
            + self.feature_bias.unsqueeze(0)
        )
        return t + self.pos

    def encode(self, x):
        h = self.enc(self.embed(x))
        h = self.norm(h).mean(1)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logv):
        std = torch.exp(0.5 * logv)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparam(mu, logv)
        rec = self.decode(z)
        return rec, mu, logv
