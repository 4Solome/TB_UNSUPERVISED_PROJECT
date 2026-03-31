import torch
import torch.nn as nn

class TTVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, d_model=64, n_heads=4, 
                 n_layers=2, ff_dim=128, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.feature_weight = nn.Parameter(torch.randn(input_dim, d_model)*0.02)
        self.feature_bias   = nn.Parameter(torch.zeros(input_dim, d_model))
        self.pos_embedding  = nn.Parameter(torch.randn(1, input_dim, d_model)*0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        self.mu_layer     = nn.Linear(d_model, latent_dim)
        self.logvar_layer = nn.Linear(d_model, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def embed(self, x):
        tokens = x.unsqueeze(-1)*self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        return tokens + self.pos_embedding

    def encode(self, x):
        h = self.encoder(self.embed(x))
        h = self.norm(h).mean(1)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rec = self.decode(z)
        return rec, mu, logvar
