import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, latent_dimension: int, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, latent_dimension),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, input_dimension),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(features)
        reconstruction = self.decoder(latent)
        return reconstruction, latent

