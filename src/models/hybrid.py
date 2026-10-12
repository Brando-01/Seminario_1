import torch
from torch import nn

from .autoencoder import Autoencoder
from .graphsage import GraphSAGEEncoder


class HybridGNNAutoencoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, latent_dimension: int, gnn_layers: int = 2, dropout: float = 0.3, attention: bool = False, normalize_fusion: bool = False):
        super().__init__()
        self.autoencoder = Autoencoder(input_dimension, hidden_dimension, latent_dimension, dropout, attention)
        self.gnn = GraphSAGEEncoder(input_dimension, hidden_dimension, gnn_layers, dropout)
        self.tabular_projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_dimension = latent_dimension + hidden_dimension * 2 + 1
        self.fusion_normalization = nn.LayerNorm(fusion_dimension) if normalize_fusion else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor, first_neighbor_mean: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        reconstruction, latent = self.autoencoder(features)
        graph_embedding = self.gnn(features, edge_index, first_neighbor_mean)
        tabular_embedding = self.tabular_projection(features)
        reconstruction_error = torch.mean(torch.abs(features - reconstruction), dim=1, keepdim=True)
        fused = torch.cat([latent, graph_embedding, tabular_embedding, reconstruction_error], dim=1)
        logits = self.classifier(self.fusion_normalization(fused)).squeeze(1)
        return {
            "logits": logits,
            "reconstruction": reconstruction,
            "latent": latent,
            "graph_embedding": graph_embedding,
            "tabular_embedding": tabular_embedding,
            "reconstruction_error": reconstruction_error.squeeze(1),
        }
