import torch
from torch import nn

from .graphsage import GraphSAGEEncoder


class GNNFraudClassifier(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dimension, hidden_dimension, layers, dropout)
        self.tabular_projection = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        graph_embedding = self.encoder(features, edge_index)
        tabular_embedding = self.tabular_projection(features)
        return self.classifier(torch.cat([graph_embedding, tabular_embedding], dim=1)).squeeze(1)
