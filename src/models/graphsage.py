import torch
from torch import nn


class GraphSAGELayer(nn.Module):
    """Mean-aggregation GraphSAGE layer implemented with native PyTorch."""

    def __init__(self, input_dimension: int, output_dimension: int):
        super().__init__()
        self.linear = nn.Linear(input_dimension * 2, output_dimension)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        aggregated = torch.zeros_like(features)
        degree = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
        if edge_index.numel() > 0:
            aggregated.index_add_(0, destination, features[source])
            degree.index_add_(0, destination, torch.ones_like(destination, dtype=features.dtype))
        aggregated = aggregated / degree.clamp_min(1).unsqueeze(1)
        return self.linear(torch.cat([features, aggregated], dim=1))


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if layers < 1:
            raise ValueError("GraphSAGE requires at least one layer")
        dimensions = [input_dimension] + [hidden_dimension] * layers
        self.layers = nn.ModuleList(GraphSAGELayer(dimensions[i], dimensions[i + 1]) for i in range(layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = features
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index < len(self.layers) - 1:
                hidden = self.dropout(torch.relu(hidden))
        return hidden

