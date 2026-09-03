import torch
from torch.nn import functional as functional


def hybrid_loss(outputs: dict[str, torch.Tensor], features: torch.Tensor, labels: torch.Tensor, positive_weight: float, reconstruction_weight: float) -> tuple[torch.Tensor, dict[str, float]]:
    weight = torch.tensor([positive_weight], device=features.device, dtype=features.dtype)
    classification = functional.binary_cross_entropy_with_logits(outputs["logits"], labels.float(), pos_weight=weight)
    normal_mask = labels == 0
    reconstruction = functional.mse_loss(outputs["reconstruction"][normal_mask], features[normal_mask]) if normal_mask.any() else torch.zeros((), device=features.device)
    total = classification + reconstruction_weight * reconstruction
    return total, {"classification": float(classification.detach()), "reconstruction": float(reconstruction.detach())}

