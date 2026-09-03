from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.nn import functional as functional

from .losses import hybrid_loss


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]


def _positive_weight(labels: torch.Tensor, maximum: float | None = None) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    weight = float(negatives / max(positives, 1))
    return min(weight, maximum) if maximum is not None else weight


def fit_autoencoder(model, features: torch.Tensor, labels: torch.Tensor, epochs: int, learning_rate: float) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    normal = features[labels == 0]
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstruction, _ = model(normal)
        loss = functional.mse_loss(reconstruction, normal)
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch + 1), "training_loss": float(loss.detach())})
    return TrainingResult(epochs, history[-1]["training_loss"], history)


def fit_gnn(
    model,
    train_features: torch.Tensor,
    train_edges: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_edges: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_start: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    positive_weight_cap: float | None = None,
) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_weight = torch.tensor([_positive_weight(train_labels, positive_weight_cap)], device=train_features.device, dtype=train_features.dtype)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_features, train_edges)
        train_loss = functional.binary_cross_entropy_with_logits(logits, train_labels.float(), pos_weight=positive_weight)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_features, validation_edges)[validation_start:]
            validation_loss = functional.binary_cross_entropy_with_logits(validation_logits, validation_labels.float(), pos_weight=positive_weight)
        record = {"epoch": float(epoch), "training_loss": float(train_loss.detach()), "validation_loss": float(validation_loss.detach())}
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, best_loss, history)


def fit_hybrid(
    model,
    train_features: torch.Tensor,
    train_edges: torch.Tensor,
    train_labels: torch.Tensor,
    validation_features: torch.Tensor,
    validation_edges: torch.Tensor,
    validation_labels: torch.Tensor,
    validation_start: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    reconstruction_weight: float,
    patience: int,
    positive_weight_cap: float | None = None,
) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positive_weight = _positive_weight(train_labels, positive_weight_cap)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(train_features, train_edges)
        train_loss, _ = hybrid_loss(train_outputs, train_features, train_labels, positive_weight, reconstruction_weight)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_outputs_all = model(validation_features, validation_edges)
            validation_outputs = {key: value[validation_start:] for key, value in validation_outputs_all.items()}
            validation_loss, _ = hybrid_loss(validation_outputs, validation_features[validation_start:], validation_labels, positive_weight, reconstruction_weight)
        record = {"epoch": float(epoch), "training_loss": float(train_loss.detach()), "validation_loss": float(validation_loss.detach())}
        history.append(record)
        if record["validation_loss"] < best_loss:
            best_loss = record["validation_loss"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, best_loss, history)
