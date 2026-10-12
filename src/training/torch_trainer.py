from copy import deepcopy
from dataclasses import dataclass
import time

import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as functional

from .losses import hybrid_loss


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]]
    timing: dict[str, float] | None = None


def _positive_weight(labels: torch.Tensor, maximum: float | None = None) -> float:
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    weight = float(negatives / max(positives, 1))
    return min(weight, maximum) if maximum is not None else weight


def fit_autoencoder(model, features: torch.Tensor, labels: torch.Tensor, epochs: int, learning_rate: float, validation_features=None, validation_labels=None, batch_size=1024, patience=10) -> TrainingResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    normal = features[labels == 0]
    if validation_features is None or validation_labels is None:
        raise ValueError("Autoencoder requires a separate validation partition")
    validation_normal = validation_features[validation_labels == 0]
    if not len(normal) or not len(validation_normal):
        raise ValueError("Normal transactions required in train and validation")
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = deepcopy(model.state_dict())
    history = []
    train_seconds = validation_seconds = 0.0
    for epoch in range(epochs):
        if features.is_cuda:
            torch.cuda.synchronize()
        tick = time.perf_counter()
        model.train()
        total = 0.0
        for indices in torch.randperm(len(normal), device=normal.device).split(batch_size):
            batch = normal[indices]
            optimizer.zero_grad()
            reconstruction, _ = model(batch)
            loss = functional.mse_loss(reconstruction, batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        if features.is_cuda:
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        model.eval()
        with torch.no_grad():
            val_loss = sum(float(functional.mse_loss(model(batch)[0], batch)) * len(batch) for batch in validation_normal.split(batch_size)) / len(validation_normal)
        if features.is_cuda:
            torch.cuda.synchronize()
        validation_seconds += time.perf_counter() - tick
        history.append({"epoch": epoch + 1, "training_loss": total / len(normal), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, best_loss, history, {"train_seconds": train_seconds, "validation_seconds": validation_seconds})


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
    best_ap = -1.0
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
        ap = float(average_precision_score(validation_labels.cpu().numpy(), validation_logits.detach().cpu().numpy()))
        record["validation_ap"] = ap
        if ap > best_ap:
            best_ap = ap
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
    best_ap = -1.0
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
        ap = float(average_precision_score(validation_labels.cpu().numpy(), validation_outputs["logits"].detach().cpu().numpy()))
        record["validation_ap"] = ap
        if ap > best_ap:
            best_ap = ap
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
