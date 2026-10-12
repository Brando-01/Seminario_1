"""Supervised target minibatches with exact causal graph context."""

from copy import deepcopy
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from src.graph.subgraphs import IncomingNeighborhoods
from .losses import hybrid_loss
from .torch_trainer import TrainingResult, _positive_weight


def _batch_tensors(features, means, index, targets, depth, device):
    # The raw neighbor mean replaces one traversal with an exact cached value.
    batch = index.extract(targets, max(0, depth - 1))
    return (
        torch.as_tensor(features[batch.nodes], device=device),
        torch.as_tensor(batch.edge_index, device=device),
        torch.as_tensor(batch.target_positions, device=device),
        torch.as_tensor(means[batch.nodes], device=device),
    )


def predict_hybrid_batches(model, features, index, targets, depth, batch_size=8192, means=None):
    device = next(model.parameters()).device
    scores = []
    model.eval()
    means = index.mean_features(features) if means is None else means
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            x, edges, positions, first_mean = _batch_tensors(features, means, index, targets[start:start + batch_size], depth, device)
            scores.append(model(x, edges, first_mean)["logits"][positions].cpu().numpy())
    return np.concatenate(scores)


def fit_hybrid_batches(model, prepared, epochs, batch_size=4096, inference_batch_size=8192):
    if epochs < 1 or batch_size < 1 or inference_batch_size < 1:
        raise ValueError("Epochs and batch sizes must be positive")
    cfg = prepared.config["training"]
    depth = prepared.config["model"]["gnn_layers"]
    device = next(model.parameters()).device
    train_index = IncomingNeighborhoods(prepared.graphs["train"].edge_index, len(prepared.features["train"]))
    validation_index = IncomingNeighborhoods(prepared.graphs["validation"].edge_index, len(prepared.graph_features["validation"]))
    validation_start = len(prepared.graph_features["validation"]) - len(prepared.labels["validation"])
    validation_targets = np.arange(validation_start, len(prepared.graph_features["validation"]))
    train_means = train_index.mean_features(prepared.features["train"])
    validation_means = validation_index.mean_features(prepared.graph_features["validation"])
    train_labels = prepared.labels["train"]
    positive_weight = _positive_weight(torch.as_tensor(train_labels), cfg.get("positive_weight_cap"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    best_ap, best_epoch, stale, updates = -1.0, 0, 0, 0
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history = []
    train_seconds = validation_seconds = 0.0
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        model.train()
        total_loss = 0.0
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), batch_size):
            targets = order[start:start + batch_size]
            x, edges, positions, first_mean = _batch_tensors(prepared.features["train"], train_means, train_index, targets, depth, device)
            labels = torch.as_tensor(train_labels[targets], device=device)
            optimizer.zero_grad()
            output = model(x, edges, first_mean)
            target_outputs = {name: value[positions] for name, value in output.items()}
            loss, _ = hybrid_loss(target_outputs, x[positions], labels, positive_weight, cfg["reconstruction_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            updates += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        logits = predict_hybrid_batches(model, prepared.graph_features["validation"], validation_index, validation_targets, depth, inference_batch_size, validation_means)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        validation_seconds += time.perf_counter() - tick
        ap = float(average_precision_score(prepared.labels["validation"], logits))
        val_loss = float(F.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.as_tensor(prepared.labels["validation"], dtype=torch.float32), pos_weight=torch.tensor(positive_weight)))
        history.append({"epoch": epoch, "training_loss": total_loss / len(order),
                        "validation_loss": val_loss, "validation_ap": ap, "optimizer_steps": updates})
        print(f"  epoch={epoch} val AP={100 * ap:.2f}% updates={updates}", flush=True)
        if ap > best_ap:
            best_ap, best_epoch, best_loss, stale = ap, epoch, val_loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, best_loss, history), {
        "train_seconds": train_seconds, "validation_seconds": validation_seconds,
        "optimizer_steps": updates, "updates_per_epoch": int(np.ceil(len(train_labels) / batch_size)),
        "positive_weight": positive_weight, "validation_ap": best_ap,
    }
