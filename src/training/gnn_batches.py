"""Exact target minibatches for the independent GraphSAGE classifier."""
from copy import deepcopy
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from src.graph.subgraphs import IncomingNeighborhoods
from .graph_batches import _batch_tensors
from .torch_trainer import TrainingResult, _positive_weight


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_gnn_batches(model, features, index, targets, depth, batch_size=8192, means=None):
    model.eval()
    device = next(model.parameters()).device
    means = index.mean_features(features) if means is None else means
    scores = []
    with torch.no_grad():
        for start in range(0, len(targets), batch_size):
            x, edges, positions, first_mean = _batch_tensors(features, means, index, targets[start:start + batch_size], depth, device)
            scores.append(model(x, edges, first_mean)[positions].cpu().numpy())
    return np.concatenate(scores)


def fit_gnn_batches(model, prepared, epochs, batch_size=4096, inference_batch_size=8192):
    if min(epochs, batch_size, inference_batch_size) < 1:
        raise ValueError("Positive epochs and batch sizes required")
    cfg = prepared.config["training"]
    depth = prepared.config["model"]["gnn_layers"]
    device = next(model.parameters()).device
    x = prepared.features["train"]
    vx = prepared.graph_features["validation"]
    y = prepared.labels["train"]
    index = IncomingNeighborhoods(prepared.graphs["train"].edge_index, len(x))
    vi = IncomingNeighborhoods(prepared.graphs["validation"].edge_index, len(vx))
    means, vm = index.mean_features(x), vi.mean_features(vx)
    vt = np.arange(len(vx) - len(prepared.labels["validation"]), len(vx))
    weight = _positive_weight(torch.as_tensor(y), cfg.get("positive_weight_cap"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    best_ap, best_epoch, stale, updates = -1.0, 0, 0, 0
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    history = []
    train_seconds = validation_seconds = 0.0
    for epoch in range(1, epochs + 1):
        synchronize(device); tick = time.perf_counter()
        model.train(); total = 0.0
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch_size):
            targets = order[start:start + batch_size]
            bx, edges, positions, first_mean = _batch_tensors(x, means, index, targets, depth, device)
            labels = torch.as_tensor(y[targets], device=device, dtype=torch.float32)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(bx, edges, first_mean)[positions], labels,
                                                       pos_weight=torch.tensor(weight, device=device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(targets); updates += 1
        synchronize(device); train_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        logits = predict_gnn_batches(model, vx, vi, vt, depth, inference_batch_size, vm)
        synchronize(device); validation_seconds += time.perf_counter() - tick
        ap = float(average_precision_score(prepared.labels["validation"], logits))
        loss = float(F.binary_cross_entropy_with_logits(torch.as_tensor(logits), torch.as_tensor(prepared.labels["validation"], dtype=torch.float32)))
        history.append({"epoch": epoch, "training_loss": total / len(y), "validation_ap": ap, "validation_loss": loss})
        print(f"  GNN epoch={epoch} val AP={100 * ap:.2f}%", flush=True)
        if ap > best_ap:
            best_ap, best_epoch, best_loss, stale = ap, epoch, loss, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    return TrainingResult(best_epoch, best_loss, history), {
        "train_seconds": train_seconds, "validation_seconds": validation_seconds,
        "validation_ap": best_ap, "optimizer_steps": updates, "positive_weight": weight,
    }
