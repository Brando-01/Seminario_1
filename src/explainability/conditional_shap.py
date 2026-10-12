"""Explain target attributes through the REAL hybrid, with past context fixed.

Grouped transformed columns preserve one-hot/missing-indicator coherence.
This is interventional grouped SHAP, not an attribution of graph edges.
"""
import numpy as np
import torch
from src.graph.subgraphs import IncomingNeighborhoods


def feature_groups(columns):
    names, groups = [], []
    lookup = {}
    for index, column in enumerate(columns):
        name = column.split("__category_")[0]
        if name.endswith("__missing"):
            name = name[:-9]
        if name not in lookup:
            lookup[name] = len(names); names.append(name); groups.append([])
        groups[lookup[name]].append(index)
    return names, groups


class FixedContextPredictor:
    """Vectorized disjoint copies retain exact target and ancestor computation."""
    def __init__(self, model, features, index, means, target, depth, batch_size=128):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        batch = index.extract(np.array([target]), depth - 1)
        self.x = torch.as_tensor(features[batch.nodes], device=self.device)
        self.means = torch.as_tensor(means[batch.nodes], device=self.device)
        self.edges = torch.as_tensor(batch.edge_index, device=self.device)
        self.position = int(batch.target_positions[0])
        self.batch_size = batch_size

    def __call__(self, target_features):
        output = []
        n = len(self.x)
        with torch.no_grad():
            for start in range(0, len(target_features), self.batch_size):
                values = torch.as_tensor(target_features[start:start + self.batch_size], dtype=self.x.dtype, device=self.device)
                count = len(values)
                x = self.x.unsqueeze(0).repeat(count, 1, 1)
                x[:, self.position, :] = values
                edges = (self.edges.unsqueeze(0) + torch.arange(count, device=self.device)[:, None, None] * n).permute(1, 0, 2).reshape(2, -1)
                means = self.means.repeat(count, 1)
                logits = self.model(x.reshape(count * n, -1), edges, means)["logits"]
                scores = torch.sigmoid(logits.reshape(count, n)[:, self.position])
                output.append(scores.cpu().numpy())
        return np.concatenate(output)


class GroupCoalitions:
    def __init__(self, predictor, target_features, background, groups):
        self.predictor = predictor
        self.target = np.asarray(target_features, dtype=np.float32)
        self.background = np.asarray(background, dtype=np.float32)
        self.mapping = np.empty(len(self.target), dtype=np.int64)
        for group, columns in enumerate(groups):
            self.mapping[columns] = group

    def __call__(self, masks):
        masks = np.asarray(masks)
        scores = []
        for start in range(0, len(masks), 64):
            expanded = masks[start:start + 64, self.mapping].astype(bool)
            features = np.where(expanded[:, None, :], self.target[None, None, :], self.background[None, :, :])
            predictions = self.predictor(features.reshape(-1, len(self.target)))
            scores.append(predictions.reshape(len(expanded), len(self.background)).mean(axis=1))
        return np.concatenate(scores)


def explain_target(model, features, index, means, target, depth, background, groups, names, seed=42, permutations=3):
    import shap
    predictor = FixedContextPredictor(model, features, index, means, target, depth)
    game = GroupCoalitions(predictor, features[target], background, groups)
    explainer = shap.PermutationExplainer(game, np.zeros((1, len(groups))), feature_names=names, seed=seed)
    result = explainer(np.ones((1, len(groups))), max_evals=permutations * (2 * len(groups) + 1), silent=True)
    values = result.values[0]
    base = float(np.asarray(result.base_values).reshape(-1)[0])
    score = float(predictor(features[target:target + 1])[0])
    residual = abs(base + float(values.sum()) - score)
    if not np.isfinite(values).all() or residual > 1e-5:
        raise ValueError(f"SHAP additivity failed: {residual}")
    return values, base, score, residual
