import numpy as np
import json
import pytest
import torch

from improve_hybrid import choose_batch_candidate
from src.graph.subgraphs import IncomingNeighborhoods
from src.models.hybrid import HybridGNNAutoencoder
from src.training.graph_batches import predict_hybrid_batches
from src.training.graph_batches import fit_hybrid_batches
from types import SimpleNamespace
import improve_hybrid


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("attention", [False, True])
def test_batched_predictions_match_full_graph_with_old_context_and_isolated_nodes(depth, attention):
    torch.manual_seed(7)
    features = np.random.default_rng(7).normal(size=(13, 5)).astype(np.float32)
    edges = np.array([[0, 1, 2, 0, 4, 5, 6, 2, 8, 9], [2, 3, 4, 5, 6, 6, 7, 8, 9, 11]], dtype=np.int64)
    model = HybridGNNAutoencoder(5, 8, 3, depth, dropout=0.0, attention=attention, normalize_fusion=attention).eval()
    targets = np.array([11, 7, 12, 0, 9])
    with torch.no_grad():
        expected = model(torch.from_numpy(features), torch.from_numpy(edges))["logits"][targets].numpy()
    actual = predict_hybrid_batches(model, features, IncomingNeighborhoods(edges, len(features)), targets, depth, batch_size=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_neighborhood_excludes_unrelated_future_and_preserves_target_order():
    index = IncomingNeighborhoods(np.array([[0, 1, 2, 3], [1, 2, 3, 4]]), 6)
    batch = index.extract(np.array([2, 1]), depth=1)
    assert batch.nodes.tolist() == [0, 1, 2]
    assert batch.target_positions.tolist() == [2, 1]
    assert batch.edge_index.tolist() == [[0, 1], [1, 2]]


def test_variant_selection_ignores_test_values():
    assert choose_batch_candidate([
        {"candidate": "minibatch", "validation_ap": 0.5, "test_ap": 0.99},
        {"candidate": "attention_norm", "validation_ap": 0.6, "test_ap": 0.01},
    ]) == "attention_norm"


def test_supervised_optimizer_updates_once_per_target_batch():
    x = np.random.default_rng(9).normal(size=(8, 3)).astype(np.float32)
    prepared = SimpleNamespace(
        config={"model": {"gnn_layers": 1}, "training": {"learning_rate": 0.001,
                "weight_decay": 0.0001, "seed": 42, "positive_weight_cap": 20,
                "reconstruction_weight": 0.01, "early_stopping_patience": 10}},
        features={"train": x[:6]}, graph_features={"validation": x},
        labels={"train": np.array([0, 1, 0, 0, 1, 0]), "validation": np.array([0, 1])},
        graphs={"train": SimpleNamespace(edge_index=np.array([[0, 1, 2], [1, 2, 4]])),
                "validation": SimpleNamespace(edge_index=np.array([[0, 1, 2, 4], [1, 2, 4, 6]]))},
    )
    model = HybridGNNAutoencoder(3, 4, 2, gnn_layers=1, dropout=0)
    result, info = fit_hybrid_batches(model, prepared, epochs=2, batch_size=3, inference_batch_size=2)
    assert info["updates_per_epoch"] == 2
    assert info["optimizer_steps"] == 4
    assert len(result.history) == 2


def _frozen_run(tmp_path):
    root = tmp_path / "outputs" / "minibatch_search_20261012_011449"
    selected = root / "ieee"
    selected.mkdir(parents=True)
    (root / "protocol.json").write_text(json.dumps({
        "datasets": ["ieee", "paysim"], "seeds": [42], "search_complete": False,
        "caveat": "Exploratory historical test", "test_evaluated": False,
    }))
    (selected / "selection.json").write_text(json.dumps({"candidate": "attention_norm", "test_used": False}))
    (selected / "selected_validation.json").write_text(json.dumps([
        {"candidate": "attention_norm", "seed": 42},
    ]))
    return root


def test_completed_dataset_can_be_read_while_other_dataset_is_training(tmp_path, monkeypatch):
    root = _frozen_run(tmp_path)
    monkeypatch.setattr(improve_hybrid, "PROJECT_ROOT", tmp_path)
    metrics = {"auc_pr": 0.5, "auc_roc": 0.8, "precision": 0.6, "recall": 0.4, "f1": 0.48}
    rows = [{"dataset": "ieee", "seed": 42, "model": name, "metrics": metrics}
            for name in ("hybrid", "baseline")]
    (root / "results.json").write_text(json.dumps(rows))
    improve_hybrid.evaluate_run(root.name, "ieee")
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["evaluated_datasets"] == ["ieee"]
    assert protocol["test_evaluated"] is False
    assert protocol["search_complete"] is False


@pytest.mark.parametrize("fault", ["missing_seed", "different_variant", "test_used"])
def test_evaluation_rejects_inconsistent_frozen_selection(tmp_path, monkeypatch, fault):
    root = _frozen_run(tmp_path)
    monkeypatch.setattr(improve_hybrid, "PROJECT_ROOT", tmp_path)
    if fault == "missing_seed":
        (root / "ieee/selected_validation.json").write_text("[]")
    elif fault == "different_variant":
        (root / "ieee/selected_validation.json").write_text(json.dumps([{"candidate": "minibatch", "seed": 42}]))
    else:
        (root / "ieee/selection.json").write_text(json.dumps({"candidate": "attention_norm", "test_used": True}))
    with pytest.raises(ValueError, match="Incomplete or inconsistent"):
        improve_hybrid.evaluate_run(root.name, "ieee")
    assert not (root / "results.json").exists()
