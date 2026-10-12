import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import TabularPreprocessor, load_dataset, temporal_split
from src.graph import TemporalGraph, build_temporal_entity_graph
from src.utils import load_config, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class PreparedExperiment:
    config: dict
    bundle: Any
    partitions: Any
    features: dict[str, np.ndarray]
    graph_features: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    graphs: dict[str, TemporalGraph]
    summary: dict


def split_summary(frame, target: str, time_column: str) -> dict:
    return {
        "rows": int(len(frame)),
        "fraud": int(frame[target].sum()),
        "fraud_rate": float(frame[target].mean()),
        "time_min": int(frame[time_column].min()),
        "time_max": int(frame[time_column].max()),
    }


def output_dir(config: dict, category: str) -> Path:
    path = PROJECT_ROOT / "outputs" / config["experiment"]["name"] / category
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_experiment(config: dict, max_rows: int | None, sample_mode: str = "prefix") -> PreparedExperiment:
    bundle = load_dataset(config, PROJECT_ROOT, max_rows, sample_mode)
    split_cfg = config["split"]
    partitions = temporal_split(bundle.frame, bundle.time_column, split_cfg["train"], split_cfg["validation"], split_cfg.get("train_before"), split_cfg.get("test_from"))
    excluded = set(config["data"].get("exclude_features", []))
    excluded.update(bundle.metadata.get("excluded_from_features", []))
    excluded.add(bundle.target_column)
    preprocessor = TabularPreprocessor(
        excluded, config["data"].get("missing_threshold", 0.95), tuple(config["data"].get("log_columns", [])),
        config["data"].get("clip_value"), add_missing_indicators=config["data"].get("add_missing_indicators", False),
        categorical_encoding=config["data"].get("categorical_encoding", "frequency"),
        max_categories=config["data"].get("max_categories", 16),
    )
    features = {
        "train": preprocessor.fit_transform(partitions.train),
        "validation": preprocessor.transform(partitions.validation),
        "test": preprocessor.transform(partitions.test),
    }
    graph_features = {
        "train": features["train"],
        "validation": np.concatenate([features["train"], features["validation"]], axis=0),
        "test": np.concatenate([features["train"], features["validation"], features["test"]], axis=0),
    }
    labels = {
        "train": partitions.train[bundle.target_column].to_numpy(dtype=np.int64, copy=True),
        "validation": partitions.validation[bundle.target_column].to_numpy(dtype=np.int64, copy=True),
        "test": partitions.test[bundle.target_column].to_numpy(dtype=np.int64, copy=True),
    }
    for name, values in labels.items():
        if set(np.unique(values)) != {0, 1}:
            raise ValueError(f"{name} must contain both fraud and legitimate transactions; revise temporal coverage")
    graph_cfg = config["graph"]
    graph_frames = {
        "train": partitions.train,
        "validation": pd.concat([partitions.train, partitions.validation], ignore_index=True),
        "test": pd.concat([partitions.train, partitions.validation, partitions.test], ignore_index=True),
    }
    graphs: dict[str, TemporalGraph] = {}
    for name, frame in graph_frames.items():
        graphs[name] = build_temporal_entity_graph(
            frame,
            bundle.time_column,
            graph_cfg["entity_groups"],
            int(graph_cfg["time_window_hours"] * 3600),
            int(graph_cfg["max_previous_neighbors"]),
        )
    summary = {
        "experiment": config["experiment"],
        "source_metadata": bundle.metadata,
        "raw_rows": int(len(bundle.frame)),
        "raw_columns": int(len(bundle.frame.columns)),
        "model_feature_count": int(features["train"].shape[1]),
        "retained_feature_columns": preprocessor.output_columns,
        "splits": {
            "train": split_summary(partitions.train, bundle.target_column, bundle.time_column),
            "validation": split_summary(partitions.validation, bundle.target_column, bundle.time_column),
            "test": split_summary(partitions.test, bundle.target_column, bundle.time_column),
        },
        "feature_shapes": {name: list(values.shape) for name, values in features.items()},
        "graphs": {
            name: {
                "context_nodes": graph.node_count,
                "target_nodes": int(len(features[name])),
                "edges": graph.edge_count,
            }
            for name, graph in graphs.items()
        },
    }
    save_json(output_dir(config, "metrics") / "prepare_summary.json", summary)
    import pickle
    with (output_dir(config, "checkpoints") / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream)
    return PreparedExperiment(config, bundle, partitions, features, graph_features, labels, graphs, summary)


def run_baseline(prepared: PreparedExperiment) -> dict:
    from src.models.baseline import TabularBaseline

    training_cfg = prepared.config["training"]
    model = TabularBaseline(random_state=training_cfg["seed"])
    model.fit(prepared.features["train"], prepared.labels["train"])
    validation_scores = model.predict_proba(prepared.features["validation"])
    test_scores = model.predict_proba(prepared.features["test"])
    metrics, operating_metrics = evaluate_scores(prepared, validation_scores, test_scores, "baseline")
    result = {"model": "tabular_baseline", "metrics": metrics, "operating_metrics": operating_metrics}
    save_json(output_dir(prepared.config, "metrics") / "baseline_metrics.json", result)
    return result


def torch_data(prepared: PreparedExperiment, device):
    import torch

    features = {name: torch.from_numpy(values).to(device) for name, values in prepared.features.items()}
    graph_features = {name: torch.from_numpy(values).to(device) for name, values in prepared.graph_features.items()}
    labels = {name: torch.from_numpy(values).to(device) for name, values in prepared.labels.items()}
    edges = {name: torch.from_numpy(graph.edge_index).long().to(device) for name, graph in prepared.graphs.items()}
    return features, graph_features, labels, edges


def torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_scores(prepared: PreparedExperiment, validation_scores: np.ndarray, test_scores: np.ndarray, model_name: str) -> tuple[dict, dict]:
    from src.evaluation import classification_metrics, select_f1_threshold, select_threshold_at_fp_budget

    validation_labels = prepared.labels["validation"]
    test_labels = prepared.labels["test"]
    f1_threshold = select_f1_threshold(validation_labels, validation_scores)
    budget = float(prepared.config["training"].get("max_false_positives_per_1000", 10))
    operating_threshold = select_threshold_at_fp_budget(validation_labels, validation_scores, budget)
    np.savez_compressed(output_dir(prepared.config, "predictions") / f"{model_name}.npz", labels=test_labels, scores=test_scores, validation_labels=validation_labels, validation_scores=validation_scores, time=prepared.partitions.test[prepared.bundle.time_column].to_numpy(), threshold=f1_threshold)
    return (
        classification_metrics(test_labels, test_scores, f1_threshold),
        classification_metrics(test_labels, test_scores, operating_threshold),
    )


def run_autoencoder(prepared: PreparedExperiment, epochs: int) -> dict:
    import torch

    from src.models.autoencoder import Autoencoder
    from src.training.torch_trainer import fit_autoencoder

    device = torch_device()
    features, _, labels, _ = torch_data(prepared, device)
    model_cfg = prepared.config["model"]
    training_cfg = prepared.config["training"]
    set_global_seed(int(training_cfg["seed"]))
    model = Autoencoder(features["train"].shape[1], model_cfg["hidden_dimension"], model_cfg["latent_dimension"], model_cfg["dropout"]).to(device)
    training = fit_autoencoder(model, features["train"], labels["train"], epochs, training_cfg["learning_rate"], features["validation"], labels["validation"], training_cfg["batch_size"], training_cfg["early_stopping_patience"])
    model.eval()
    with torch.no_grad():
        validation_reconstruction, _ = model(features["validation"])
        test_reconstruction, _ = model(features["test"])
        validation_scores = torch.mean(torch.abs(features["validation"] - validation_reconstruction), dim=1).cpu().numpy()
        test_scores = torch.mean(torch.abs(features["test"] - test_reconstruction), dim=1).cpu().numpy()
    metrics, operating_metrics = evaluate_scores(prepared, validation_scores, test_scores, "autoencoder")
    save_json(output_dir(prepared.config, "metrics") / "autoencoder_history.json", training.history)
    torch.save(model.state_dict(), output_dir(prepared.config, "checkpoints") / "autoencoder.pt")
    result = {"model": "autoencoder", "training": {"best_epoch": training.best_epoch, "best_loss": training.best_validation_loss}, "metrics": metrics, "operating_metrics": operating_metrics}
    save_json(output_dir(prepared.config, "metrics") / "autoencoder_metrics.json", result)
    return result


def run_gnn(prepared: PreparedExperiment, epochs: int) -> dict:
    import torch

    from src.models.gnn_classifier import GNNFraudClassifier
    from src.training.torch_trainer import fit_gnn

    device = torch_device()
    _, features, labels, edges = torch_data(prepared, device)
    model_cfg = prepared.config["model"]
    training_cfg = prepared.config["training"]
    set_global_seed(int(training_cfg["seed"]))
    model = GNNFraudClassifier(features["train"].shape[1], model_cfg["hidden_dimension"], model_cfg["gnn_layers"], model_cfg["dropout"]).to(device)
    validation_start = len(features["validation"]) - len(labels["validation"])
    training = fit_gnn(model, features["train"], edges["train"], labels["train"], features["validation"], edges["validation"], labels["validation"], validation_start, epochs, training_cfg["learning_rate"], training_cfg["weight_decay"], training_cfg["early_stopping_patience"], training_cfg.get("positive_weight_cap"))
    model.eval()
    with torch.no_grad():
        validation_scores = torch.sigmoid(model(features["validation"], edges["validation"])[validation_start:]).cpu().numpy()
        test_start = len(features["test"]) - len(labels["test"])
        test_scores = torch.sigmoid(model(features["test"], edges["test"])[test_start:]).cpu().numpy()
    metrics, operating_metrics = evaluate_scores(prepared, validation_scores, test_scores, "gnn")
    save_json(output_dir(prepared.config, "metrics") / "gnn_history.json", training.history)
    torch.save(model.state_dict(), output_dir(prepared.config, "checkpoints") / "gnn.pt")
    result = {"model": "graphsage", "training": {"best_epoch": training.best_epoch, "best_loss": training.best_validation_loss}, "metrics": metrics, "operating_metrics": operating_metrics}
    save_json(output_dir(prepared.config, "metrics") / "gnn_metrics.json", result)
    return result


def train_hybrid_model(prepared: PreparedExperiment, epochs: int):
    import torch

    from src.models.hybrid import HybridGNNAutoencoder
    from src.training.torch_trainer import fit_autoencoder, fit_hybrid

    device = torch_device()
    _, features, labels, edges = torch_data(prepared, device)
    model_cfg = prepared.config["model"]
    training_cfg = prepared.config["training"]
    set_global_seed(int(training_cfg["seed"]))
    model = HybridGNNAutoencoder(features["train"].shape[1], model_cfg["hidden_dimension"], model_cfg["latent_dimension"], model_cfg["gnn_layers"], model_cfg["dropout"]).to(device)
    validation_start = len(features["validation"]) - len(labels["validation"])
    pretrain_epochs = int(training_cfg.get("ae_pretrain_epochs", 0))
    pretraining = None
    if pretrain_epochs:
        pretraining = fit_autoencoder(
            model.autoencoder, features["train"], labels["train"], pretrain_epochs,
            training_cfg["learning_rate"], features["validation"][validation_start:],
            labels["validation"], training_cfg["batch_size"], training_cfg["early_stopping_patience"],
        )
    training = fit_hybrid(model, features["train"], edges["train"], labels["train"], features["validation"], edges["validation"], labels["validation"], validation_start, epochs, training_cfg["learning_rate"], training_cfg["weight_decay"], training_cfg["reconstruction_weight"], training_cfg["early_stopping_patience"], training_cfg.get("positive_weight_cap"))
    return model, training, pretraining


def run_hybrid(prepared: PreparedExperiment, epochs: int) -> dict:
    import torch

    model, training, pretraining = train_hybrid_model(prepared, epochs)
    device = torch_device()
    _, features, labels, edges = torch_data(prepared, device)
    validation_start = len(features["validation"]) - len(labels["validation"])
    model.eval()
    with torch.no_grad():
        validation_scores = torch.sigmoid(model(features["validation"], edges["validation"])["logits"][validation_start:]).cpu().numpy()
        test_start = len(features["test"]) - len(labels["test"])
        test_scores = torch.sigmoid(model(features["test"], edges["test"])["logits"][test_start:]).cpu().numpy()
    metrics, operating_metrics = evaluate_scores(prepared, validation_scores, test_scores, "hybrid")
    save_json(output_dir(prepared.config, "metrics") / "hybrid_history.json", training.history)
    if pretraining is not None:
        save_json(output_dir(prepared.config, "metrics") / "hybrid_ae_pretraining_history.json", pretraining.history)
    torch.save(model.state_dict(), output_dir(prepared.config, "checkpoints") / "hybrid.pt")
    result = {"model": "hybrid_gnn_autoencoder", "training": {"best_epoch": training.best_epoch, "best_loss": training.best_validation_loss, "ae_pretrain_best_epoch": pretraining.best_epoch if pretraining is not None else None}, "metrics": metrics, "operating_metrics": operating_metrics}
    save_json(output_dir(prepared.config, "metrics") / "hybrid_metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fraud detection experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["prepare", "baseline", "autoencoder", "gnn", "hybrid", "all"], default="prepare")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["prefix", "uniform", "window"], default="prefix")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for smoke tests")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured random seed")
    args = parser.parse_args()
    if args.sample_mode == "uniform" and args.stage in ("gnn", "hybrid", "all"):
        parser.error("Uniform row sampling breaks graph neighborhoods. Use prefix, window, or the full dataset for GNN/hybrid stages.")
    config = load_config(PROJECT_ROOT / args.config)
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    set_global_seed(int(config["training"]["seed"]))
    prepared = prepare_experiment(config, args.max_rows, args.sample_mode)
    epochs = args.epochs or int(config["training"]["epochs"])
    results: list[dict] = []
    if args.stage in ("baseline", "all"):
        results.append(run_baseline(prepared))
    if args.stage in ("autoencoder", "all"):
        results.append(run_autoencoder(prepared, epochs))
    if args.stage in ("gnn", "all"):
        results.append(run_gnn(prepared, epochs))
    if args.stage in ("hybrid", "all"):
        results.append(run_hybrid(prepared, epochs))
    payload = prepared.summary if args.stage == "prepare" else {
        "experiment": config["experiment"],
        "run": {"stage": args.stage, "sample_mode": args.sample_mode, "max_rows": args.max_rows, "epochs": epochs, "seed": config["training"]["seed"], "max_false_positives_per_1000": config["training"].get("max_false_positives_per_1000", 10)},
        "data_summary": {"model_feature_count": prepared.summary["model_feature_count"], "splits": prepared.summary["splits"], "graphs": prepared.summary["graphs"]},
        "results": results,
    }
    if args.stage != "prepare":
        save_json(output_dir(config, "metrics") / "comparison_latest.json", payload)
        save_json(output_dir(config, "metrics") / f"comparison_seed_{config['training']['seed']}.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
