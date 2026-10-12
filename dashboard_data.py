"""Build small, anonymized graph examples from the project's real test data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_experiment import PROJECT_ROOT
from src.data import load_dataset, temporal_split
from src.graph import build_temporal_entity_graph


OUTPUTS = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUTS / "dashboard_cache"
DATASETS = ("ieee", "paysim")


def complete_validation_run() -> Path | None:
    """Prefer the frozen four-model study; retain historical v2 compatibility."""
    for directory in sorted(OUTPUTS.glob("thesis_v1_*"), reverse=True):
        protocol_file = directory / "protocol.json"
        if not protocol_file.is_file() or not (directory / "results.json").is_file():
            continue
        protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
        if not protocol.get("test_evaluated") or not all(dataset in protocol.get("datasets", []) for dataset in DATASETS):
            continue
        seed = min(protocol["seeds"])
        if all((directory / dataset / f"seed_{seed}" / name / "predictions.npz").is_file()
               for dataset in DATASETS for name in ("baseline", "autoencoder", "gnn", "hybrid")):
            return directory
    for directory in sorted(OUTPUTS.glob("validation_v2_*"), reverse=True):
        if (directory / "results.json").is_file() and all(
            (directory / f"{dataset}_config.json").is_file() for dataset in DATASETS
        ):
            return directory
    return None


def _relation(frame: pd.DataFrame, source: int, destination: int, dataset: str) -> str:
    columns = ("card_proxy", "card_email_proxy", "device_proxy") if dataset == "ieee" else ("nameOrig", "nameDest")
    left = {str(frame.at[source, column]) for column in columns if column in frame and pd.notna(frame.at[source, column])}
    right = {str(frame.at[destination, column]) for column in columns if column in frame and pd.notna(frame.at[destination, column])}
    if not left.intersection(right):
        return "relacion compartida"
    if dataset == "paysim":
        return "cuenta compartida"
    labels = {"card_proxy": "tarjeta aproximada", "card_email_proxy": "tarjeta/correo", "device_proxy": "dispositivo"}
    matched = [
        labels[column] for column in columns
        if column in frame and pd.notna(frame.at[source, column])
        and pd.notna(frame.at[destination, column])
        and frame.at[source, column] == frame.at[destination, column]
    ]
    return ", ".join(matched) if matched else "entidad compartida"


def _case_graph(frame, edge_index, target: int, dataset: str, train_end: int, validation_end: int,
                scores: dict[str, np.ndarray], test_index: int, feature_count: int, latent_dimension: int,
                case_title: str) -> dict:
    source, destination = edge_index
    direct = source[destination == target].tolist()
    chosen = {target, *direct}
    # Add a few earlier transactions so the animation shows a real temporal chain.
    for node in direct:
        ancestors = source[destination == node].tolist()
        chosen.update(ancestors[-3:])
    if len(chosen) > 18:
        keep = {target, *direct}
        extras = sorted(chosen - keep, reverse=True)[: max(0, 18 - len(keep))]
        chosen = keep.union(extras)
    times = frame["TransactionDT" if dataset == "ieee" else "_time_seconds"].to_numpy()
    ordered = sorted(chosen, key=lambda node: (int(times[node]), node))
    id_map = {node: f"T{node + 1}" for node in ordered}
    nodes = []
    for node in ordered:
        timestamp = int(times[node])
        partition = "train" if node < train_end else "validacion" if node < validation_end else "prueba"
        nodes.append({
            "id": id_map[node], "order": len(nodes), "partition": partition,
            "role": "objetivo" if node == target else "vecino directo" if node in direct else "historial",
            "time": f"dia {timestamp // 86400 + 1}" if dataset == "ieee" else f"paso {timestamp // 3600}",
        })
    edges = [
        {"source": id_map[int(start)], "target": id_map[int(end)],
         "relation": _relation(frame, int(start), int(end), dataset)}
        for start, end in zip(source, destination) if int(start) in chosen and int(end) in chosen
    ]
    label = int(frame.at[target, "isFraud"])
    amount_column = "TransactionAmt" if dataset == "ieee" else "amount"
    amount = float(frame.at[target, amount_column]) if amount_column in frame else None
    ae_score = float(scores["autoencoder"][test_index])
    ae_percentile = float(100 * np.mean(scores["autoencoder"] <= ae_score))
    return {
        "title": case_title, "target": id_map[target], "actual_fraud": bool(label),
        "amount": amount, "nodes": nodes, "edges": edges, "direct_neighbors": len(direct),
        "feature_count": feature_count, "latent_dimension": latent_dimension,
        "model_scores": {name: float(values[test_index]) for name, values in scores.items()},
        "ae_percentile": ae_percentile,
        "explanation": "Las aristas y los puntajes son reales; la disposicion y el recorrido animado son una visualizacion explicativa.",
    }


def build_dataset_cache(dataset: str, source_run: Path | None = None) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Dataset desconocido: {dataset}")
    source_run = source_run or complete_validation_run()
    if source_run is None:
        raise FileNotFoundError("No hay una corrida thesis_v1 o validacion v2 completa con predicciones guardadas")
    config = json.loads((source_run / f"{dataset}_config.json").read_text(encoding="utf-8"))
    print(f"Preparando ejemplo visual real para {dataset.upper()}...", flush=True)
    bundle = load_dataset(config, PROJECT_ROOT, None, "window")
    partitions = temporal_split(bundle.frame, bundle.time_column, config["split"]["train"],
                                config["split"]["validation"], config["split"].get("train_before"),
                                config["split"].get("test_from"))
    frame = pd.concat([partitions.train, partitions.validation, partitions.test], ignore_index=True)
    graph_config = config["graph"]
    graph = build_temporal_entity_graph(frame, bundle.time_column, graph_config["entity_groups"],
                                        int(graph_config["time_window_hours"] * 3600),
                                        int(graph_config["max_previous_neighbors"]))
    train_end = len(partitions.train)
    validation_end = train_end + len(partitions.validation)
    thesis = source_run.name.startswith("thesis_v1_")
    seed = min(json.loads((source_run / "protocol.json").read_text(encoding="utf-8"))["seeds"]) if thesis else 42
    seed_directory = source_run / dataset / f"seed_{seed}"
    raw_predictions = {name: np.load(seed_directory / name / "predictions.npz" if thesis
                                    else seed_directory / "predictions" / f"{name}.npz")
                       for name in ("baseline", "autoencoder", "gnn", "hybrid")}
    try:
        labels = raw_predictions["hybrid"]["labels"]
        scores = {name: prediction["scores"] for name, prediction in raw_predictions.items()}
        if len(labels) != len(partitions.test) or any(len(values) != len(labels) for values in scores.values()):
            raise ValueError("Las predicciones guardadas no coinciden con la particion de prueba")
        if not all(np.array_equal(labels, prediction["labels"]) for prediction in raw_predictions.values()):
            raise ValueError("Las etiquetas de los modelos no estan alineadas")
        in_degree = np.bincount(graph.edge_index[1], minlength=len(frame))[validation_end:]
        linked = in_degree > 0
        fraud = np.flatnonzero((labels == 1) & linked)
        legitimate = np.flatnonzero((labels == 0) & linked)
        if not len(fraud) or not len(legitimate):
            raise ValueError("No se encontraron ejemplos con relaciones en la prueba")
        hybrid_scores = scores["hybrid"]
        chosen_fraud = int(fraud[np.argmax(hybrid_scores[fraud])])
        chosen_alert = int(legitimate[np.argmax(hybrid_scores[legitimate])])
        chosen_normal = int(legitimate[np.argmin(hybrid_scores[legitimate])])
        summary = json.loads((source_run / dataset / "prepared" / "metrics" / "prepare_summary.json").read_text(encoding="utf-8"))
        cases = [
            _case_graph(frame, graph.edge_index, validation_end + index, dataset, train_end, validation_end,
                        scores, index, summary["model_feature_count"], config["model"]["latent_dimension"], title)
            for index, title in ((chosen_fraud, "Fraude real con mayor score entre casos enlazados"),
                                 (chosen_alert, "Alerta en operacion legitima"),
                                 (chosen_normal, "Operacion legitima de bajo riesgo"))
        ]
        payload = {
            "dataset": dataset, "source_run": source_run.name, "source_seed": seed,
            "summary": {"splits": summary["splits"], "graphs": summary["graphs"],
                        "model_feature_count": summary["model_feature_count"]},
            "cases": cases,
            "note": "Ejemplos anonimizados de transacciones de prueba. El grafo completo no se dibuja para evitar saturacion; se muestra un subgrafo real.",
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{dataset}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print(f"Ejemplo de {dataset.upper()} listo: {path}", flush=True)
        return path
    finally:
        for prediction in raw_predictions.values():
            prediction.close()
