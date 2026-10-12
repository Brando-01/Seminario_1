"""Present validation metrics in a paper-style table without changing their values."""

from collections import defaultdict

import numpy as np


MODEL_ORDER = ("baseline", "autoencoder", "gnn", "hybrid")
MODEL_LABELS = {
    "baseline": "Baseline tabular",
    "autoencoder": "Autoencoder",
    "gnn": "GraphSAGE",
    "hybrid": "Hibrido GNN-AE",
}
METRICS = (
    ("auc_pr", "AP (%)"),
    ("auc_roc", "ROC-AUC (%)"),
    ("precision", "Precision (%)"),
    ("recall", "Recall (%)"),
    ("f1", "F1 (%)"),
)


def _format_metric(values: list[float], scale: float = 100.0, show_variability: bool = False) -> str:
    numbers = np.asarray(values, dtype=float)
    if not len(numbers):
        return "N/D"
    mean = f"{scale * numbers.mean():.2f}"
    if not show_variability:
        return mean
    if len(numbers) == 1:
        return f"{mean} (DE N/D; n=1)"
    deviation = numbers.std(ddof=1)
    return f"{mean} +/- {scale * deviation:.2f}"


def render_validation_report(rows: list[dict], summaries: dict | None = None, show_variability: bool = False) -> str:
    """Format stored test results; never reselect thresholds or recalculate predictions."""
    summaries = summaries or {}
    datasets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        datasets[row["dataset"]][row["model"]].append(row)
    lines = [
        "# Resultados de validacion temporal",
        "",
        "AP = average precision (precision promedio), no accuracy ni area PR trapezoidal.",
        "Cada celda muestra el promedio de las semillas como porcentaje, sin +/-.",
        "La variacion entre semillas se puede consultar por separado; comparten el mismo test.",
        "F1 es la media de los F1 individuales, no el F1 calculado a partir de Precision y Recall medios.",
        "",
    ]
    if show_variability:
        lines[3:5] = [
            "Cada celda muestra media +/- desviacion estandar muestral; en las metricas porcentuales,",
            "ambas cifras estan en puntos porcentuales. Las semillas comparten el mismo periodo de prueba.",
        ]
    for dataset in sorted(datasets):
        by_model = datasets[dataset]
        selected = [model for model in MODEL_ORDER if model in by_model]
        selected += sorted(set(by_model) - set(selected))
        seeds = sorted({row["seed"] for models in by_model.values() for row in models})
        lines.extend([f"## {dataset.upper()}", "", f"Semillas: {', '.join(map(str, seeds))}."])
        test = summaries.get(dataset, {}).get("splits", {}).get("test", {})
        if test:
            lines.append(
                f"Prueba: {test['rows']:,} operaciones, {test['fraud']:,} fraudes "
                f"({100 * test['fraud_rate']:.4f}% de prevalencia)."
            )
        lines.extend([
            "Umbral de Precision, Recall y F1: elegido solo en validacion para maximizar F1.",
            "",
            "| Modelo | " + " | ".join(label for _, label in METRICS) + " |",
            "|---|" + "---:|" * len(METRICS),
        ])
        for model in selected:
            model_rows = by_model[model]
            cells = [_format_metric([row["metrics"][key] for row in model_rows], show_variability=show_variability) for key, _ in METRICS]
            lines.append(f"| {MODEL_LABELS.get(model, model)} | " + " | ".join(cells) + " |")
        lines.append("")
        if all(model in by_model for model in ("baseline", "hybrid")):
            mean_ap = {
                model: np.mean([row["metrics"]["auc_pr"] for row in model_rows])
                for model, model_rows in by_model.items()
            }
            winner = max(mean_ap, key=mean_ap.get)
            comparison = (
                f"Mayor AP media observada: {MODEL_LABELS.get(winner, winner)}. "
                f"Hibrido - baseline: {100 * (mean_ap['hybrid'] - mean_ap['baseline']):+.2f} "
                f"puntos porcentuales"
            )
            if "gnn" in mean_ap:
                comparison += (f"; hibrido - GNN: "
                               f"{100 * (mean_ap['hybrid'] - mean_ap['gnn']):+.2f} puntos porcentuales")
            lines.append(comparison + ".")
            lines.append("")
        if all("operating_metrics" in row for model_rows in by_model.values() for row in model_rows):
            lines.extend([
                "Punto operativo: umbral fijado en validacion con presupuesto de falsas alarmas.",
                "| Modelo | Recall operativo (%) | FP por 1000 legitimas |",
                "|---|---:|---:|",
            ])
            for model in selected:
                model_rows = by_model[model]
                recall = _format_metric([row["operating_metrics"]["recall"] for row in model_rows], show_variability=show_variability)
                fp = _format_metric([row["operating_metrics"]["false_positives_per_1000"] for row in model_rows], 1.0, show_variability)
                lines.append(f"| {MODEL_LABELS.get(model, model)} | {recall} | {fp} |")
            lines.append("")
        if all("seconds" in row and "inference_seconds" in row
               for model_rows in by_model.values() for row in model_rows):
            lines.extend([
                "Tiempos locales medios por semilla; inferencia sobre toda la particion de test.",
                "Entrenamiento incluye preentrenamiento AE; excluye preparacion, busqueda y validacion.",
                "| Modelo | Entrenamiento (s) | Inferencia test (s) |",
                "|---|---:|---:|",
            ])
            for model in selected:
                model_rows = by_model[model]
                train_time = _format_metric([row["seconds"] for row in model_rows], 1.0, show_variability)
                inference_time = _format_metric([row["inference_seconds"] for row in model_rows], 1.0, show_variability)
                lines.append(f"| {MODEL_LABELS.get(model, model)} | {train_time} | {inference_time} |")
            lines.append("")
    lines.extend([
        "No se demuestra superioridad estadistica con semillas sobre un solo periodo de test.",
        "Comparar modelos dentro de cada dataset; la AP entre datasets cambia con la prevalencia.",
        "Un cambio de presentacion a porcentajes no modifica ningun resultado.",
    ])
    return "\n".join(lines) + "\n"


def render_hybrid_search_report(datasets: dict[str, tuple[list[dict], list[dict]]], show_variability: bool = False) -> str:
    """Show validation-selected exploratory results without hiding weak seeds."""
    lines = [
        "# Exploracion del entrenamiento hibrido",
        "",
        "Candidatos elegidos por AP de validacion; el test historico ya habia sido inspeccionado.",
        "Por tanto, estas cifras son exploratorias y requieren un nuevo periodo de prueba.",
        "AP significa average precision, no porcentaje de transacciones acertadas.",
        "",
    ]
    for dataset in sorted(datasets):
        validation, test = datasets[dataset]
        if not validation or not test:
            continue
        selected = {row["candidate"] for row in test}
        if len(selected) != 1:
            raise ValueError("A dataset must have exactly one selected candidate")
        winner = selected.pop()
        candidates = sorted({row["candidate"] for row in validation})
        lines.extend([f"## {dataset.upper()}", "", "AP de validacion (%) usada para elegir:", "", "| Variante | AP de validacion (%) |", "|---|---:|"])
        for candidate in candidates:
            values = [row["validation_ap"] for row in validation if row["candidate"] == candidate]
            name = candidate + (" (elegida)" if candidate == winner else "")
            lines.append(f"| {name} | {_format_metric(values, show_variability=show_variability)} |")
        lines.extend(["", "Resultado en test de la variante elegida:", "", "| AP (%) | ROC-AUC (%) | Precision (%) | Recall (%) | F1 (%) | Recall operativo (%) | FP/1000 |", "|---:|---:|---:|---:|---:|---:|---:|"])
        cells = [_format_metric([row["metrics"][key] for row in test], show_variability=show_variability) for key in ("auc_pr", "auc_roc", "precision", "recall", "f1")]
        cells += [
            _format_metric([row["operating_metrics"]["recall"] for row in test], show_variability=show_variability),
            _format_metric([row["operating_metrics"]["false_positives_per_1000"] for row in test], 1.0, show_variability),
        ]
        lines.extend(["| " + " | ".join(cells) + " |", ""])
        if dataset == "paysim":
            lines.append("PaySim tiene solo 36 fraudes en el test historico; revisar la variacion por semilla.")
            lines.append("")
    lines.append("No presentar esta exploracion como superioridad confirmada frente al baseline.")
    return "\n".join(lines) + "\n"
