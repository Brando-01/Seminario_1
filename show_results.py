import argparse
import json
from pathlib import Path

import numpy as np

from validation_report import render_hybrid_search_report, render_validation_report


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "ieee": "experiment_1_ieee",
    "paysim": "experiment_2_paysim",
    "paysim-strict": "experiment_2_paysim_strict",
}
MAIN_EXPERIMENTS = ["ieee", "paysim"]
MODEL_LABELS = {
    "tabular_baseline": "Baseline",
    "autoencoder": "Autoencoder",
    "graphsage": "GraphSAGE",
    "hybrid_gnn_autoencoder": "Hibrido",
}


def load_results(experiment: str) -> dict | None:
    metrics_dir = ROOT / "outputs" / experiment / "metrics"
    latest = metrics_dir / "comparison_latest.json"
    if latest.is_file():
        payload = json.loads(latest.read_text(encoding="utf-8"))
        prepare = metrics_dir / "prepare_summary.json"
        if "data_summary" not in payload and prepare.is_file():
            summary = json.loads(prepare.read_text(encoding="utf-8"))
            payload["data_summary"] = {key: summary[key] for key in ("model_feature_count", "splits", "graphs") if key in summary}
        return payload
    results = []
    for filename in ("baseline_metrics.json", "autoencoder_metrics.json", "gnn_metrics.json", "hybrid_metrics.json"):
        path = metrics_dir / filename
        if path.is_file():
            results.append(json.loads(path.read_text(encoding="utf-8")))
    return {"experiment": {"name": experiment}, "run": {}, "results": results} if results else None


def percentage(value: float) -> str:
    return f"{100 * value:6.2f}%"


def print_experiment(key: str, payload: dict, details: bool) -> None:
    print(f"\n{'=' * 86}\n{key.upper()} - RESULTADOS DISPONIBLES\n{'=' * 86}")
    run = payload.get("run", {})
    if run:
        print(f"Modo={run.get('sample_mode')} | Filas={run.get('max_rows')} | Epocas={run.get('epochs')} | Semilla={run.get('seed', 'N/D')}")
    else:
        print("Metadatos de la corrida no disponibles (resultados anteriores al visor actual).")
    headers = ("Modelo", "AUC-PR", "AUC-ROC", "Precision", "Recall", "F1", "FP/1000")
    print(f"{headers[0]:<16}{headers[1]:>10}{headers[2]:>11}{headers[3]:>11}{headers[4]:>10}{headers[5]:>10}{headers[6]:>11}")
    print("-" * 86)
    rows = []
    for result in payload.get("results", []):
        metrics = result["metrics"]
        rows.append((result["model"], metrics))
        print(
            f"{MODEL_LABELS.get(result['model'], result['model']):<16}"
            f"{percentage(metrics['auc_pr']):>10}{percentage(metrics['auc_roc']):>11}"
            f"{percentage(metrics['precision']):>11}{percentage(metrics['recall']):>10}"
            f"{percentage(metrics['f1']):>10}{metrics['false_positives_per_1000']:>11.2f}"
        )
    if not rows:
        print("No hay resultados de modelos.")
        return
    best_auc_pr = max(rows, key=lambda item: item[1]["auc_pr"])
    best_f1 = max(rows, key=lambda item: item[1]["f1"])
    best_recall = max(rows, key=lambda item: item[1]["recall"])
    lowest_fp = min(rows, key=lambda item: item[1]["false_positives_per_1000"])
    print("\nINTERPRETACION AUTOMATICA")
    print(f"- Mejor discriminacion con desbalance (AUC-PR): {MODEL_LABELS.get(best_auc_pr[0], best_auc_pr[0])} ({percentage(best_auc_pr[1]['auc_pr']).strip()}).")
    print(f"- Mejor equilibrio precision-recall (F1): {MODEL_LABELS.get(best_f1[0], best_f1[0])} ({percentage(best_f1[1]['f1']).strip()}).")
    print(f"- Mayor captura de fraudes (Recall): {MODEL_LABELS.get(best_recall[0], best_recall[0])} ({percentage(best_recall[1]['recall']).strip()}).")
    print(f"- Menos falsas alarmas: {MODEL_LABELS.get(lowest_fp[0], lowest_fp[0])} ({lowest_fp[1]['false_positives_per_1000']:.2f} por 1000 legitimas).")
    hybrid = next((metrics for model, metrics in rows if model == "hybrid_gnn_autoencoder"), None)
    gnn = next((metrics for model, metrics in rows if model == "graphsage"), None)
    autoencoder = next((metrics for model, metrics in rows if model == "autoencoder"), None)
    if hybrid and gnn and autoencoder:
        reference = max(gnn["auc_pr"], autoencoder["auc_pr"])
        delta = hybrid["auc_pr"] - reference
        verdict = "supera" if delta > 0 else "no supera"
        print(f"- El hibrido {verdict} al mejor modelo individual en AUC-PR (diferencia {delta:+.4f}).")
    operational = [(result["model"], result.get("operating_metrics")) for result in payload.get("results", [])]
    operational = [(model, metrics) for model, metrics in operational if metrics]
    if operational:
        budget = run.get("max_false_positives_per_1000", 10)
        print(f"\nPUNTO OPERATIVO (umbral elegido en validacion con maximo {budget:g} FP/1000)")
        print(f"{'Modelo':<16}{'Precision':>12}{'Recall':>12}{'F1':>12}{'FP/1000':>12}")
        for model, metrics in operational:
            print(
                f"{MODEL_LABELS.get(model, model):<16}{percentage(metrics['precision']):>12}"
                f"{percentage(metrics['recall']):>12}{percentage(metrics['f1']):>12}"
                f"{metrics['false_positives_per_1000']:>12.2f}"
            )
        exceeded = [MODEL_LABELS.get(model, model) for model, metrics in operational if metrics["false_positives_per_1000"] > budget]
        if exceeded:
            print(f"ADVERTENCIA DE DERIVA: el limite validado no se mantuvo en prueba para: {', '.join(exceeded)}.")
    splits = payload.get("data_summary", {}).get("splits", {})
    if all(name in splits for name in ("train", "validation", "test")):
        train_rate = splits["train"]["fraud_rate"]
        future_rate = max(splits["validation"]["fraud_rate"], splits["test"]["fraud_rate"])
        if train_rate > 0 and future_rate / train_rate >= 2:
            print(f"ADVERTENCIA DE DERIVA TEMPORAL: la tasa futura de fraude llega a ser {future_rate / train_rate:.1f} veces la de entrenamiento.")
    if run.get("epochs", 0) and run.get("epochs", 0) < 10:
        print("\nADVERTENCIA: menos de 10 epocas; esta corrida es tecnica y no debe reportarse como resultado final.")
    if run.get("max_rows"):
        print("ADVERTENCIA: se uso una muestra; documenta el modo y numero de filas antes de citar resultados.")
    if not run:
        print("\nADVERTENCIA: no se conocen filas, epocas ni semilla; trata esta corrida como tecnica y vuelve a entrenar antes de citarla.")
    if details:
        print("\nCOMO LEER LAS METRICAS")
        print("- AUC-PR: metrica principal; cuanto mayor, mejor bajo desbalance extremo.")
        print("- Precision: de las alertas emitidas, cuantas eran fraude.")
        print("- Recall: de todos los fraudes reales, cuantos fueron detectados.")
        print("- F1: equilibrio entre precision y recall.")
        print("- FP/1000: falsas alarmas por cada 1000 operaciones legitimas; cuanto menor, mejor.")
        print("- AUC-ROC: capacidad general de separar clases; puede verse optimista con fraude muy raro.")


def print_seed_aggregate(key: str, experiment: str, latest: dict) -> None:
    metrics_dir = ROOT / "outputs" / experiment / "metrics"
    latest_run = latest.get("run", {})
    payloads = []
    for path in sorted(metrics_dir.glob("comparison_seed_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run = payload.get("run", {})
        comparable = all(run.get(field) == latest_run.get(field) for field in ("stage", "sample_mode", "max_rows", "epochs"))
        if comparable and len(payload.get("results", [])) == 4:
            payloads.append(payload)
    if len(payloads) < 2:
        print(f"\n{key.upper()}: se necesitan al menos 2 semillas comparables para calcular media y desviacion.")
        return
    by_model: dict[str, list[dict]] = {}
    for payload in payloads:
        for result in payload["results"]:
            by_model.setdefault(result["model"], []).append(result["metrics"])
    seeds = [payload["run"].get("seed") for payload in payloads]
    print(f"\n{key.upper()} - ESTABILIDAD ENTRE SEMILLAS {seeds}")
    print("=" * 72)
    print(f"{'Modelo':<18}{'AUC-PR media+/-std':>23}{'F1 media+/-std':>22}")
    for model, metrics in by_model.items():
        auc = np.asarray([row["auc_pr"] for row in metrics])
        f1 = np.asarray([row["f1"] for row in metrics])
        label = MODEL_LABELS.get(model, model)
        print(f"{label:<18}{100 * auc.mean():>8.2f}% +/- {100 * auc.std(ddof=1):>5.2f}{100 * f1.mean():>10.2f}% +/- {100 * f1.std(ddof=1):>5.2f}")
    print("Una desviacion menor indica que la conclusion es mas estable frente a la semilla aleatoria.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestra e interpreta resultados de los experimentos")
    parser.add_argument("--experiment", choices=[*EXPERIMENTS, "all"], default="all")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--show-variability", action="store_true", help="Muestra tambien la desviacion estandar entre semillas")
    parser.add_argument("--aggregate-seeds", action="store_true")
    parser.add_argument("--validation-run", help="Nombre de carpeta validation_v2_... o latest; no mezcla corridas antiguas")
    parser.add_argument("--hybrid-search", help="Nombre de carpeta hybrid_search_... o latest")
    args = parser.parse_args()
    if args.validation_run and args.hybrid_search:
        parser.error("Use --validation-run o --hybrid-search, no ambos")
    if args.hybrid_search:
        if args.hybrid_search == "latest":
            candidates = sorted((ROOT / "outputs").glob("hybrid_search_*"))
            if not candidates:
                parser.error("No hay exploraciones hybrid_search")
            directory = candidates[-1]
        else:
            directory = ROOT / "outputs" / args.hybrid_search
        datasets = {}
        for dataset in ("ieee", "paysim"):
            validation_file = directory / dataset / "candidate_validation.json"
            test_file = directory / dataset / "selected_test.json"
            if validation_file.is_file() and test_file.is_file():
                datasets[dataset] = (
                    json.loads(validation_file.read_text(encoding="utf-8")),
                    json.loads(test_file.read_text(encoding="utf-8")),
                )
        if not datasets:
            parser.error("No hay resultados completos de la exploracion seleccionada")
        print(f"Corrida: {directory}")
        print(render_hybrid_search_report(datasets, show_variability=args.show_variability))
        if args.details:
            print("Por semilla: AP en test y operacion al umbral fijado en validacion")
            print(f"{'Dataset':<10}{'Semilla':<9}{'AP (%)':>10}{'Recall op. (%)':>17}{'FP/1000':>10}")
            for dataset, (_, rows) in datasets.items():
                for row in rows:
                    print(
                        f"{dataset:<10}{row['seed']:<9}{100 * row['metrics']['auc_pr']:>9.2f}%"
                        f"{100 * row['operating_metrics']['recall']:>16.2f}%"
                        f"{row['operating_metrics']['false_positives_per_1000']:>10.2f}"
                    )
        return
    if args.validation_run:
        if args.validation_run == "latest":
            candidates = sorted((ROOT / "outputs").glob("validation_v2_*"))
            if not candidates:
                parser.error("No hay corridas validation_v2")
            directory = candidates[-1]
        else:
            directory = ROOT / "outputs" / args.validation_run
        report = directory / "REPORT.md"
        print(f"Corrida: {directory}")
        results_file = directory / "results.json"
        if report.is_file() and results_file.is_file():
            rows = json.loads(results_file.read_text(encoding="utf-8"))
            summaries = {}
            for dataset in {row["dataset"] for row in rows}:
                summary_file = directory / dataset / "prepared" / "metrics" / "prepare_summary.json"
                if summary_file.is_file():
                    summaries[dataset] = json.loads(summary_file.read_text(encoding="utf-8"))
            print(render_validation_report(rows, summaries, show_variability=args.show_variability))
            if args.details:
                print("Detalle por semilla: AP en test; recall y FP/1000 al umbral operativo fijado en validacion")
                print(f"{'Dataset':<10}{'Semilla':<9}{'Modelo':<15}{'AP (%)':>10}{'Recall (%)':>13}{'FP/1000':>10}")
                for row in rows:
                    op = row["operating_metrics"]
                    print(
                        f"{row['dataset']:<10}{row['seed']:<9}{row['model']:<15}"
                        f"{100 * row['metrics']['auc_pr']:>9.2f}%"
                        f"{100 * op['recall']:>12.2f}%"
                        f"{op['false_positives_per_1000']:>10.2f}"
                    )
        elif report.is_file():
            print(report.read_text(encoding="utf-8"))
        else:
            print("CORRIDA INCOMPLETA: no hay informe final. Revisar terminal de entrenamiento.")
        if not report.is_file():
            raise SystemExit(1)
        return
    keys = MAIN_EXPERIMENTS if args.experiment == "all" else [args.experiment]
    found = False
    for key in keys:
        payload = load_results(EXPERIMENTS[key])
        if payload:
            found = True
            print_experiment(key, payload, args.details)
            if args.aggregate_seeds:
                print_seed_aggregate(key, EXPERIMENTS[key], payload)
        else:
            print(f"No se encontraron resultados para {key}.")
    if not found:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
