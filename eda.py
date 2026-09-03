import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_dataset
from src.utils import load_config


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "ieee": ROOT / "configs" / "experiment_1_ieee.yaml",
    "paysim": ROOT / "configs" / "experiment_2_paysim.yaml",
}
AMOUNT_COLUMNS = {"ieee": "TransactionAmt", "paysim": "amount"}
GROUP_COLUMNS = {"ieee": "ProductCD", "paysim": "type"}


def safe_number(value):
    if pd.isna(value):
        return None
    return float(value)


def analyze(experiment: str, max_rows: int, sample_mode: str, seed: int) -> dict:
    config = load_config(CONFIGS[experiment])
    config["training"]["seed"] = seed
    bundle = load_dataset(config, ROOT, max_rows=max_rows, sample_mode=sample_mode)
    frame = bundle.frame
    target = bundle.target_column
    amount_column = AMOUNT_COLUMNS[experiment]
    group_column = GROUP_COLUMNS[experiment]

    missing = frame.isna().mean().sort_values(ascending=False).head(15)
    amount = pd.to_numeric(frame[amount_column], errors="coerce")
    grouped = (
        frame.groupby(group_column, dropna=False)[target]
        .agg(transactions="size", frauds="sum", fraud_rate="mean")
        .sort_values("transactions", ascending=False)
    )
    quantiles = amount.quantile([0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    summary = {
        "experiment": experiment,
        "dataset": bundle.name,
        "sample": {"mode": sample_mode, "requested_rows": max_rows, "seed": seed},
        "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
        "target": {
            "column": target,
            "frauds": int(frame[target].sum()),
            "legitimate": int((frame[target] == 0).sum()),
            "fraud_rate": float(frame[target].mean()),
            "legitimate_per_fraud": float((frame[target] == 0).sum() / max(frame[target].sum(), 1)),
        },
        "time": {
            "column": bundle.time_column,
            "minimum": safe_number(frame[bundle.time_column].min()),
            "maximum": safe_number(frame[bundle.time_column].max()),
        },
        "column_types": {
            "numeric": int(sum(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)),
            "categorical": int(sum(not pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)),
        },
        "top_missing": {column: float(rate) for column, rate in missing.items()},
        "amount": {
            "column": amount_column,
            "mean": safe_number(amount.mean()),
            "std": safe_number(amount.std()),
            "quantiles": {str(index): safe_number(value) for index, value in quantiles.items()},
        },
        "groups": {
            "column": group_column,
            "values": {
                str(index): {
                    "transactions": int(row["transactions"]),
                    "frauds": int(row["frauds"]),
                    "fraud_rate": float(row["fraud_rate"]),
                }
                for index, row in grouped.iterrows()
            },
        },
        "source_metadata": bundle.metadata,
    }
    output = ROOT / "outputs" / config["experiment"]["name"] / "eda"
    output.mkdir(parents=True, exist_ok=True)
    (output / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    target = summary["target"]
    print(f"\nEDA {summary['experiment'].upper()}")
    print("=" * 72)
    print(f"Filas analizadas: {summary['shape']['rows']:,} | Columnas: {summary['shape']['columns']}")
    print(f"Fraudes: {target['frauds']:,} | Tasa de fraude: {100 * target['fraud_rate']:.4f}%")
    print(f"Desbalance: {target['legitimate_per_fraud']:.1f} operaciones legitimas por fraude")
    amount = summary["amount"]
    print(f"Monto ({amount['column']}): mediana={amount['quantiles']['0.5']:.2f} | p95={amount['quantiles']['0.95']:.2f} | max={amount['quantiles']['1.0']:.2f}")
    print(f"\nFraude por {summary['groups']['column']}:")
    print(f"{'Valor':<22}{'Transacciones':>15}{'Fraudes':>12}{'Tasa':>12}")
    for value, row in summary["groups"]["values"].items():
        print(f"{value:<22}{row['transactions']:>15,}{row['frauds']:>12,}{100 * row['fraud_rate']:>11.4f}%")
    print("\nMayor porcentaje de datos faltantes:")
    for column, rate in list(summary["top_missing"].items())[:8]:
        print(f"- {column}: {100 * rate:.2f}%")
    print("\nInterpretacion:")
    print("- Una tasa de fraude muy baja confirma que accuracy no es una metrica adecuada.")
    print("- Compara AUC-PR, recall, precision, F1 y falsas alarmas por 1000 operaciones.")
    print("- Este EDA usa una muestra; los resultados finales deben documentar exactamente la muestra usada.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisis exploratorio reproducible de IEEE-CIS o PaySim")
    parser.add_argument("--experiment", choices=["ieee", "paysim"], required=True)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--sample-mode", choices=["prefix", "uniform", "window"], default="uniform")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_rows < 1:
        parser.error("--max-rows debe ser positivo")
    summary = analyze(args.experiment, args.max_rows, args.sample_mode, args.seed)
    print_summary(summary)


if __name__ == "__main__":
    main()
