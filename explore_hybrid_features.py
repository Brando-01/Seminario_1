"""Train-only feature ranking and validation-AP experiment for the GNN-AE.

By default no test metric is computed. --evaluate-test is exploratory only:
the historical v2 test was already inspected; confirmation needs a fresh holdout.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from run_experiment import PROJECT_ROOT, evaluate_scores, prepare_experiment, save_json, torch_data, torch_device, train_hybrid_model
from validate_experiments import build_v2_config


def select_train_features(features: np.ndarray, labels: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank each column using training labels only, preserving original order."""
    if top_k < 1 or top_k > features.shape[1]:
        raise ValueError("top_k must be between 1 and the number of input columns")
    rank = np.zeros(features.shape[1], dtype=np.float64)
    for column in range(features.shape[1]):
        values = features[:, column]
        if np.isfinite(values).all() and np.ptp(values) > 0:
            auc = roc_auc_score(labels, values)
            rank[column] = abs(auc - 0.5)
    selected = np.sort(np.argsort(-rank, kind="stable")[:top_k])
    return selected, rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("ieee", "paysim"), default="ieee")
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--ae-pretrain-epochs", type=int, default=10)
    parser.add_argument("--evaluate-test", action="store_true", help="Exploratory only: historical test was already inspected")
    args = parser.parse_args()
    if args.epochs < 1 or args.ae_pretrain_epochs < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive, pretraining nonnegative, and seeds unique")
    torch.set_num_threads(4)
    run_id = "feature_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "outputs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    config = build_v2_config(args.experiment, run_id, args.epochs)
    config["training"]["ae_pretrain_epochs"] = args.ae_pretrain_epochs
    save_json(root / "protocol.json", {
        "dataset": args.experiment, "top_k": args.top_k, "epochs": args.epochs,
        "ae_pretrain_epochs": args.ae_pretrain_epochs, "seeds": args.seeds,
        "selection": "univariate feature ranking on train; candidate judged by validation AP",
        "test_evaluated": args.evaluate_test,
        "caveat": "Historical v2 test was inspected before this experiment; any test result is exploratory.",
    })
    save_json(root / "config.json", config)
    prepared = prepare_experiment(config, None, "window")
    selected, rank = select_train_features(prepared.features["train"], prepared.labels["train"], args.top_k)
    column_names = prepared.summary["retained_feature_columns"]
    save_json(root / "selection.json", {
        "dataset": args.experiment,
        "method": "absolute univariate ROC-AUC on training labels only",
        "top_k": args.top_k,
        "selected_indices": selected.tolist(),
        "selected_columns": [column_names[index] for index in selected],
        "feature_scores": rank[selected].tolist(),
        "test_used_for_selection": False,
    })
    reduced = replace(
        prepared,
        features={name: np.ascontiguousarray(values[:, selected]) for name, values in prepared.features.items()},
        graph_features={name: np.ascontiguousarray(values[:, selected]) for name, values in prepared.graph_features.items()},
    )
    rows = []
    for seed in args.seeds:
        reduced.config = json.loads(json.dumps(config))
        reduced.config["training"]["seed"] = seed
        reduced.config["experiment"]["name"] = f"{run_id}/{args.experiment}/seed_{seed}"
        started = time.perf_counter()
        model, training, pretraining = train_hybrid_model(reduced, args.epochs)
        device = torch_device()
        _, graph_features, labels, edges = torch_data(reduced, device)
        validation_start = len(graph_features["validation"]) - len(labels["validation"])
        model.eval()
        with torch.no_grad():
            logits = model(graph_features["validation"], edges["validation"])["logits"][validation_start:]
        validation_ap = float(average_precision_score(labels["validation"].cpu().numpy(), logits.cpu().numpy()))
        row = {"dataset": args.experiment, "seed": seed, "top_k": args.top_k,
               "validation_ap": validation_ap, "best_epoch": training.best_epoch,
               "pretrain_best_epoch": pretraining.best_epoch if pretraining else None,
               "seconds": time.perf_counter() - started}
        if args.evaluate_test:
            test_start = len(graph_features["test"]) - len(labels["test"])
            with torch.no_grad():
                test_logits = model(graph_features["test"], edges["test"])["logits"][test_start:]
            validation_scores = torch.sigmoid(logits).cpu().numpy()
            test_scores = torch.sigmoid(test_logits).cpu().numpy()
            metrics, operating_metrics = evaluate_scores(reduced, validation_scores, test_scores, "hybrid")
            row["metrics"] = metrics
            row["operating_metrics"] = operating_metrics
            print(f"EXPLORATORY TEST seed={seed}: AP={100*metrics['auc_pr']:.2f}% "
                  f"F1={100*metrics['f1']:.2f}%", flush=True)
        rows.append(row)
        save_json(root / ("exploratory_test.json" if args.evaluate_test else "validation_only.json"), rows)
        print(f"{args.experiment} seed={seed} top_k={args.top_k}: validation AP={100*validation_ap:.2f}% "
              f"best_epoch={training.best_epoch}, time={row['seconds']:.1f}s", flush=True)
        del model, graph_features, labels, edges
    print(f"{'Exploratory test' if args.evaluate_test else 'Validation-only'} results: {root}", flush=True)


if __name__ == "__main__":
    main()
