"""Compare hybrid training strategies by validation AP before evaluating test.

This is exploratory: the historical v2 test period has already been inspected.
For a confirmatory thesis claim, use a fresh temporal holdout after selection.
"""

import argparse
import copy
from datetime import datetime
import json
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from run_experiment import (PROJECT_ROOT, evaluate_scores, prepare_experiment,
                            save_json, torch_data, torch_device, train_hybrid_model)
from src.models.hybrid import HybridGNNAutoencoder
from validate_experiments import build_v2_config
from validation_report import render_hybrid_search_report


CANDIDATES = {"longer_training": 0, "ae_pretrained": 10}


def choose_candidate(records: list[dict]) -> str:
    """Select by validation AP only; test metrics are never read here."""
    means = {
        name: float(np.mean([row["validation_ap"] for row in records if row["candidate"] == name]))
        for name in CANDIDATES
    }
    return max(CANDIDATES, key=lambda name: means[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["ieee", "paysim", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive epochs and unique seeds required")
    torch.set_num_threads(4)
    run_id = "hybrid_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "outputs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "protocol.json", {
        "seeds": args.seeds, "epochs": args.epochs, "candidates": CANDIDATES,
        "selection": "highest mean validation average precision across seeds",
        "caveat": "Exploratory: the historical v2 test period was previously inspected.",
    })
    for dataset in ("ieee", "paysim"):
        if args.experiment not in ("all", dataset):
            continue
        config = build_v2_config(dataset, run_id, args.epochs)
        save_json(root / f"{dataset}_config.json", config)
        print(f"Preparing {dataset}...", flush=True)
        prepared = prepare_experiment(config, None, "window")
        records = []
        for candidate, pretrain_epochs in CANDIDATES.items():
            for seed in args.seeds:
                prepared.config = copy.deepcopy(config)
                prepared.config["training"]["seed"] = seed
                prepared.config["training"]["ae_pretrain_epochs"] = pretrain_epochs
                prepared.config["experiment"]["name"] = f"{run_id}/{dataset}/{candidate}/seed_{seed}"
                started = time.perf_counter()
                model, training, pretraining = train_hybrid_model(prepared, args.epochs)
                device = torch_device()
                _, features, labels, edges = torch_data(prepared, device)
                validation_start = len(features["validation"]) - len(labels["validation"])
                model.eval()
                with torch.no_grad():
                    logits = model(features["validation"], edges["validation"])["logits"][validation_start:]
                validation_ap = float(average_precision_score(labels["validation"].cpu().numpy(), logits.cpu().numpy()))
                checkpoint = root / dataset / candidate / f"seed_{seed}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), checkpoint)
                record = {
                    "dataset": dataset, "candidate": candidate, "seed": seed,
                    "validation_ap": validation_ap, "best_epoch": training.best_epoch,
                    "ae_pretrain_best_epoch": pretraining.best_epoch if pretraining else None,
                    "seconds": time.perf_counter() - started,
                }
                records.append(record)
                save_json(root / dataset / "candidate_validation.json", records)
                print(f"{dataset} {candidate} seed={seed}: validation AP={100 * validation_ap:.2f}% "
                      f"best_epoch={training.best_epoch}", flush=True)
                del model, features, labels, edges
        winner = choose_candidate(records)
        print(f"{dataset}: selected {winner} by validation AP; evaluating test once per seed", flush=True)
        test_rows = []
        for seed in args.seeds:
            prepared.config = copy.deepcopy(config)
            prepared.config["training"]["seed"] = seed
            prepared.config["training"]["ae_pretrain_epochs"] = CANDIDATES[winner]
            prepared.config["experiment"]["name"] = f"{run_id}/{dataset}/selected/seed_{seed}"
            device = torch_device()
            _, features, labels, edges = torch_data(prepared, device)
            dimensions = prepared.config["model"]
            model = HybridGNNAutoencoder(
                features["train"].shape[1], dimensions["hidden_dimension"],
                dimensions["latent_dimension"], dimensions["gnn_layers"], dimensions["dropout"],
            ).to(device)
            checkpoint = root / dataset / winner / f"seed_{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                validation_start = len(features["validation"]) - len(labels["validation"])
                test_start = len(features["test"]) - len(labels["test"])
                validation_scores = torch.sigmoid(model(features["validation"], edges["validation"])["logits"][validation_start:]).cpu().numpy()
                test_scores = torch.sigmoid(model(features["test"], edges["test"])["logits"][test_start:]).cpu().numpy()
            metrics, operating_metrics = evaluate_scores(prepared, validation_scores, test_scores, "hybrid")
            test_rows.append({"dataset": dataset, "candidate": winner, "seed": seed,
                              "metrics": metrics, "operating_metrics": operating_metrics})
            save_json(root / dataset / "selected_test.json", test_rows)
            print(f"{dataset} selected seed={seed}: test AP={100 * metrics['auc_pr']:.2f}% "
                  f"operating recall={100 * operating_metrics['recall']:.2f}%", flush=True)
        (root / dataset / "REPORT.md").write_text(
            render_hybrid_search_report({dataset: (records, test_rows)}), encoding="utf-8"
        )
    print(f"Results: {root}", flush=True)


if __name__ == "__main__":
    main()
