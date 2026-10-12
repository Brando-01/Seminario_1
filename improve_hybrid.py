"""Two-stage hybrid pilot: select on validation, then evaluate a frozen run.

Same temporal windows/features as v2. Historical test results are exploratory.
Default: screen two candidates on seed 42, freeze the variant, then repeat it
with the requested seeds. Test predictions require a separate --evaluate-run.
"""

import argparse
import copy
from datetime import datetime
import gc
import json
import platform
import re
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from run_experiment import PROJECT_ROOT, evaluate_scores, output_dir, prepare_experiment, save_json, torch_device
from src.graph.subgraphs import IncomingNeighborhoods
from src.models.baseline import TabularBaseline
from src.models.hybrid import HybridGNNAutoencoder
from src.training.graph_batches import fit_hybrid_batches, predict_hybrid_batches
from src.training.torch_trainer import fit_autoencoder
from src.utils import set_global_seed
from validate_experiments import build_v2_config
from validation_report import render_validation_report


CANDIDATES = {
    "minibatch": {"attention": False, "normalize_fusion": False},
    "attention_norm": {"attention": True, "normalize_fusion": True},
}
RUN_PATTERN = re.compile(r"^minibatch_search_\d{8}_\d{6}$")


def choose_batch_candidate(records):
    names = sorted({row["candidate"] for row in records})
    if not names:
        raise ValueError("Candidate validation records are required")
    means = {name: np.mean([row["validation_ap"] for row in records if row["candidate"] == name]) for name in names}
    return max(names, key=lambda name: means[name])


def make_model(prepared, candidate):
    cfg = prepared.config["model"]
    return HybridGNNAutoencoder(
        prepared.features["train"].shape[1], cfg["hidden_dimension"], cfg["latent_dimension"],
        cfg["gnn_layers"], cfg["dropout"], **CANDIDATES[candidate],
    ).to(torch_device())


def predict_partition(model, prepared, partition, batch_size):
    values = prepared.graph_features[partition]
    index = IncomingNeighborhoods(prepared.graphs[partition].edge_index, len(values))
    targets = np.arange(len(values) - len(prepared.labels[partition]), len(values))
    logits = predict_hybrid_batches(model, values, index, targets, prepared.config["model"]["gnn_layers"], batch_size)
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def train_candidate(prepared, root, dataset, candidate, seed, args):
    prepared.config["training"]["seed"] = seed
    prepared.config["experiment"]["name"] = f"{root.name}/{dataset}/{candidate}/seed_{seed}"
    set_global_seed(seed)
    model = make_model(prepared, candidate)
    cfg = prepared.config["training"]
    tick = time.perf_counter()
    if args.ae_pretrain_epochs:
        fit_autoencoder(
            model.autoencoder, torch.as_tensor(prepared.features["train"], device=torch_device()),
            torch.as_tensor(prepared.labels["train"], device=torch_device()), args.ae_pretrain_epochs,
            cfg["learning_rate"], torch.as_tensor(prepared.features["validation"], device=torch_device()),
            torch.as_tensor(prepared.labels["validation"], device=torch_device()),
            args.batch_size, cfg["early_stopping_patience"],
        )
    pretraining_seconds = time.perf_counter() - tick
    training, timing = fit_hybrid_batches(model, prepared, args.epochs, args.batch_size, args.inference_batch_size)
    directory = root / dataset / candidate / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "hybrid.pt")
    save_json(directory / "history.json", training.history)
    validation_scores = predict_partition(model, prepared, "validation", args.inference_batch_size)
    np.savez_compressed(directory / "validation.npz", labels=prepared.labels["validation"], scores=validation_scores)
    row = {"dataset": dataset, "candidate": candidate, "seed": seed,
           "best_epoch": training.best_epoch, "pretraining_seconds": pretraining_seconds, **timing}
    save_json(directory / "training.json", row)
    print(f"{dataset} {candidate} seed={seed}: BEST val AP={100 * row['validation_ap']:.2f}% epoch={training.best_epoch}", flush=True)
    del model
    gc.collect()
    return row


def search(args):
    run_id = "minibatch_search_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "outputs" / run_id
    root.mkdir(parents=True, exist_ok=False)
    datasets = [args.experiment] if args.experiment != "all" else ["ieee", "paysim"]
    protocol = {
        "datasets": datasets, "epochs": args.epochs, "batch_size": args.batch_size,
        "inference_batch_size": args.inference_batch_size, "ae_pretrain_epochs": args.ae_pretrain_epochs,
        "search_seeds": args.search_seeds, "seeds": args.seeds, "candidates": CANDIDATES, "feature_set": args.feature_set,
        "selection": "mean validation AP over search seeds; freeze before additional seed repetitions",
        "test_evaluated": False, "search_complete": False,
        "python": platform.python_version(), "torch": torch.__version__,
        "trainer": "target minibatches; exact cached first-layer incoming mean; no neighbor sampling",
        "caveat": "Exploratory: v2 historical test was previously inspected. No new temporal holdout.",
    }
    save_json(root / "protocol.json", protocol)
    for dataset in datasets:
        config = build_v2_config(dataset, run_id, args.epochs)
        config["training"]["batch_size"] = args.batch_size
        if args.feature_set == "enriched":
            config["data"].update(add_missing_indicators=True, categorical_encoding="onehot_frequency", max_categories=16)
            config["model"]["hidden_dimension"] = 128
        save_json(root / f"{dataset}_config.json", config)
        print(f"Preparing {dataset}: {root}", flush=True)
        prepared = prepare_experiment(config, None, "window")
        print(json.dumps(prepared.summary["splits"]), flush=True)
        records = []
        for candidate in CANDIDATES:
            for seed in args.search_seeds:
                print(f"{dataset} candidate={candidate} seed={seed}", flush=True)
                records.append(train_candidate(prepared, root, dataset, candidate, seed, args))
                save_json(root / dataset / "candidate_validation.json", records)
        winner = choose_batch_candidate(records)
        # This file is frozen before any additional seed or test evaluation.
        save_json(root / dataset / "selection.json", {"candidate": winner, "search_seeds": args.search_seeds,
                                                      "criterion": "validation AP", "test_used": False})
        print(f"{dataset}: FROZEN variant={winner}", flush=True)
        selected = [row for row in records if row["candidate"] == winner and row["seed"] in args.seeds]
        for seed in args.seeds:
            if seed not in args.search_seeds:
                selected.append(train_candidate(prepared, root, dataset, winner, seed, args))
        save_json(root / dataset / "selected_validation.json", selected)
        del prepared
        gc.collect()
    # Keep any dataset evaluation recorded while the other dataset trained.
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["search_complete"] = True
    save_json(root / "protocol.json", protocol)
    print(f"Validation complete: {root}\nTest remains unevaluated in this run.", flush=True)


def evaluate_run(name, experiment="all"):
    if name == "latest":
        candidates = sorted((PROJECT_ROOT / "outputs").glob("minibatch_search_*"), reverse=True)
        candidates = [p for p in candidates if (p / "protocol.json").is_file() and json.loads((p / "protocol.json").read_text())["search_complete"]]
        if not candidates:
            raise ValueError("No completed minibatch search")
        name = candidates[0].name
    if not RUN_PATTERN.fullmatch(name):
        raise ValueError("Invalid run name")
    root = PROJECT_ROOT / "outputs" / name
    protocol = json.loads((root / "protocol.json").read_text())
    requested = protocol["datasets"] if experiment == "all" else [experiment]
    if not set(requested).issubset(protocol["datasets"]):
        raise ValueError("The requested dataset is not in this run")
    if not protocol["search_complete"] and experiment == "all":
        raise ValueError("Finish validation before evaluating all datasets")
    # A finished dataset can be evaluated while the independent other dataset
    # is still training. Every requested seed and the selected variant must be
    # frozen; merely finding a single checkpoint is not sufficient.
    for dataset in requested:
        selected_file = root / dataset / "selected_validation.json"
        selection_file = root / dataset / "selection.json"
        if not selected_file.is_file() or not selection_file.is_file():
            raise ValueError(f"Finish {dataset} validation before evaluating test")
        selected = json.loads(selected_file.read_text())
        selection = json.loads(selection_file.read_text())
        if (selection.get("test_used") or len(selected) != len(protocol["seeds"])
                or {row["seed"] for row in selected} != set(protocol["seeds"])
                or any(row["candidate"] != selection["candidate"] for row in selected)):
            raise ValueError(f"Incomplete or inconsistent frozen selection for {dataset}")
    result_file = root / "results.json"
    rows = json.loads(result_file.read_text()) if result_file.is_file() else []
    completed = {(row["dataset"], row["seed"], row["model"]) for row in rows}
    summaries = {}
    for dataset in requested:
        if all((dataset, seed, model) in completed for seed in protocol["seeds"] for model in ("hybrid", "baseline")):
            print(f"{dataset}: test already evaluated; using saved results", flush=True)
            continue
        config = json.loads((root / f"{dataset}_config.json").read_text())
        prepared = prepare_experiment(config, None, "window")
        summaries[dataset] = prepared.summary
        winner = json.loads((root / dataset / "selection.json").read_text())["candidate"]
        for seed in protocol["seeds"]:
            prepared.config = copy.deepcopy(config)
            prepared.config["training"]["seed"] = seed
            prepared.config["experiment"]["name"] = f"{root.name}/{dataset}/selected/seed_{seed}"
            directory = root / dataset / winner / f"seed_{seed}"
            if (dataset, seed, "hybrid") not in completed:
                model = make_model(prepared, winner)
                model.load_state_dict(torch.load(directory / "hybrid.pt", map_location=torch_device(), weights_only=True))
                with np.load(directory / "validation.npz") as saved:
                    validation_scores = saved["scores"].copy()
                    if not np.array_equal(saved["labels"], prepared.labels["validation"]):
                        raise ValueError("Dataset validation labels changed since selection")
                tick = time.perf_counter()
                test_scores = predict_partition(model, prepared, "test", protocol["inference_batch_size"])
                inference_seconds = time.perf_counter() - tick
                metrics, operating = evaluate_scores(prepared, validation_scores, test_scores, "hybrid")
                training = json.loads((directory / "training.json").read_text())
                rows.append({"dataset": dataset, "seed": seed, "model": "hybrid", "candidate": winner,
                             "metrics": metrics, "operating_metrics": operating, "inference_seconds": inference_seconds,
                             "seconds": training["train_seconds"] + training["pretraining_seconds"], "training": training})
                save_json(result_file, rows)
                completed.add((dataset, seed, "hybrid"))
                del model
            if (dataset, seed, "baseline") in completed:
                continue
            tick = time.perf_counter()
            baseline = TabularBaseline(random_state=seed).fit(prepared.features["train"], prepared.labels["train"])
            baseline_train_seconds = time.perf_counter() - tick
            baseline_validation = baseline.predict_proba(prepared.features["validation"])
            tick = time.perf_counter()
            baseline_scores = baseline.predict_proba(prepared.features["test"])
            baseline_inference_seconds = time.perf_counter() - tick
            baseline_metrics, baseline_operating = evaluate_scores(prepared, baseline_validation, baseline_scores, "baseline")
            rows.append({"dataset": dataset, "seed": seed, "model": "baseline", "metrics": baseline_metrics,
                         "operating_metrics": baseline_operating, "seconds": baseline_train_seconds,
                         "inference_seconds": baseline_inference_seconds})
            save_json(root / "results.json", rows)
            completed.add((dataset, seed, "baseline"))
            metrics = next(row["metrics"] for row in rows if row["dataset"] == dataset and row["seed"] == seed and row["model"] == "hybrid")
            print(f"EXPLORATORY {dataset} seed={seed}: hybrid AP={100 * metrics['auc_pr']:.2f}% F1={100 * metrics['f1']:.2f}% "
                  f"baseline AP={100 * baseline_metrics['auc_pr']:.2f}% F1={100 * baseline_metrics['f1']:.2f}%", flush=True)
            del baseline
            gc.collect()
        del prepared
        gc.collect()
    for dataset in protocol["datasets"]:
        summary_file = root / dataset / "prepared" / "metrics" / "prepare_summary.json"
        if dataset not in summaries and summary_file.is_file():
            summaries[dataset] = json.loads(summary_file.read_text())
    report = "# Exploracion del hibrido por lotes\n\n" + protocol["caveat"] + "\n\n" + render_validation_report(rows, summaries)
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    # Preserve a search completion update if training finished during testing.
    protocol = json.loads((root / "protocol.json").read_text())
    protocol["evaluated_datasets"] = [dataset for dataset in protocol["datasets"]
                                      if all((dataset, seed, model) in completed for seed in protocol["seeds"] for model in ("hybrid", "baseline"))]
    protocol["test_evaluated"] = len(protocol["evaluated_datasets"]) == len(protocol["datasets"])
    save_json(root / "protocol.json", protocol)
    print(report, flush=True)
    print(f"Results: {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["ieee", "paysim", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--ae-pretrain-epochs", type=int, default=10)
    parser.add_argument("--feature-set", choices=["original", "enriched"], default="original")
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62])
    parser.add_argument("--evaluate-run", help="Evaluate a completed frozen run, e.g. latest")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.inference_batch_size) < 1 or args.ae_pretrain_epochs < 0:
        parser.error("Positive epochs/batch sizes and nonnegative pretraining required")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.search_seeds)) != len(args.search_seeds):
        parser.error("Seeds must be unique")
    torch.set_num_threads(4)
    if args.evaluate_run:
        evaluate_run(args.evaluate_run, args.experiment)
    else:
        search(args)


if __name__ == "__main__":
    main()
