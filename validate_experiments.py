"""Frozen temporal comparison. Never select configurations using test scores."""
import argparse
import copy
from datetime import datetime
import gc
import json
import platform
import time

import torch

from run_experiment import (PROJECT_ROOT, prepare_experiment, run_baseline,
                            run_autoencoder, run_gnn, run_hybrid, save_json)
from src.utils import load_config
from validation_report import render_validation_report


def build_v2_config(dataset: str, run_id: str, epochs: int) -> dict:
    filename = 'experiment_1_ieee.yaml' if dataset == 'ieee' else 'experiment_2_paysim.yaml'
    config = load_config(PROJECT_ROOT / 'configs' / filename)
    config['experiment']['name'] = f'{run_id}/{dataset}/prepared'
    config['experiment']['variant'] = 'causal_v2'
    config['data']['exclude_features'].append('elapsed_days')
    config['data']['clip_value'] = 10.0
    config['model'].update(hidden_dimension=64, latent_dimension=16, dropout=0.15)
    config['training'].update(epochs=epochs, positive_weight_cap=20, reconstruction_weight=0.01)
    if dataset == 'ieee':
        config['data']['development_window'] = {'min': 30*86400, 'max': 60*86400-1}
        config['data']['log_columns'] = ['TransactionAmt']
    else:
        config['data']['development_window'] = {'min': 1, 'max': 24}
        config['data']['exclude_post_transaction_balances'] = True
        config['data']['log_columns'] = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
        config['split'].update(train_before=17*3600, test_from=21*3600)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all', 'ieee', 'paysim'], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62])
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epochs and unique seeds required')
    torch.set_num_threads(4)
    run_id = 'validation_v2_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    root = PROJECT_ROOT / 'outputs' / run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = {'epochs': args.epochs, 'seeds': args.seeds, 'python': platform.python_version(),
                'torch': torch.__version__, 'primary_metric': 'average_precision',
                'selection': 'validation AP for supervised networks; validation normal MSE for AE',
                'scope': 'Exploratory fixed temporal windows, not full-dataset or cross-domain transfer validation'}
    save_json(root / 'protocol.json', manifest)
    rows = []
    summaries = {}
    for dataset in ('ieee', 'paysim'):
        if args.experiment not in ('all', dataset):
            continue
        config = build_v2_config(dataset, run_id, args.epochs)
        save_json(root / f'{dataset}_config.json', config)
        print(f'Preparing {dataset}; protocol saved to {root}', flush=True)
        prepared = prepare_experiment(config, None, 'window')
        summaries[dataset] = prepared.summary
        print(json.dumps(prepared.summary['splits']), flush=True)
        for seed in args.seeds:
            prepared.config = copy.deepcopy(config)
            prepared.config['experiment']['name'] = f'{run_id}/{dataset}/seed_{seed}'
            prepared.config['training']['seed'] = seed
            for name, function in [('baseline', run_baseline), ('autoencoder', run_autoencoder), ('gnn', run_gnn), ('hybrid', run_hybrid)]:
                start = time.perf_counter()
                print(f'{dataset} seed={seed} model={name}: training', flush=True)
                result = function(prepared) if name == 'baseline' else function(prepared, args.epochs)
                rows.append({'dataset': dataset, 'seed': seed, 'model': name, 'seconds': time.perf_counter()-start, **result})
                # Keep stable comparison key separately from the descriptive model name.
                rows[-1]['model'] = name
                save_json(root / 'results.json', rows)
                print(f'{dataset} seed={seed} {name}: AP={result["metrics"]["auc_pr"]:.5f} F1={result["metrics"]["f1"]:.5f}', flush=True)
                gc.collect()
        del prepared
        gc.collect()
    report = render_validation_report(rows, summaries)
    (root / 'REPORT.md').write_text(report, encoding='utf-8')
    print(report, flush=True)
    print(f'Results: {root}', flush=True)


if __name__ == '__main__':
    main()
