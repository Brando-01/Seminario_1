"""Frozen temporal comparison. Never select configurations using test scores."""
import argparse
import copy
from datetime import datetime
import gc
import json
import platform
import time

import numpy as np
import torch

from run_experiment import (PROJECT_ROOT, prepare_experiment, run_baseline,
                            run_autoencoder, run_gnn, run_hybrid, save_json)
from src.utils import load_config


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
    for dataset, filename in [('ieee', 'experiment_1_ieee.yaml'), ('paysim', 'experiment_2_paysim.yaml')]:
        if args.experiment not in ('all', dataset):
            continue
        config = load_config(PROJECT_ROOT / 'configs' / filename)
        config['experiment']['name'] = f'{run_id}/{dataset}/prepared'
        config['experiment']['variant'] = 'causal_v2'
        config['data']['exclude_features'].append('elapsed_days')
        config['data']['clip_value'] = 10.0
        config['model'].update(hidden_dimension=64, latent_dimension=16, dropout=0.15)
        config['training'].update(epochs=args.epochs, positive_weight_cap=20, reconstruction_weight=0.01)
        if dataset == 'ieee':
            config['data']['development_window'] = {'min': 30*86400, 'max': 60*86400-1}
            config['data']['log_columns'] = ['TransactionAmt']
        else:
            config['data']['development_window'] = {'min': 1, 'max': 24}
            config['data']['exclude_post_transaction_balances'] = True
            config['data']['log_columns'] = ['amount', 'oldbalanceOrg', 'oldbalanceDest']
            config['split'].update(train_before=17*3600, test_from=21*3600)
        save_json(root / f'{dataset}_config.json', config)
        print(f'Preparing {dataset}; protocol saved to {root}', flush=True)
        prepared = prepare_experiment(config, None, 'window')
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
    lines = ['# Temporal validation v2', '', 'AP is average precision, not trapezoidal PR area. Mean +/- sample SD over training seeds; same test set, not independent temporal replications.']
    for dataset in sorted({row['dataset'] for row in rows}):
        lines.extend(['', f'## {dataset}', '', '| Dataset | Model | AP | F1 | Recall |', '|---|---|---:|---:|---:|'])
        means = {}
        for model in ['baseline', 'autoencoder', 'gnn', 'hybrid']:
            selected = [row for row in rows if row['dataset'] == dataset and row['model'] == model]
            values = np.array([[row['metrics'][key] for key in ['auc_pr', 'f1', 'recall']] for row in selected])
            means[model] = values[:, 0].mean()
            cells = [f'{column.mean():.4f} +/- {column.std(ddof=1) if len(column)>1 else 0:.4f}' for column in values.T]
            lines.append(f'| {dataset} | {model} | ' + ' | '.join(cells) + ' |')
        lines.extend(['', f'{dataset}: highest observed mean AP = {max(means, key=means.get)}. Hybrid minus baseline = {means["hybrid"]-means["baseline"]:+.4f}; hybrid minus GNN = {means["hybrid"]-means["gnn"]:+.4f}.', ''])
    lines.extend(['No statistical superiority claim: seeds share a test period. Different windows/features from legacy runs prevent attributing score changes solely to architecture.', 'Thresholds use validation only; the false-positive budget may be exceeded on test. PaySim excludes post-transaction balances and is synthetic.'])
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Results: {root}', flush=True)


if __name__ == '__main__':
    main()
