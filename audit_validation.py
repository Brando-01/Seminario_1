"""Recompute validation-v2 metrics from saved predictions, without retraining."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation import classification_metrics, select_f1_threshold, select_threshold_at_fp_budget


def audit(root: Path, minibatch: bool = False):
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    rows = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    datasets = (protocol.get('evaluated_datasets', []) if minibatch else
                [path.name.removesuffix('_config.json') for path in root.glob('*_config.json')])
    models = ['baseline', 'hybrid'] if minibatch else ['baseline', 'autoencoder', 'gnn', 'hybrid']
    expected = {(dataset, seed, model) for dataset in datasets for seed in protocol['seeds'] for model in models}
    actual = {(row['dataset'], row['seed'], row['model']) for row in rows}
    if not rows or actual != expected or len(rows) != len(expected) or not (root / 'REPORT.md').is_file():
        raise ValueError('Corrida incompleta o duplicada: no se puede certificar su integridad')
    reference = {}
    for row in rows:
        dataset = row['dataset']
        config = json.loads((root / f'{dataset}_config.json').read_text(encoding='utf-8'))
        summary = json.loads((root / dataset / 'prepared/metrics/prepare_summary.json').read_text(encoding='utf-8'))
        base = root / dataset / 'selected' if minibatch else root / dataset
        path = base / f'seed_{row["seed"]}' / 'predictions' / f'{row["model"]}.npz'
        with np.load(path) as arrays:
            if not np.isfinite(arrays['scores']).all() or not np.isfinite(arrays['validation_scores']).all():
                raise ValueError(f'Nonfinite scores: {path}')
            labels = arrays['labels']
            if len(labels) != summary['splits']['test']['rows'] or labels.sum() != summary['splits']['test']['fraud']:
                raise ValueError(f'Test partition mismatch: {path}')
            if arrays['time'].min() <= summary['splits']['validation']['time_max']:
                raise ValueError(f'Temporal overlap: {path}')
            identity = (labels.copy(), arrays['time'].copy(), arrays['validation_labels'].copy())
            if dataset in reference and not all(np.array_equal(a, b) for a, b in zip(reference[dataset], identity)):
                raise ValueError(f'Different evaluation sets: {path}')
            reference[dataset] = identity
            if minibatch and row['model'] == 'hybrid':
                from sklearn.metrics import average_precision_score
                selection = json.loads((root / dataset / 'selection.json').read_text(encoding='utf-8'))
                if selection.get('test_used') or row['candidate'] != selection['candidate']:
                    raise ValueError(f'Selected variant mismatch: {path}')
                validation_ap = average_precision_score(arrays['validation_labels'], arrays['validation_scores'])
                if not np.isclose(validation_ap, row['training']['validation_ap']):
                    raise ValueError(f'Saved validation checkpoint mismatch: {path}')
            thresholds = {
                'metrics': select_f1_threshold(arrays['validation_labels'], arrays['validation_scores']),
                'operating_metrics': select_threshold_at_fp_budget(arrays['validation_labels'], arrays['validation_scores'], config['training']['max_false_positives_per_1000']),
            }
            for key, threshold in thresholds.items():
                calculated = classification_metrics(labels, arrays['scores'], threshold)
                if not all(np.isclose(calculated[name], value) for name, value in row[key].items()):
                    raise ValueError(f'Metrics or validation threshold mismatch: {path} / {key}')
    print(f'AUDITORIA CORRECTA: {len(rows)} resultados; mismas particiones por dataset, scores finitos, umbrales de validacion y metricas reproducidas.')
    print('Esto comprueba integridad, no superioridad estadistica ni ausencia de todo posible sesgo.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='latest')
    parser.add_argument('--minibatch', action='store_true', help='Audita una evaluacion del hibrido por lotes')
    args = parser.parse_args()
    outputs = Path(__file__).resolve().parent / 'outputs'
    if args.run == 'latest':
        prefix = 'minibatch_search_*' if args.minibatch else 'validation_v2_*'
        candidates = sorted(outputs.glob(prefix))
        if args.minibatch:
            candidates = [candidate for candidate in candidates if (candidate / 'results.json').is_file()]
        if not candidates:
            parser.error('No completed runs found')
        directory = candidates[-1]
    else:
        directory = outputs / args.run
    audit(directory, minibatch=args.minibatch)
