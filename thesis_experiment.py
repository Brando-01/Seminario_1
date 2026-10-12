"""Complete common-protocol experiment: four models, costs, audit and SHAP.

Defaults are frozen before data preparation. Historical pilot scores are NOT
used to choose new test thresholds. This is a bounded thesis experiment, not a
claim of production readiness, cross-domain transfer, or statistical superiority.
"""
import argparse
import copy
from datetime import datetime
from zoneinfo import ZoneInfo
import gc
import hashlib
import json
from pathlib import Path
import pickle
import platform
import re
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix
from run_experiment import PROJECT_ROOT, prepare_experiment, save_json, torch_device
from src.evaluation.metrics import classification_metrics, select_f1_threshold, select_threshold_at_fp_budget
from src.graph.subgraphs import IncomingNeighborhoods
from src.models.autoencoder import Autoencoder
from src.models.baseline import TabularBaseline
from src.models.gnn_classifier import GNNFraudClassifier
from src.models.hybrid import HybridGNNAutoencoder
from src.training.gnn_batches import fit_gnn_batches, predict_gnn_batches, synchronize
from src.training.graph_batches import fit_hybrid_batches, predict_hybrid_batches
from src.training.torch_trainer import fit_autoencoder
from src.utils import set_global_seed
from validate_experiments import build_v2_config
from validation_report import MODEL_ORDER, render_validation_report

RUN_PATTERN = re.compile(r'^thesis_v1_\d{8}_\d{6}$')
DATASETS = ('ieee', 'paysim')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def build_thesis_config(dataset, run, epochs):
    config = build_v2_config(dataset, run, epochs)
    config['experiment'].update(name=f'{run}/{dataset}/prepared', variant='thesis_v1_frozen')
    config['data'].update(add_missing_indicators=True, categorical_encoding='onehot_frequency', max_categories=16)
    config['model'].update(hidden_dimension=128, latent_dimension=16, dropout=0.15, attention=True, normalize_fusion=True)
    config['training'].update(batch_size=4096, inference_batch_size=8192, ae_pretrain_epochs=10)
    if dataset == 'ieee':
        config['data']['development_window'] = {'min': 30 * 86400, 'max': 75 * 86400 - 1}
        config['split'].update(train_before=50 * 86400, test_from=60 * 86400)
    else:
        config['data'].update(development_window={'min': 1, 'max': 72},
                              exclude_balance_columns=True, exclude_flagged_fraud=True,
                              add_balance_consistency_features=False, add_causal_history_features=True,
                              log_columns=['amount'])
        config['split'].update(train_before=17 * 3600, test_from=25 * 3600)
    return config


def fingerprint_configs(root, datasets):
    return {dataset: digest(root / f'{dataset}_config.json') for dataset in datasets}


def fingerprint_sources():
    paths = [PROJECT_ROOT / 'run_experiment.py', PROJECT_ROOT / 'thesis_experiment.py']
    paths += sorted((PROJECT_ROOT / 'src').rglob('*.py'))
    return {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in paths}


def fingerprint_data(root, datasets):
    result = {}
    for dataset in datasets:
        cfg = read_json(root / f'{dataset}_config.json')['data']
        for key in ('path', 'transaction_path', 'identity_path'):
            if key in cfg:
                path = PROJECT_ROOT / cfg[key]
                result[cfg[key]] = digest(path)
    return result


def create_run(args):
    run = 'thesis_v1_' + datetime.now(ZoneInfo('America/Lima')).strftime('%Y%m%d_%H%M%S')
    root = PROJECT_ROOT / 'outputs' / run
    root.mkdir(parents=True, exist_ok=False)
    datasets = list(DATASETS) if args.experiment == 'all' else [args.experiment]
    for dataset in datasets:
        config = build_thesis_config(dataset, run, args.epochs)
        config['training']['ae_pretrain_epochs'] = args.ae_pretrain_epochs
        save_json(root / f'{dataset}_config.json', config)
    protocol = {
        'schema': 'thesis_v1', 'datasets': datasets, 'models': list(MODEL_ORDER), 'seeds': args.seeds,
        'epochs': args.epochs, 'created_at': datetime.now(ZoneInfo('America/Lima')).isoformat(),
        'feature_set': 'enriched_common_strict_paysim', 'primary_metric': 'average_precision',
        'train_complete': False, 'test_evaluated': False, 'shap_complete': False,
        'python': platform.python_version(), 'torch': torch.__version__, 'device': str(torch_device()),
        'selection': 'Fixed architecture; supervised checkpoint by validation AP; AE checkpoint by validation normal MSE; thresholds frozen on validation.',
        'holdout': {'ieee': 'TransactionDT days [60,75)', 'paysim': 'step 25..72'},
        'caveat': 'Frozen prospective local protocol on later time windows; prior access to these records outside the recorded pilots cannot be certified. A single period and three seeds do not establish statistical superiority.',
        'timing_scope': 'Training optimization (AE pretraining included for hybrid), validation separately; test score inference including graph-context preparation; excludes CSV loading, general preprocessing, checkpoint I/O and SHAP.',
        'shap_scope': 'Approximate grouped SHAP of transformed target features through the actual hybrid; past features and graph fixed; not graph-edge or causal attribution.',
        'config_sha256': fingerprint_configs(root, datasets), 'source_sha256': fingerprint_sources(),
        'explanations': {'random_sample': args.shap_samples, 'background_train_rows': args.shap_background,
                         'permutations': args.shap_permutations, 'seed': args.seeds[0]},
    }
    save_json(root / 'protocol.json', protocol)
    print(f'FROZEN PROTOCOL BEFORE PREPARATION: {root}', flush=True)
    protocol['data_sha256'] = fingerprint_data(root, datasets)
    save_json(root / 'protocol.json', protocol)
    return root


def resolve_run(name):
    if name == 'latest':
        paths = sorted((PROJECT_ROOT / 'outputs').glob('thesis_v1_*'))
        if not paths:
            raise ValueError('No thesis runs available')
        return paths[-1]
    if not RUN_PATTERN.fullmatch(name):
        raise ValueError('Invalid thesis run name')
    root = PROJECT_ROOT / 'outputs' / name
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def check_frozen(root):
    protocol = read_json(root / 'protocol.json')
    if fingerprint_configs(root, protocol['datasets']) != protocol['config_sha256']:
        raise ValueError('Frozen configurations changed; create a new run')
    if fingerprint_sources() != protocol['source_sha256']:
        raise ValueError('Frozen model/pipeline source changed; create a new run')
    if fingerprint_data(root, protocol['datasets']) != protocol['data_sha256']:
        raise ValueError('Raw data changed since protocol freeze')
    return protocol


def make_model(name, prepared, seed):
    cfg = prepared.config['model']; dim = prepared.features['train'].shape[1]
    if name == 'baseline':
        return TabularBaseline(random_state=seed)
    if name == 'autoencoder':
        model = Autoencoder(dim, cfg['hidden_dimension'], cfg['latent_dimension'], cfg['dropout'], cfg['attention'])
    elif name == 'gnn':
        model = GNNFraudClassifier(dim, cfg['hidden_dimension'], cfg['gnn_layers'], cfg['dropout'])
    else:
        model = HybridGNNAutoencoder(dim, cfg['hidden_dimension'], cfg['latent_dimension'], cfg['gnn_layers'],
                                   cfg['dropout'], cfg['attention'], cfg['normalize_fusion'])
    return model.to(torch_device())


def predict(name, model, prepared, partition):
    size = prepared.config['training']['inference_batch_size']
    if name == 'baseline':
        return model.predict_proba(prepared.features[partition])
    model.eval()
    if name == 'autoencoder':
        values = prepared.features[partition]; result = []
        with torch.no_grad():
            for start in range(0, len(values), size):
                x = torch.as_tensor(values[start:start + size], device=torch_device())
                result.append((x - model(x)[0]).abs().mean(1).cpu().numpy())
        return np.concatenate(result)
    x = prepared.graph_features[partition]
    index = IncomingNeighborhoods(prepared.graphs[partition].edge_index, len(x))
    targets = np.arange(len(x) - len(prepared.labels[partition]), len(x))
    function = predict_hybrid_batches if name == 'hybrid' else predict_gnn_batches
    logits = function(model, x, index, targets, prepared.config['model']['gnn_layers'], size)
    return torch.sigmoid(torch.from_numpy(logits)).numpy()


def fit_one(name, model, prepared, epochs):
    cfg = prepared.config['training']; device = torch_device()
    if name == 'baseline':
        tick = time.perf_counter()
        model.fit(prepared.features['train'], prepared.labels['train'])
        return [], {'train_seconds': time.perf_counter() - tick, 'validation_seconds': 0.0}
    if name in ('autoencoder', 'hybrid'):
        ae = model if name == 'autoencoder' else model.autoencoder
        steps = epochs if name == 'autoencoder' else cfg['ae_pretrain_epochs']
        pretraining = None
        if steps:
            pretraining = fit_autoencoder(ae,
                torch.as_tensor(prepared.features['train'], device=device), torch.as_tensor(prepared.labels['train'], device=device),
                steps, cfg['learning_rate'], torch.as_tensor(prepared.features['validation'], device=device),
                torch.as_tensor(prepared.labels['validation'], device=device), cfg['batch_size'], cfg['early_stopping_patience'])
        if name == 'autoencoder':
            return pretraining.history, {**pretraining.timing, 'best_epoch': pretraining.best_epoch}
    function = fit_hybrid_batches if name == 'hybrid' else fit_gnn_batches
    training, timing = function(model, prepared, epochs, cfg['batch_size'], cfg['inference_batch_size'])
    timing['best_epoch'] = training.best_epoch
    if name == 'hybrid' and pretraining is not None:
        timing['ae_pretraining_seconds'] = pretraining.timing['train_seconds']
        timing['train_seconds'] += pretraining.timing['train_seconds']
        timing['validation_seconds'] += pretraining.timing['validation_seconds']
    return training.history, timing


def train_run(root):
    protocol = check_frozen(root)
    if protocol['test_evaluated']:
        raise ValueError('Training forbidden after evaluating test; create a new run')
    records_file = root / 'training_results.json'
    records = read_json(records_file) if records_file.exists() else []
    for dataset in protocol['datasets']:
        prepared = prepare_experiment(read_json(root / f'{dataset}_config.json'), None, 'window')
        print(f'{dataset}: common splits {prepared.summary["splits"]}', flush=True)
        for seed in protocol['seeds']:
            prepared.config['training']['seed'] = seed
            for name in MODEL_ORDER:
                directory = root / dataset / f'seed_{seed}' / name
                if any(row['dataset'] == dataset and row['seed'] == seed and row['model'] == name for row in records):
                    print(f'Using frozen training: {dataset}/{seed}/{name}', flush=True); continue
                directory.mkdir(parents=True, exist_ok=True)
                print(f'TRAIN {dataset} seed={seed} {name}', flush=True)
                set_global_seed(seed); model = make_model(name, prepared, seed)
                history, timing = fit_one(name, model, prepared, protocol['epochs'])
                synchronize(torch_device()); tick = time.perf_counter()
                scores = predict(name, model, prepared, 'validation')
                synchronize(torch_device()); timing['validation_seconds'] += time.perf_counter() - tick
                labels = prepared.labels['validation']
                checkpoint = directory / ('model.pkl' if name == 'baseline' else 'model.pt')
                if name == 'baseline':
                    with checkpoint.open('wb') as stream:
                        pickle.dump(model, stream)
                else:
                    torch.save(model.state_dict(), checkpoint)
                np.savez_compressed(directory / 'validation.npz', labels=labels, scores=scores)
                row = {'dataset': dataset, 'seed': seed, 'model': name, 'timing': timing,
                       'validation_ap': float(average_precision_score(labels, scores)),
                       'threshold': select_f1_threshold(labels, scores),
                       'operating_threshold': select_threshold_at_fp_budget(labels, scores, prepared.config['training']['max_false_positives_per_1000']),
                       'checkpoint_sha256': digest(checkpoint), 'validation_sha256': digest(directory / 'validation.npz')}
                save_json(directory / 'history.json', history); save_json(directory / 'frozen_validation.json', row)
                records.append(row); save_json(records_file, records)
                print(f'FROZEN {dataset}/{seed}/{name}: val AP={100 * row["validation_ap"]:.2f}%', flush=True)
                del model; gc.collect()
        del prepared; gc.collect()
    expected = len(protocol['datasets']) * len(protocol['seeds']) * len(MODEL_ORDER)
    protocol['train_complete'] = len(records) == expected
    protocol['frozen_training_sha256'] = digest(records_file)
    save_json(root / 'protocol.json', protocol)


def validate_checkpoint(directory, row):
    checkpoint = directory / ('model.pkl' if row['model'] == 'baseline' else 'model.pt')
    if digest(checkpoint) != row['checkpoint_sha256'] or digest(directory / 'validation.npz') != row['validation_sha256']:
        raise ValueError('Checkpoint or validation scores changed since freeze')
    return checkpoint


def load_model(name, prepared, seed, directory, row):
    checkpoint = validate_checkpoint(directory, row)
    if name == 'baseline':
        with checkpoint.open('rb') as stream:
            return pickle.load(stream)  # Only locally generated, hash-verified checkpoints.
    model = make_model(name, prepared, seed)
    model.load_state_dict(torch.load(checkpoint, map_location=torch_device(), weights_only=True))
    return model.eval()


def render_report(root):
    protocol = read_json(root / 'protocol.json')
    rows = read_json(root / 'results.json')
    summaries = {dataset: read_json(root / dataset / 'prepared' / 'metrics' / 'prepare_summary.json') for dataset in protocol['datasets']}
    report = '# Experimento completo de tesis\n\n' + protocol['caveat'] + '\n\n'
    report += render_validation_report(rows, summaries)
    report += '\n## Alcance del costo\n\n' + protocol['timing_scope'] + '\n'
    report += '\n## Explicabilidad\n\n' + protocol['shap_scope'] + '\n'
    report += '\nLos resultados no garantizan que el híbrido sea superior. No volver a seleccionar con test.\n'
    (root / 'REPORT.md').write_text(report, encoding='utf-8')
    return report


def evaluate_run(root):
    protocol = check_frozen(root)
    if not protocol['train_complete'] or digest(root / 'training_results.json') != protocol['frozen_training_sha256']:
        raise ValueError('Complete and freeze ALL training before test evaluation')
    frozen = read_json(root / 'training_results.json')
    rows = read_json(root / 'results.json') if (root / 'results.json').exists() else []
    for dataset in protocol['datasets']:
        prepared = prepare_experiment(read_json(root / f'{dataset}_config.json'), None, 'window')
        for row in [r for r in frozen if r['dataset'] == dataset]:
            name, seed = row['model'], row['seed']
            directory = root / dataset / f'seed_{seed}' / name
            if any(r['dataset'] == dataset and r['seed'] == seed and r['model'] == name for r in rows):
                continue
            model = load_model(name, prepared, seed, directory, row)
            with np.load(directory / 'validation.npz') as saved:
                if not np.array_equal(saved['labels'], prepared.labels['validation']):
                    raise ValueError('Validation partition changed')
                validation_scores = saved['scores'].copy()
            synchronize(torch_device()); tick = time.perf_counter()
            scores = predict(name, model, prepared, 'test')
            synchronize(torch_device()); elapsed = time.perf_counter() - tick
            labels = prepared.labels['test']
            metrics = classification_metrics(labels, scores, row['threshold'])
            operating = classification_metrics(labels, scores, row['operating_threshold'])
            np.savez_compressed(directory / 'predictions.npz', labels=labels, scores=scores,
                validation_labels=prepared.labels['validation'], validation_scores=validation_scores,
                time=prepared.partitions.test[prepared.bundle.time_column].to_numpy())
            result = {'dataset': dataset, 'seed': seed, 'model': name, 'metrics': metrics,
                      'operating_metrics': operating, 'seconds': row['timing']['train_seconds'],
                      'inference_seconds': elapsed, 'validation_seconds': row['timing']['validation_seconds'],
                      'confusion_matrix': confusion_matrix(labels, scores >= row['threshold'], labels=[0, 1]).tolist()}
            rows.append(result); save_json(root / 'results.json', rows)
            print(f'TEST {dataset}/{seed}/{name}: AP={100 * metrics["auc_pr"]:.2f}% F1={100 * metrics["f1"]:.2f}%', flush=True)
            del model; gc.collect()
        del prepared; gc.collect()
    protocol['test_evaluated'] = len(rows) == len(frozen)
    save_json(root / 'protocol.json', protocol)
    print(render_report(root), flush=True)


def audit_run(root):
    protocol = check_frozen(root)
    if not protocol['train_complete'] or not protocol['test_evaluated']:
        raise ValueError('Incomplete training or test')
    if digest(root / 'training_results.json') != protocol['frozen_training_sha256']:
        raise ValueError('Frozen training manifest changed')
    frozen = read_json(root / 'training_results.json'); rows = read_json(root / 'results.json')
    expected = {(d, s, m) for d in protocol['datasets'] for s in protocol['seeds'] for m in MODEL_ORDER}
    if len(rows) != len(expected) or {(r['dataset'], r['seed'], r['model']) for r in rows} != expected:
        raise ValueError('Missing or duplicate model/seed results')
    references = {}
    for row in rows:
        key = (row['dataset'], row['seed'], row['model'])
        selected = next(r for r in frozen if (r['dataset'], r['seed'], r['model']) == key)
        directory = root / key[0] / f'seed_{key[1]}' / key[2]
        validate_checkpoint(directory, selected)
        with np.load(directory / 'predictions.npz') as arrays, np.load(directory / 'validation.npz') as validation:
            for name in ('labels', 'scores', 'validation_labels', 'validation_scores', 'time'):
                if not np.isfinite(arrays[name]).all():
                    raise ValueError('Nonfinite saved scores or labels')
            if not np.array_equal(arrays['validation_scores'], validation['scores']) or not np.array_equal(arrays['validation_labels'], validation['labels']):
                raise ValueError('Validation predictions replaced after freeze')
            current = (arrays['labels'].copy(), arrays['time'].copy(), arrays['validation_labels'].copy())
            if key[0] in references and any(not np.array_equal(a, b) for a, b in zip(current, references[key[0]])):
                raise ValueError('Different temporal test/validation partitions between models')
            references[key[0]] = current
            for field, threshold in [('metrics', selected['threshold']), ('operating_metrics', selected['operating_threshold'])]:
                calculated = classification_metrics(arrays['labels'], arrays['scores'], threshold)
                if any(not np.isclose(value, row[field][name]) for name, value in calculated.items()):
                    raise ValueError('Stored metrics or frozen threshold mismatch')
            cm = confusion_matrix(arrays['labels'], arrays['scores'] >= selected['threshold'], labels=[0, 1]).tolist()
            if cm != row['confusion_matrix'] or min(row['seconds'], row['inference_seconds'], row['validation_seconds']) < 0:
                raise ValueError('Invalid confusion matrix or timing')
    record = {'passed': True, 'rows': len(rows), 'same_partitions': True, 'frozen_hashes': True,
              'metrics_reproduced': True, 'caveat': 'Integrity check, not a proof of universal leakage absence or statistical superiority.'}
    save_json(root / 'audit.json', record)
    print(f'AUDITORIA CORRECTA: {len(rows)} resultados, cuatro modelos, particiones y umbrales congelados.', flush=True)
    return record


def explain_run(root):
    from src.explainability.report import generate_shap_report
    protocol = check_frozen(root)
    audit_run(root)
    seed = protocol['explanations']['seed']
    frozen = read_json(root / 'training_results.json')
    for dataset in protocol['datasets']:
        prepared = prepare_experiment(read_json(root / f'{dataset}_config.json'), None, 'window')
        directory = root / dataset / f'seed_{seed}' / 'hybrid'
        row = next(r for r in frozen if r['dataset'] == dataset and r['seed'] == seed and r['model'] == 'hybrid')
        model = load_model('hybrid', prepared, seed, directory, row)
        generate_shap_report(model, prepared, directory, root / dataset / 'explanations', protocol['explanations'], row['threshold'])
        del model, prepared; gc.collect()
    protocol['shap_complete'] = True
    save_json(root / 'protocol.json', protocol)
    render_report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train', 'evaluate', 'explain', 'audit', 'report', 'all'], default='all')
    parser.add_argument('--run', help='Existing thesis_v1_... or latest; allows resume without retraining completed models')
    parser.add_argument('--experiment', choices=['all', *DATASETS], default='all')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--ae-pretrain-epochs', type=int, default=10)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62])
    parser.add_argument('--shap-samples', type=int, default=16)
    parser.add_argument('--shap-background', type=int, default=8)
    parser.add_argument('--shap-permutations', type=int, default=3)
    args = parser.parse_args()
    if min(args.epochs, args.shap_samples, args.shap_background, args.shap_permutations) < 1 or args.ae_pretrain_epochs < 0:
        parser.error('Positive experiment sizes and nonnegative pretraining required')
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Nonempty unique seeds required')
    if not args.run and args.stage not in ('train', 'all'):
        parser.error('--run required for this stage')
    torch.set_num_threads(4)
    root = resolve_run(args.run) if args.run else create_run(args)
    if args.stage in ('train', 'all'):
        train_run(root)
    if args.stage in ('evaluate', 'all'):
        evaluate_run(root)
    if args.stage in ('audit', 'all'):
        audit_run(root)
    if args.stage in ('explain', 'all'):
        explain_run(root)
    if args.stage == 'report':
        print(render_report(root))
    print(f'RUN: {root}', flush=True)


if __name__ == '__main__':
    main()
