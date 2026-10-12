"""Reproducible SHAP samples, local cases, global-sample ranking and figures."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from src.graph.subgraphs import IncomingNeighborhoods
from .conditional_shap import feature_groups, explain_target


def choose_cases(labels, scores, threshold, sample_count, seed):
    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(labels), min(sample_count, len(labels)), replace=False).tolist()
    flags = scores >= threshold
    cases = {}
    for name, mask in [('TP', (labels == 1) & flags), ('FP', (labels == 0) & flags),
                       ('FN', (labels == 1) & ~flags), ('TN', (labels == 0) & ~flags)]:
        candidates = np.flatnonzero(mask)
        if len(candidates):
            cases[name] = int(candidates[0])
    return random_indices, cases


def generate_shap_report(model, prepared, model_directory, folder, settings, threshold):
    import shap
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_experiment import save_json
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    tick = time.perf_counter()
    with np.load(model_directory / 'predictions.npz') as saved:
        labels, scores = saved['labels'].copy(), saved['scores'].copy()
    random_indices, cases = choose_cases(labels, scores, threshold, settings['random_sample'], settings['seed'])
    selected = list(dict.fromkeys(random_indices + list(cases.values())))
    rng = np.random.default_rng(settings['seed'])
    train = prepared.features['train']
    background_indices = rng.choice(len(train), min(settings['background_train_rows'], len(train)), replace=False)
    background = train[background_indices]
    names, groups = feature_groups(prepared.summary['retained_feature_columns'])
    x = prepared.graph_features['test']
    index = IncomingNeighborhoods(prepared.graphs['test'].edge_index, len(x))
    means = index.mean_features(x)
    start = len(x) - len(labels)
    values, bases, predicted, residuals = [], [], [], []
    for number, local in enumerate(selected, 1):
        print(f'SHAP {prepared.bundle.name} case {number}/{len(selected)} test-index={local}', flush=True)
        result, base, probability, residual = explain_target(model, x, index, means, start + local,
            prepared.config['model']['gnn_layers'], background, groups, names,
            settings['seed'], settings['permutations'])
        if not np.isclose(probability, scores[local], rtol=1e-5, atol=1e-6):
            raise ValueError('SHAP adapter does not reproduce the evaluated hybrid prediction')
        values.append(result); bases.append(base); predicted.append(probability); residuals.append(residual)
    values = np.asarray(values)
    global_mean = np.abs(values[:len(random_indices)]).mean(axis=0)
    ranking = np.argsort(global_mean)[::-1]
    np.savez_compressed(folder / 'shap_values.npz', values=values, base_values=bases, scores=predicted,
                        test_indices=selected, feature_names=np.asarray(names), background_train_indices=background_indices)
    global_features = [{'feature': names[i], 'mean_absolute_shap': float(global_mean[i])} for i in ranking]
    metadata = {
        'method': 'shap.PermutationExplainer', 'shap_version': shap.__version__,
        'scope': 'Grouped transformed target attributes through the actual hybrid, conditional on fixed past features and graph. Not edge, neighbor-feature, causal or raw-unit attribution.',
        'units': 'Fraud score probability; positive contributions increase the score.',
        'background_source': 'train only, uniform seeded sample', 'background_train_indices': background_indices.tolist(),
        'group_count': len(groups), 'group_members': dict(zip(names, groups)),
        'permutations': settings['permutations'], 'seed': settings['seed'],
        'random_test_sample': random_indices, 'illustrative_local_cases': cases,
        'global_scope': f'Mean absolute SHAP on {len(random_indices)} uniformly sampled test transactions, not the whole population. Illustrative TP/FP/FN/TN extras excluded from global aggregation.',
        'max_additivity_error': max(residuals), 'prediction_parity_checked': True,
        'seconds': time.perf_counter() - tick, 'top_features': global_features[:20],
        'local': [{'test_index': local, 'actual_label': int(labels[local]), 'base_value': bases[i], 'score': predicted[i],
                   'additivity_error': residuals[i]} for i, local in enumerate(selected)],
    }
    save_json(folder / 'summary.json', metadata)
    top = ranking[:20][::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh([names[i] for i in top], global_mean[top], color='#315c76')
    ax.set_xlabel('Media de |SHAP| en unidades de probabilidad')
    ax.set_title(f'{prepared.bundle.name} · SHAP agrupado · muestra aleatoria n={len(random_indices)}')
    fig.tight_layout(); fig.savefig(folder / 'global_importance.png', dpi=160); plt.close(fig)
    for case, local in cases.items():
        position = selected.index(local)
        explanation = shap.Explanation(values=values[position], base_values=bases[position], feature_names=names)
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.gcf().set_size_inches(10, 7)
        plt.title(f'{prepared.bundle.name} · {case} · score={predicted[position]:.4f}')
        plt.tight_layout(); plt.savefig(folder / f'local_{case}.png', dpi=160, bbox_inches='tight'); plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 5))
    seed_directory = model_directory.parent
    for name in ('baseline', 'autoencoder', 'gnn', 'hybrid'):
        with np.load(seed_directory / name / 'predictions.npz') as saved:
            precision, recall, _ = precision_recall_curve(saved['labels'], saved['scores'])
        ax.plot(recall, precision, label=name, linewidth=1.5)
    ax.set(xlabel='Recall', ylabel='Precision', title=f'{prepared.bundle.name} · semilla {settings["seed"]} · no curva promedio')
    ax.legend(); fig.tight_layout(); fig.savefig(folder / 'precision_recall.png', dpi=160); plt.close(fig)
    report = '# Explicabilidad del híbrido\n\n' + metadata['scope'] + '\n\n' + metadata['global_scope']
    report += f'\n\nFondo: {len(background)} filas de train. Permutaciones: {settings["permutations"]}. Error máximo de aditividad: {max(residuals):.8g}.\n\n'
    report += '![Importancia agregada](global_importance.png)\n\n'
    for case in cases:
        report += f'## Caso {case}\n\n![SHAP local](local_{case}.png)\n\n'
    report += 'La contribución indica sensibilidad del modelo bajo este juego de perturbación, no causalidad ni justificación humana del fraude. Las aproximaciones dependen del fondo y del número de permutaciones.\n'
    (folder / 'REPORT_SHAP.md').write_text(report, encoding='utf-8')
    return metadata
