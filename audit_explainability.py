"""Reproduce SHAP artifact additivity, sample ranking and prediction parity."""
import argparse
import numpy as np
from thesis_experiment import resolve_run, read_json, save_json, audit_run


def audit_explanations(root):
    audit_run(root)
    protocol = read_json(root / 'protocol.json')
    if not protocol['shap_complete']:
        raise ValueError('SHAP generation is incomplete')
    verified = []
    for dataset in protocol['datasets']:
        folder = root / dataset / 'explanations'
        metadata = read_json(folder / 'summary.json')
        preparation = read_json(root / dataset / 'prepared' / 'metrics' / 'prepare_summary.json')
        seed = protocol['explanations']['seed']
        with np.load(folder / 'shap_values.npz') as saved, np.load(root / dataset / f'seed_{seed}' / 'hybrid' / 'predictions.npz') as predictions:
            values, base, scores = saved['values'], saved['base_values'], saved['scores']
            indices = saved['test_indices']
            train_indices = saved['background_train_indices']
            if not np.isfinite(values).all() or not np.isfinite(base).all():
                raise ValueError('Nonfinite SHAP values')
            if not np.allclose(base + values.sum(axis=1), scores, atol=1e-5, rtol=1e-5):
                raise ValueError('SHAP additivity failed')
            if not np.allclose(scores, predictions['scores'][indices], atol=1e-6, rtol=1e-5):
                raise ValueError('Explanations differ from test predictions')
            if train_indices.min() < 0 or train_indices.max() >= preparation['splits']['train']['rows']:
                raise ValueError('Background reference indices outside train')
            count = len(metadata['random_test_sample'])
            if indices[:count].tolist() != metadata['random_test_sample']:
                raise ValueError('Global sample selection differs from saved explanations')
            aggregate = np.abs(values[:count]).mean(axis=0)
            mapping = dict(zip(saved['feature_names'], aggregate))
            for record in metadata['top_features']:
                if not np.isclose(mapping[record['feature']], record['mean_absolute_shap']):
                    raise ValueError('SHAP global ranking is not reproduced')
            if len(metadata['group_members']) != values.shape[1]:
                raise ValueError('Feature groups inconsistent with attribution shape')
            verified.append({'dataset': dataset, 'explained_cases': len(values), 'groups': values.shape[1],
                             'global_random_sample': count, 'max_additivity_error': float(np.max(np.abs(base + values.sum(1) - scores)))})
    result = {'passed': True, 'datasets': verified,
              'scope': 'Numeric integrity of conditional grouped SHAP, not proof of causal validity or global population representativeness.'}
    save_json(root / 'audit_shap.json', result)
    print(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='latest')
    audit_explanations(resolve_run(parser.parse_args().run))
