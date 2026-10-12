from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
import thesis_experiment as runner
from src.graph.subgraphs import IncomingNeighborhoods
from src.models.hybrid import HybridGNNAutoencoder
from src.models.gnn_classifier import GNNFraudClassifier
from src.training.gnn_batches import predict_gnn_batches, fit_gnn_batches
from src.explainability.conditional_shap import feature_groups, FixedContextPredictor, GroupCoalitions, explain_target
from src.explainability.report import choose_cases


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_independent_gnn_batches_match_full_graph(depth):
    torch.manual_seed(3)
    x = np.random.default_rng(3).normal(size=(10, 5)).astype(np.float32)
    edges = np.array([[0, 1, 2, 1, 4, 6], [1, 3, 3, 4, 7, 8]])
    model = GNNFraudClassifier(5, 8, depth, dropout=0).eval()
    targets = np.array([8, 7, 9, 0])
    with torch.no_grad():
        expected = model(torch.from_numpy(x), torch.from_numpy(edges))[targets].numpy()
    actual = predict_gnn_batches(model, x, IncomingNeighborhoods(edges, len(x)), targets, depth, batch_size=2)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('depth', [1, 2, 3])
def test_shap_fixed_context_matches_real_hybrid_for_perturbations(depth):
    torch.manual_seed(4)
    x = np.random.default_rng(4).normal(size=(9, 5)).astype(np.float32)
    edges = np.array([[0, 1, 2, 3, 1, 4, 5], [2, 3, 4, 4, 5, 6, 7]])
    model = HybridGNNAutoencoder(5, 8, 3, depth, dropout=0, attention=True, normalize_fusion=True).eval()
    index = IncomingNeighborhoods(edges, len(x))
    predictor = FixedContextPredictor(model, x, index, index.mean_features(x), 7, depth, batch_size=2)
    changed = np.random.default_rng(8).normal(size=(4, 5)).astype(np.float32)
    expected = []
    with torch.no_grad():
        for row in changed:
            copy = x.copy(); copy[7] = row
            expected.append(torch.sigmoid(model(torch.from_numpy(copy), torch.from_numpy(edges))['logits'][7]).item())
    np.testing.assert_allclose(predictor(changed), expected, rtol=1e-5, atol=1e-6)


def test_grouped_shap_is_additive_and_group_coalitions_use_train_references():
    names, groups = feature_groups(['amount', 'amount__missing', 'type', 'type__category_0'])
    assert names == ['amount', 'type'] and groups == [[0, 1], [2, 3]]
    x = np.array([[0, 0, 1, 1], [2, 0, 0, 0], [4, 0, 1, 1]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]])
    index = IncomingNeighborhoods(edges, 3)
    model = HybridGNNAutoencoder(4, 5, 2, 2, dropout=0).eval()
    result, base, score, residual = explain_target(model, x, index, index.mean_features(x), 2, 2, x[:2], groups, names, permutations=2)
    assert result.shape == (2,) and residual < 1e-5
    assert np.isclose(base + result.sum(), score, atol=1e-5)
    game = GroupCoalitions(lambda values: values[:, 0], x[2], x[:2], groups)
    np.testing.assert_allclose(game(np.array([[0, 0], [1, 1]])), [1, 4])


def test_thesis_protocol_uses_shared_enriched_features_and_strict_paysim():
    ieee = runner.build_thesis_config('ieee', 'demo', 30)
    paysim = runner.build_thesis_config('paysim', 'demo', 30)
    assert ieee['split']['test_from'] == 60 * 86400
    assert paysim['split']['test_from'] == 25 * 3600
    assert paysim['data']['exclude_balance_columns']
    assert not paysim['data']['add_balance_consistency_features']
    for config in (ieee, paysim):
        assert config['data']['categorical_encoding'] == 'onehot_frequency'
        assert config['model']['hidden_dimension'] == 128


def test_test_evaluation_forbidden_before_all_training_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'check_frozen', lambda root: {'train_complete': False})
    monkeypatch.setattr(runner, 'prepare_experiment', lambda *a: pytest.fail('Test data must not be prepared'))
    with pytest.raises(ValueError, match='ALL training'):
        runner.evaluate_run(tmp_path)


def test_training_forbidden_after_test(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'check_frozen', lambda root: {'test_evaluated': True})
    with pytest.raises(ValueError, match='Training forbidden'):
        runner.train_run(tmp_path)


def test_frozen_checkpoint_tampering_rejected(tmp_path):
    (tmp_path / 'model.pt').write_bytes(b'original')
    (tmp_path / 'validation.npz').write_bytes(b'original')
    row = {'model': 'hybrid', 'checkpoint_sha256': runner.digest(tmp_path / 'model.pt'),
           'validation_sha256': runner.digest(tmp_path / 'validation.npz')}
    (tmp_path / 'model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed since freeze'):
        runner.validate_checkpoint(tmp_path, row)


def test_shap_random_sample_is_independent_of_case_labels_and_adds_local_cases():
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([.1, .9, .8, .2, .1, .9])
    random, cases = choose_cases(labels, scores, .5, 2, 42)
    other, _ = choose_cases(1 - labels, scores, .5, 2, 42)
    assert random == other
    assert set(cases) == {'TP', 'FP', 'FN', 'TN'}


def test_complete_runner_trains_freezes_evaluates_and_audits_four_models(tmp_path, monkeypatch):
    import pandas as pd
    root = tmp_path / 'thesis_v1_20261012_000000'; root.mkdir()
    config = runner.build_thesis_config('ieee', root.name, 1)
    config['model'].update(hidden_dimension=5, latent_dimension=2, gnn_layers=1, dropout=0)
    config['training'].update(batch_size=4, inference_batch_size=4, ae_pretrain_epochs=1)
    (root / 'ieee_config.json').write_text(json.dumps(config))
    protocol = {'datasets': ['ieee'], 'seeds': [42], 'epochs': 1, 'train_complete': False,
                'test_evaluated': False, 'caveat': 'Toy fixture', 'timing_scope': 'test', 'shap_scope': 'test'}
    (root / 'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr(runner, 'check_frozen', lambda path: runner.read_json(path / 'protocol.json'))
    x = np.random.default_rng(7).normal(size=(24, 3)).astype(np.float32)
    labels = np.tile([0, 1], 12)
    times = np.arange(24)
    summaries = {'splits': {'test': {'rows': 8, 'fraud': 4, 'fraud_rate': .5}}}
    (root / 'ieee' / 'prepared' / 'metrics').mkdir(parents=True)
    (root / 'ieee' / 'prepared' / 'metrics' / 'prepare_summary.json').write_text(json.dumps(summaries))
    def prepare(cfg, *args):
        return SimpleNamespace(config=copy_config(cfg), summary=summaries, features={'train': x[:8], 'validation': x[8:16], 'test': x[16:]},
            graph_features={'train': x[:8], 'validation': x[:16], 'test': x},
            labels={'train': labels[:8], 'validation': labels[8:16], 'test': labels[16:]},
            graphs={key: SimpleNamespace(edge_index=np.array([np.arange(n-1), np.arange(1,n)]))
                    for key, n in [('train',8),('validation',16),('test',24)]},
            partitions=SimpleNamespace(test=pd.DataFrame({'time':times[16:]})), bundle=SimpleNamespace(time_column='time'))
    def copy_config(cfg):
        return json.loads(json.dumps(cfg))
    monkeypatch.setattr(runner, 'prepare_experiment', prepare)
    runner.train_run(root)
    assert runner.read_json(root / 'protocol.json')['train_complete']
    assert len(runner.read_json(root / 'training_results.json')) == 4
    runner.evaluate_run(root)
    assert runner.read_json(root / 'protocol.json')['test_evaluated']
    record = runner.audit_run(root)
    assert record['passed'] and record['rows'] == 4
    rows = runner.read_json(root / 'results.json')
    assert all(row['seconds'] >= 0 and row['inference_seconds'] >= 0 for row in rows)


@pytest.mark.parametrize('tamper', [False, True])
def test_shap_artifact_audit_checks_probability_additivity(tmp_path, monkeypatch, tamper):
    import audit_explainability as audit
    monkeypatch.setattr(audit, 'audit_run', lambda root: None)
    root = tmp_path
    (root / 'protocol.json').write_text(json.dumps({'shap_complete': True, 'datasets': ['ieee'], 'explanations': {'seed':42}}))
    folder = root / 'ieee' / 'explanations'; folder.mkdir(parents=True)
    prep = root / 'ieee' / 'prepared' / 'metrics'; prep.mkdir(parents=True)
    (prep / 'prepare_summary.json').write_text(json.dumps({'splits':{'train':{'rows':2}}}))
    predictions = root / 'ieee' / 'seed_42' / 'hybrid'; predictions.mkdir(parents=True)
    np.savez(predictions / 'predictions.npz', scores=[.7,.8])
    values = np.array([[.2,.3]])
    if tamper:
        values[0,0] = .6
    np.savez(folder / 'shap_values.npz', values=values, base_values=[.2], scores=[.7],
             test_indices=[0], background_train_indices=[0,1], feature_names=['a','b'])
    (folder / 'summary.json').write_text(json.dumps({'random_test_sample':[0], 'top_features':[
        {'feature':'a','mean_absolute_shap':.2},{'feature':'b','mean_absolute_shap':.3}], 'group_members':{'a':[0],'b':[1]}}))
    if tamper:
        with pytest.raises(ValueError, match='additivity'):
            audit.audit_explanations(root)
    else:
        assert audit.audit_explanations(root)['passed']
