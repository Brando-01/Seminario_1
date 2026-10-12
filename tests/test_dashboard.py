"""Small regressions for the local results dashboard."""

import json
import math

import dashboard_app
import dashboard_data
import numpy as np
from dashboard_app import _finite_json


def test_graph_cache_prefers_completed_thesis_and_ignores_partial_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_data, 'OUTPUTS', tmp_path)
    historical = tmp_path / 'validation_v2_20261001_000000'
    historical.mkdir()
    (historical / 'results.json').write_text('[]')
    for dataset in dashboard_data.DATASETS:
        (historical / f'{dataset}_config.json').write_text('{}')
    directory = tmp_path / 'thesis_v1_20261012_154354'
    directory.mkdir()
    protocol = {'test_evaluated': False, 'datasets': ['ieee', 'paysim'], 'seeds': [52]}
    (directory / 'protocol.json').write_text(json.dumps(protocol))
    (directory / 'results.json').write_text('[]')
    assert dashboard_data.complete_validation_run() == historical
    protocol['test_evaluated'] = True
    (directory / 'protocol.json').write_text(json.dumps(protocol))
    for dataset in dashboard_data.DATASETS:
        for model in ('baseline', 'autoencoder', 'gnn', 'hybrid'):
            folder = directory / dataset / 'seed_52' / model
            folder.mkdir(parents=True)
            np.savez(folder / 'predictions.npz', labels=[0, 1], scores=[.1, .9])
    assert dashboard_data.complete_validation_run() == directory


def test_dashboard_thesis_has_four_curves_separate_costs_and_shap(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, 'OUTPUTS', tmp_path)
    directory = tmp_path / 'thesis_v1_20261012_154354'
    directory.mkdir()
    (directory / 'protocol.json').write_text(json.dumps({'datasets': ['ieee'], 'test_evaluated': True,
        'shap_complete': True, 'caveat': 'Frozen later period', 'feature_set': 'enriched'}))
    rows = [{'dataset': 'ieee', 'seed': 42, 'model': name, 'inference_seconds': .1, 'seconds': 1,
             'metrics': {'auc_pr': .5}} for name in ('baseline', 'autoencoder', 'gnn', 'hybrid')]
    (directory / 'results.json').write_text(json.dumps(rows))
    for row in rows:
        folder = directory / 'ieee' / 'seed_42' / row['model']; folder.mkdir(parents=True)
        np.savez(folder / 'predictions.npz', labels=[0, 1], scores=[.1, .9])
    explanations = directory / 'ieee' / 'explanations'; explanations.mkdir()
    (explanations / 'summary.json').write_text(json.dumps({'scope': 'fixed historical graph'}))
    runs = dashboard_app.list_runs()
    assert runs[0]['kind'] == 'thesis' and runs[0]['complete']
    detail = dashboard_app.run_detail(directory.name, 'ieee')
    assert len(detail['curves']) == 4 and detail['caveat'] == 'Frozen later period'
    assert detail['shap']['scope'] == 'fixed historical graph'


def test_nonfinite_historical_thresholds_are_safe_for_the_api():
    payload = {"rows": [{"threshold": math.inf, "metric": 0.7}], "nested": [-math.inf, math.nan]}

    safe = _finite_json(payload)

    assert safe == {"rows": [{"threshold": None, "metric": 0.7}], "nested": [None, None]}
    assert payload["rows"][0]["threshold"] == math.inf
    json.dumps(safe, allow_nan=False)


def test_dashboard_lists_only_feature_searches_with_completed_test(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "OUTPUTS", tmp_path)
    directory = tmp_path / "feature_search_20261001_220532"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps({"dataset": "ieee", "seeds": [42, 52], "test_evaluated": False}))
    (directory / "validation_only.json").write_text(json.dumps([{"seed": 42, "metrics": {}}]))

    assert dashboard_app.list_runs() == []

    (directory / "protocol.json").write_text(json.dumps({"dataset": "ieee", "seeds": [42], "test_evaluated": True}))
    runs = dashboard_app.list_runs()
    assert len(runs) == 1
    assert runs[0]["kind"] == "feature_selection"
    assert runs[0]["ready_datasets"] == ["ieee"]
    assert runs[0]["complete"] is True


def test_dashboard_reads_new_exploratory_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "OUTPUTS", tmp_path)
    directory = tmp_path / "feature_search_20261001_220532"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps({"dataset": "ieee", "seeds": [42], "test_evaluated": True}))
    (directory / "validation_only.json").write_text(json.dumps([{"seed": 42, "validation_ap": 0.4}]))
    (directory / "exploratory_test.json").write_text(json.dumps([{"seed": 42, "metrics": {"auc_pr": 0.5}}]))

    assert dashboard_app._feature_results_file(directory).name == "exploratory_test.json"
    assert dashboard_app.list_runs()[0]["complete"] is True


def test_minibatch_validation_is_not_shown_as_test_and_partial_test_is_labeled(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_app, "OUTPUTS", tmp_path)
    directory = tmp_path / "minibatch_search_20261012_011449"
    directory.mkdir()
    (directory / "protocol.json").write_text(json.dumps({"datasets": ["ieee"], "seeds": [42, 52], "feature_set": "enriched"}))
    assert dashboard_app.list_runs() == []
    rows = [{"dataset": "ieee", "seed": 42, "model": model, "metrics": {"auc_pr": 0.5}} for model in ("hybrid", "baseline")]
    (directory / "results.json").write_text(json.dumps(rows))
    assert dashboard_app.list_runs()[0]["complete"] is False
    detail = dashboard_app.run_detail(directory.name, "ieee")
    assert detail["kind"] == "minibatch"
    assert detail["feature_set"] == "enriched"
    assert len(detail["rows"]) == 2
    rows += [{"dataset": "ieee", "seed": 52, "model": model, "metrics": {"auc_pr": 0.5}} for model in ("hybrid", "baseline")]
    (directory / "results.json").write_text(json.dumps(rows))
    assert dashboard_app.list_runs()[0]["complete"] is True
