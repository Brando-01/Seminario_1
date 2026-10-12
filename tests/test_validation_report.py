from validation_report import render_hybrid_search_report, render_validation_report


def _row(seed, model, ap, operating_recall):
    return {
        "dataset": "ieee",
        "seed": seed,
        "model": model,
        "metrics": {
            "auc_pr": ap,
            "auc_roc": 0.9,
            "precision": 0.5,
            "recall": 0.4,
            "f1": 0.4444,
        },
        "operating_metrics": {
            "recall": operating_recall,
            "false_positives_per_1000": 10.0,
        },
    }


def test_validation_report_converts_to_percent_and_percentage_points():
    rows = [
        _row(42, "baseline", 0.6, 0.3),
        _row(52, "baseline", 0.8, 0.5),
        _row(42, "gnn", 0.5, 0.2),
        _row(52, "gnn", 0.5, 0.2),
        _row(42, "hybrid", 0.55, 0.4),
        _row(52, "hybrid", 0.55, 0.4),
    ]
    summaries = {"ieee": {"splits": {"test": {"rows": 100, "fraud": 10, "fraud_rate": 0.1}}}}

    report = render_validation_report(rows, summaries)

    assert "AP = average precision" in report
    assert "| Baseline tabular | 70.00 |" in report
    assert "| Hibrido GNN-AE | 55.00 |" in report
    assert "+/-" not in report.split("| Baseline tabular |", 1)[1].split("\n", 1)[0]
    assert "Hibrido - baseline: -15.00 puntos porcentuales" in report
    assert "10 fraudes (10.0000% de prevalencia)" in report
    assert "| Baseline tabular | 40.00 | 10.00 |" in report

    detailed = render_validation_report(rows, summaries, show_variability=True)
    assert "| Baseline tabular | 70.00 +/- 14.14" in detailed
    assert "| Baseline tabular | 40.00 +/- 14.14 | 10.00 +/- 0.00 |" in detailed


def test_validation_report_handles_single_seed_and_missing_operating_metrics():
    row = _row(42, "hybrid", 0.47, 0.4)
    del row["operating_metrics"]

    report = render_validation_report([row], show_variability=True)

    assert "47.00 (DE N/D; n=1)" in report
    assert "Punto operativo" not in report


def test_hybrid_search_report_uses_plain_percentages_by_default():
    validation = [
        {"candidate": "longer_training", "validation_ap": 0.40},
        {"candidate": "ae_pretrained", "validation_ap": 0.43},
    ]
    test = [{
        "candidate": "ae_pretrained", "seed": 42,
        "metrics": {"auc_pr": 0.49, "auc_roc": 0.84, "precision": 0.50, "recall": 0.45, "f1": 0.47},
        "operating_metrics": {"recall": 0.42, "false_positives_per_1000": 12.0},
    }]

    report = render_hybrid_search_report({"ieee": (validation, test)})

    assert "| ae_pretrained (elegida) | 43.00 |" in report
    assert "| 49.00 | 84.00 | 50.00 | 45.00 | 47.00 | 42.00 | 12.00 |" in report
    assert "test historico ya habia sido inspeccionado" in report


def test_minibatch_report_compares_two_models_and_separates_times():
    rows = [_row(42, "baseline", 0.60, 0.3), _row(42, "hybrid", 0.55, 0.4)]
    for row in rows:
        row.update(seconds=12.0, inference_seconds=0.8)
    report = render_validation_report(rows)
    assert "Mayor AP media observada: Baseline tabular" in report
    assert "Hibrido - baseline: -5.00 puntos porcentuales." in report
    assert "hibrido - GNN" not in report
    assert "| Hibrido GNN-AE | 12.00 | 0.80 |" in report
