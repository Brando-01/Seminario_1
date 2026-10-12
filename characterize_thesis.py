"""Descriptive EDA of the frozen windows; never selects models or thresholds."""
import argparse
import numpy as np
from run_experiment import PROJECT_ROOT, split_summary
from src.data import load_dataset, temporal_split
from thesis_experiment import resolve_run, read_json, save_json, check_frozen


def characterize_run(root):
    protocol = check_frozen(root)
    for dataset in protocol['datasets']:
        config = read_json(root / f'{dataset}_config.json')
        bundle = load_dataset(config, PROJECT_ROOT, None, 'window')
        cfg = config['split']
        parts = temporal_split(bundle.frame, bundle.time_column, cfg['train'], cfg['validation'], cfg['train_before'], cfg['test_from'])
        train = parts.train
        amount = 'TransactionAmt' if dataset == 'ieee' else 'amount'
        missing = train.isna().mean().sort_values(ascending=False)
        period = (bundle.frame[bundle.time_column] // (86400 if dataset == 'ieee' else 3600)).astype(int)
        by_period = bundle.frame.groupby(period)[bundle.target_column].agg(['size','sum'])
        report = {
            'scope': 'Descriptive characterization only; no model/threshold selection from test.',
            'shape': {'rows':len(bundle.frame),'columns':len(bundle.frame.columns)},
            'splits': {key: split_summary(getattr(parts,key), bundle.target_column, bundle.time_column)
                       for key in ('train','validation','test')},
            'train_missing_fraction': {str(k):float(v) for k,v in missing.items()},
            'train_amount_distribution': {str(k):float(v) for k,v in train[amount].describe(percentiles=[.25,.5,.75,.95,.99]).items() if np.isfinite(v)},
            'column_types': {str(k):str(v) for k,v in train.dtypes.items()},
            'fraud_by_period': [{'period':int(time),'rows':int(row['size']),'frauds':int(row['sum']),
                                 'fraud_rate':float(row['sum']/row['size'])} for time,row in by_period.iterrows()],
            'source':bundle.metadata,
        }
        folder = root / dataset / 'eda'; folder.mkdir(parents=True,exist_ok=True)
        save_json(folder / 'characterization.json',report)
        lines = [f'# Caracterización {dataset.upper()}','',report['scope'],'',
                 f'Filas en ventana: {len(bundle.frame):,}. Columnas antes de transformación: {len(bundle.frame.columns)}.',
                 '', '## Particiones', '', '| Partición | Operaciones | Fraudes | Tasa % | Tiempo mínimo | Tiempo máximo |',
                 '|---|---:|---:|---:|---:|---:|']
        for key,row in report['splits'].items():
            lines.append(f'| {key} | {row["rows"]} | {row["fraud"]} | {100*row["fraud_rate"]:.4f} | {row["time_min"]} | {row["time_max"]} |')
        lines += ['', '## Faltantes en train','', '| Columna | Faltantes % |','|---|---:|']
        lines += [f'| {key} | {100*value:.2f} |' for key,value in list(report['train_missing_fraction'].items())[:20]]
        lines += ['', 'Las estadísticas de test son descriptivas posteriores a la congelación; no modifican el diseño. El preprocesador se ajusta exclusivamente con train.']
        (folder / 'REPORT_EDA.md').write_text('\n'.join(lines),encoding='utf-8')
        print(f'EDA {dataset}: {folder}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='latest')
    characterize_run(resolve_run(parser.parse_args().run))
