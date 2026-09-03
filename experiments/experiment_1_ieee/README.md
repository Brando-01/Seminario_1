# Experimento 1: IEEE-CIS

Objetivo: comparar baseline, Autoencoder, GraphSAGE y modelo híbrido sobre un grafo temporal de transacciones construido mediante entidades aproximadas compartidas.

Preparación rápida:

```powershell
python run_experiment.py --config configs/experiment_1_ieee.yaml --max-rows 10000
```

La evaluación final utilizará únicamente datos etiquetados de `train_transaction.csv`, divididos cronológicamente.

