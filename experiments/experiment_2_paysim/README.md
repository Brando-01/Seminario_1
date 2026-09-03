# Experimento 2: PaySim

Objetivo: repetir la comparación de modelos bajo desbalance extremo y relaciones explícitas entre cuentas de origen y destino.

Preparación rápida:

```powershell
python run_experiment.py --config configs/experiment_2_paysim.yaml --max-rows 10000
```

El experimento principal excluye saldos e `isFlaggedFraud` de las características predictivas.

