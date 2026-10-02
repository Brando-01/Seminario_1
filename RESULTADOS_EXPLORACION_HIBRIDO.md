# Exploracion del entrenamiento del hibrido GNN-AE

Corrida: `outputs/hybrid_search_20261001_190553/`. Se usaron las mismas ventanas temporales y semillas (42, 52, 62) de la validacion v2. Dos variantes del mismo hibrido se compararon con **AP de validacion**: 60 epocas sin preentrenamiento y 60 epocas con 10 epocas previas de autoencoder sobre transacciones normales. Solo la variante elegida en cada dataset se evaluo en test. Las cifras de la tabla son promedios en porcentaje, sin `+/-`.

| Dataset | Hibrido anterior AP (%) | Hibrido exploratorio AP (%) | Baseline AP (%) | Hibrido exploratorio F1 (%) |
|---|---:|---:|---:|---:|
| IEEE-CIS | 47.02 | 49.81 | 60.08 | 47.69 |
| PaySim | 0.32 | 4.61 | 7.85 | 3.12 |

La variante con preentrenamiento fue elegida en ambos datasets. Sin embargo, en IEEE-CIS la diferencia de AP de validacion frente a solo entrenar mas tiempo fue apenas **43.05 % vs. 43.01 %**. Por eso no se puede atribuir la mejora en test especificamente al preentrenamiento; gran parte podria deberse al mayor presupuesto de epocas. En PaySim la AP de validacion fue **0.39 % vs. 0.11 %**, pero los resultados de test variaron mucho entre semillas.

| Dataset | Semilla | AP exploratoria (%) | Recall operativo (%) | FP por 1000 legitimas |
|---|---:|---:|---:|---:|
| IEEE-CIS | 42 | 50.36 | 43.38 | 10.87 |
| IEEE-CIS | 52 | 49.37 | 40.70 | 12.04 |
| IEEE-CIS | 62 | 49.69 | 41.88 | 12.81 |
| PaySim | 42 | 8.62 | 50.00 | 24.42 |
| PaySim | 52 | 0.50 | 0.00 | 0.00 |
| PaySim | 62 | 4.71 | 25.00 | 2.38 |

**Interpretacion:** el hibrido mejoro su AP media en este experimento, pero **no supero al baseline** en ninguno de los dos datasets. IEEE-CIS excedio en test el presupuesto operativo de 10 FP/1000 en las tres semillas. En PaySim una semilla no detecto ningun fraude al umbral operativo y otra excedio mucho el presupuesto; su test historico tiene apenas 36 fraudes. No recomendar este modelo para operacion con estos resultados.

Este es un analisis **exploratorio**, no una prueba confirmatoria: el periodo de test v2 ya habia sido consultado antes de disenar esta mejora. Para una afirmacion final de tesis hace falta fijar la configuracion y evaluarla en un nuevo periodo temporal no utilizado durante el desarrollo. Tampoco puede inferirse superioridad estadistica de tres semillas sobre el mismo test.

Para revisar las metricas en terminal sin volver a entrenar:

```powershell
python show_results.py --hybrid-search latest --details
python show_results.py --validation-run validation_v2_20260906_212206
```

El primer comando muestra la nueva exploracion. El segundo muestra la comparacion anterior de los cuatro modelos. Para repetir la exploracion completa en ambos datasets (tarda varios minutos en CPU):

```powershell
python explore_hybrid.py --experiment all --epochs 60 --seeds 42 52 62
```
