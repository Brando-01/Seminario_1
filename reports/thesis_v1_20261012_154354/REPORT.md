# Experimento completo de tesis

Frozen prospective local protocol on later time windows; prior access to these records outside the recorded pilots cannot be certified. A single period and three seeds do not establish statistical superiority.

# Resultados de validacion temporal

AP = average precision (precision promedio), no accuracy ni area PR trapezoidal.
Cada celda muestra el promedio de las semillas como porcentaje, sin +/-.
La variacion entre semillas se puede consultar por separado; comparten el mismo test.
F1 es la media de los F1 individuales, no el F1 calculado a partir de Precision y Recall medios.

## IEEE

Semillas: 42, 52, 62.
Prueba: 47,046 operaciones, 2,045 fraudes (4.3468% de prevalencia).
Umbral de Precision, Recall y F1: elegido solo en validacion para maximizar F1.

| Modelo | AP (%) | ROC-AUC (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|
| Baseline tabular | 53.98 | 88.19 | 68.47 | 40.54 | 50.88 |
| Autoencoder | 33.23 | 81.68 | 39.48 | 33.66 | 35.94 |
| GraphSAGE | 44.46 | 76.68 | 64.37 | 35.89 | 46.07 |
| Hibrido GNN-AE | 44.43 | 78.91 | 61.92 | 35.81 | 45.19 |

Mayor AP media observada: Baseline tabular. Hibrido - baseline: -9.56 puntos porcentuales; hibrido - GNN: -0.03 puntos porcentuales.

Punto operativo: umbral fijado en validacion con presupuesto de falsas alarmas.
| Modelo | Recall operativo (%) | FP por 1000 legitimas |
|---|---:|---:|
| Baseline tabular | 41.81 | 9.94 |
| Autoencoder | 21.63 | 7.58 |
| GraphSAGE | 36.58 | 10.09 |
| Hibrido GNN-AE | 35.78 | 10.36 |

Tiempos locales medios por semilla; inferencia sobre toda la particion de test.
Entrenamiento incluye preentrenamiento AE; excluye preparacion, busqueda y validacion.
| Modelo | Entrenamiento (s) | Inferencia test (s) |
|---|---:|---:|
| Baseline tabular | 26.86 | 0.93 |
| Autoencoder | 36.94 | 0.38 |
| GraphSAGE | 80.45 | 0.81 |
| Hibrido GNN-AE | 206.32 | 1.41 |

## PAYSIM

Semillas: 42, 52, 62.
Prueba: 456,308 operaciones, 619 fraudes (0.1357% de prevalencia).
Umbral de Precision, Recall y F1: elegido solo en validacion para maximizar F1.

| Modelo | AP (%) | ROC-AUC (%) | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|
| Baseline tabular | 5.77 | 94.52 | 68.33 | 14.54 | 5.69 |
| Autoencoder | 0.21 | 65.01 | 0.20 | 22.83 | 0.40 |
| GraphSAGE | 27.22 | 94.70 | 72.20 | 18.85 | 29.48 |
| Hibrido GNN-AE | 29.41 | 94.89 | 70.26 | 20.62 | 31.75 |

Mayor AP media observada: Hibrido GNN-AE. Hibrido - baseline: +23.65 puntos porcentuales; hibrido - GNN: +2.20 puntos porcentuales.

Punto operativo: umbral fijado en validacion con presupuesto de falsas alarmas.
| Modelo | Recall operativo (%) | FP por 1000 legitimas |
|---|---:|---:|
| Baseline tabular | 1.83 | 0.44 |
| Autoencoder | 15.72 | 100.43 |
| GraphSAGE | 68.93 | 32.76 |
| Hibrido GNN-AE | 66.56 | 46.30 |

Tiempos locales medios por semilla; inferencia sobre toda la particion de test.
Entrenamiento incluye preentrenamiento AE; excluye preparacion, busqueda y validacion.
| Modelo | Entrenamiento (s) | Inferencia test (s) |
|---|---:|---:|
| Baseline tabular | 0.86 | 0.25 |
| Autoencoder | 27.89 | 0.17 |
| GraphSAGE | 175.25 | 2.51 |
| Hibrido GNN-AE | 274.78 | 3.63 |

No se demuestra superioridad estadistica con semillas sobre un solo periodo de test.
Comparar modelos dentro de cada dataset; la AP entre datasets cambia con la prevalencia.
Un cambio de presentacion a porcentajes no modifica ningun resultado.

## Alcance del costo

Training optimization (AE pretraining included for hybrid), validation separately; test score inference including graph-context preparation; excludes CSV loading, general preprocessing, checkpoint I/O and SHAP.

## Explicabilidad

Approximate grouped SHAP of transformed target features through the actual hybrid; past features and graph fixed; not graph-edge or causal attribution.

Los resultados no garantizan que el híbrido sea superior. No volver a seleccionar con test.
