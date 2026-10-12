# Proyecto ejecutable con los seis objetivos implementados

Esta ruta añade la comparación común y SHAP que faltaban. Completar la
implementación y verificar una corrida no equivale a concluir una tesis,
probar superioridad estadística o desplegar una solución bancaria.
La evidencia de cumplimiento debe comprobarse en la corrida `thesis_v1_...`,
no deducirse solamente de que existan scripts.

## Correspondencia con la ficha

| Objetivo | Implementación | Evidencia de una ejecución completa |
|---|---|---|
| 1 Caracterizar y preprocesar ambos datasets | loaders, temporal_split, TabularPreprocessor | prepare_summary.json, columnas retenidas, faltantes, tasas y períodos |
| 2 Representación relacional adaptada | grafo causal de transacciones | nodos y aristas, claves de entidades y pruebas de temporalidad |
| 3 Cuatro modelos comparables | thesis_experiment, GNN por lotes y AE preentrenado | cuatro modelos por dataset y semilla |
| 4 AP, Precision, Recall, F1 y falsas alertas | métricas y umbrales congelados | results.json, REPORT.md, auditoría y matrices de confusión |
| 5 Costo computacional | medición de optimización, validación e inferencia | tiempos separados para los cuatro modelos |
| 6 SHAP del híbrido | src/explainability | shap_values.npz, summary.json, ranking y waterfall locales |

## Un proyecto y dos escenarios

Las configuraciones y carga cambian por dataset. La arquitectura, preparación,
métricas, semillas y reporte son comunes. Cada nodo es una **transacción**.
IEEE-CIS usa proxies de tarjeta/correo/dispositivo; PaySim relaciona
transacciones que comparten cuentas. No se entrenan modelos de IEEE para
predecir PaySim y no se comparan AP absolutas entre poblaciones.

## Protocolo temporal congelado

| Dataset | Train | Validación | Test posterior |
|---|---|---|---|
| IEEE-CIS | días [30,50) | [50,60) | [60,75) |
| PaySim | pasos 1–16 | 17–24 | 25–72 |

Los cortes se fijan antes de preparar los datos. No corresponden exactamente
al 70/15/15 de la ficha: documenta esta adaptación, que respeta timestamps
completos. Son ventanas acotadas y no todos los CSV.
El período anterior de test puede formar ahora parte de validación; no se
presenta nuevamente como un test no consultado.

Los nuevos períodos son posteriores al último piloto registrado. No es
posible certificar accesos históricos fuera del proyecto; el manifiesto lo
declara. No reajustes decisiones al mirar sus métricas. Tres semillas sobre
un mismo período no constituyen tres pruebas temporales independientes.

El preprocesamiento se ajusta solo con train: medianas, escala, categorías,
frecuencias y faltantes. Todos los modelos de un dataset reciben las mismas
variables. PaySim principal excluye **todos los saldos**, `isFlaggedFraud`
y las inconsistencias de saldo; conserva historial causal de cuentas.
Esto difiere del piloto anterior que conservaba saldos iniciales.

## Entrenamiento y costo

Parámetros por defecto: hasta 30 épocas, ancho 128, latente 16, dos capas
GraphSAGE, dropout 0,15, paciencia 10, peso positivo máximo 20, pérdida de
reconstrucción 0,01 y preentrenamiento AE de hasta 10 épocas en el híbrido.
El AE independiente usa la misma compuerta de atributos y hasta 30 épocas.
El GNN independiente combina embedding relacional y proyección tabular,
como en su implementación anterior; no se describe como un GNN puramente
relacional. El híbrido añade el AE y su error y normaliza la fusión.

Las redes supervisadas conservan el checkpoint de mayor AP de validación.
El AE usa MSE de validación legítima. Los umbrales F1 y operativo se fijan
en validación y se guardan antes de ejecutar test.

Se miden por separado optimización, validación e inferencia. En el híbrido
el entrenamiento incluye optimización de preentrenamiento AE, pero no su
validación. La inferencia de grafos incluye la preparación de medias y
subgrafos necesaria para producir scores. Se excluyen carga general de CSV,
preprocesamiento, construcción inicial de grafos, escritura y SHAP.
El baseline usa class-balanced weights; su selección interna de scikit-learn
ocurre dentro de train. Estos tiempos describen el equipo local y no una
latencia por transacción de un sistema en producción.

## Qué explica SHAP

Se ejecuta `shap.PermutationExplainer` sobre el **híbrido real**, no un
surrogate ni pesos de atención. Se perturban grupos de características de
la transacción objetivo: un campo categórico conserva juntos frecuencia,
one-hot y indicador de faltante; lo mismo ocurre con numéricos y sus
indicadores. Se sustituyen usando filas de fondo provenientes solo de train.

Las aristas, vecinos y sus características históricas se mantienen fijos.
Por ello la atribución es condicional al contexto y no explica la selección
de aristas ni cuantifica causalidad. El adapter verifica que su score coincide
con la predicción guardada y que base + suma de SHAP reproduce el score.
Las unidades son probabilidad, no valores monetarios originales.

Por defecto se usan ocho referencias, tres permutaciones y 16 transacciones
de test seleccionadas uniformemente con semilla fija. El gráfico global es
un promedio **de esa muestra**, no de toda la población. Se añaden casos
ilustrativos TP, FP, FN y TN cuando existen; estos extras no contaminan la
agregación global. La aproximación depende del fondo y número de permutaciones.
El agrupamiento reúne codificaciones e indicadores del mismo nombre de columna;
no impone todas las dependencias de las variables ingenierizadas. Por ejemplo,
`amount` y `amount_log1p` son grupos separados. Algunas perturbaciones pueden
producir combinaciones que no sean transacciones físicamente plausibles;
interpreta las atribuciones en el espacio transformado del modelo, no como
simulaciones económicas coherentes ni recomendaciones de intervención.

Referencia del algoritmo:
[documentación oficial de PermutationExplainer](https://shap.readthedocs.io/en/stable/generated/shap.PermutationExplainer.html).

## Comandos

Con `.venv` activo y ubicado en la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python thesis_experiment.py --stage all
```

Para reproducir las versiones verificadas localmente usa Python 3.12 y
`python -m pip install -r requirements-tested.txt`. En otra plataforma puede
requerirse una distribución compatible de PyTorch; esto no garantiza
resultados idénticos bit a bit entre equipos.

Por etapas, para decidir cuándo producir resultados de test:

```powershell
python thesis_experiment.py --stage train
python thesis_experiment.py --stage evaluate --run latest
python thesis_experiment.py --stage audit --run latest
python thesis_experiment.py --stage explain --run latest
python audit_explainability.py --run latest
python characterize_thesis.py --run latest
python show_results.py --thesis-run latest --details
python dashboard_app.py --no-prepare
```

Usa el nombre concreto de corrida en vez de `latest` para una exposición.
En PowerShell, `.\project.ps1 thesis-complete` encadena una nueva corrida,
la auditoría numérica de SHAP y la caracterización descriptiva. Ejecútalo
sin lanzar simultáneamente otra corrida: sus etapas posteriores resuelven
`latest`. No es necesario volver a entrenar para mostrar resultados guardados.
Si el entrenamiento se interrumpe, `--stage train --run <corrida>` conserva
los modelos ya congelados. Si cambia el código, configuración o CSV, los
hashes bloquean la continuación: crea una nueva corrida, no edites el
manifiesto para eludir el control. No se permite entrenar después de evaluar
test. Los checkpoints pickle se cargan solo si fueron generados localmente
y su hash coincide; no uses checkpoints descargados de terceros.

## Orden actualizado para exponer el código

1. `thesis_experiment.py`: `build_thesis_config`, `create_run`, `train_run`.
2. `run_experiment.py`: `prepare_experiment`; adaptadores en `src/data`.
3. `src/graph/temporal_graph.py`: nodos transacción y aristas pasado → presente.
4. `src/models`: baseline, AE, GraphSAGE y concatenación híbrida.
5. `src/training`: preentrenamiento, pérdidas y mini-lotes exactos.
6. `thesis_experiment.py`: `evaluate_run` y `audit_run`; umbrales congelados.
7. `src/explainability/conditional_shap.py`: adapter, grupos y contexto fijo.
8. `src/explainability/report.py`: muestreo, explicación local y agregación.
9. `show_results.py --thesis-run ...`: resultados guardados, no reentrenamiento.

La carpeta de cada corrida conserva parámetros, hashes, checkpoints,
scores de validación/test, historiales, tiempos, matrices de confusión,
reporte y explicaciones. Los datos crudos no se modifican y `.venv`, CSV
y outputs continúan excluidos de Git. Los reportes y gráficos actuales se
publican como copias exactas en `reports/thesis_v1_20261012_154354/`.
Los resultados anteriores se retiraron a un respaldo local fuera del repositorio.
