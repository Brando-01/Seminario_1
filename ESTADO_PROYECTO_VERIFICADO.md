# Estado del proyecto ejecutable — 12 de octubre de 2026

Corrida verificada: `thesis_v1_20261012_154354`.
Los seis objetivos específicos de la ficha tienen implementación y evidencia
de ejecución en las ventanas indicadas. Esto no equivale a terminar la tesis,
establecer superioridad estadística ni validar una solución de producción.

## Evidencias por objetivo

| Objetivo | Evidencia dentro de la carpeta de corrida |
|---|---|
| Caracterización y preprocesamiento | `<dataset>/eda/characterization.json`, `REPORT_EDA.md`, `prepared/metrics/prepare_summary.json` y preprocesador ajustado solo con train |
| Representación relacional adaptada | Grafos causales de transacciones, conteos de nodos/aristas y pruebas de temporalidad |
| Baseline, AE, GNN e híbrido | 24 checkpoints: cuatro modelos × dos datasets × tres semillas |
| Comparación de métricas | `results.json`, `REPORT.md`, scores y matrices de confusión por modelo/semilla |
| Costo computacional | Optimización, validación e inferencia de test registradas por separado |
| Explicabilidad SHAP | 20 casos por dataset, valores guardados, ranking de muestra y gráficos locales; `audit_shap.json` |

`protocol.json`: entrenamiento, test y SHAP completos.
`audit.json`: 24 resultados, checkpoints/umbrales congelados, mismas particiones
por dataset y métricas reproducidas. `audit_shap.json`: aditividad y coincidencia
con las predicciones verificadas. La suite `pytest tests` pasó 61 pruebas;
tres advertencias de deprecación futura de SHAP/Matplotlib no son fallos.

## Datos utilizados

| Dataset | Train | Validación | Test | Fraudes de test |
|---|---:|---:|---:|---:|
| IEEE-CIS | 61.720 | 28.118 | 47.046 | 2.045 |
| PaySim | 348.191 | 226.064 | 456.308 | 619 |

IEEE-CIS usa días relativos [30,50), [50,60), [60,75).
PaySim usa pasos 1–16, 17–24, 25–72. No se utiliza la totalidad de los CSV.
En PaySim principal se excluyen todos los saldos, `isFlaggedFraud` y las
inconsistencias de saldo. Los dos escenarios comparten arquitectura y criterios,
pero no se mezclan sus registros ni se evalúa transferencia entre datasets.

## Métricas de test

Porcentajes medios de las semillas 42, 52 y 62. Los umbrales de clasificación
se eligieron en validación. F1 es la media de los F1 de cada ejecución, no
la media armónica de Precision y Recall promedios. AP no es accuracy.

| Dataset | Modelo | AP % | Precision % | Recall % | F1 % |
|---|---|---:|---:|---:|---:|
| IEEE-CIS | Baseline | 53,98 | 68,47 | 40,54 | 50,88 |
| IEEE-CIS | Autoencoder | 33,23 | 39,48 | 33,66 | 35,94 |
| IEEE-CIS | GraphSAGE | 44,46 | 64,37 | 35,89 | 46,07 |
| IEEE-CIS | Híbrido | 44,43 | 61,92 | 35,81 | 45,19 |
| PaySim | Baseline | 5,77 | 68,33 | 14,54 | 5,69 |
| PaySim | Autoencoder | 0,21 | 0,20 | 22,83 | 0,40 |
| PaySim | GraphSAGE | 27,22 | 72,20 | 18,85 | 29,48 |
| PaySim | Híbrido | 29,41 | 70,26 | 20,62 | 31,75 |

En IEEE-CIS el baseline supera al híbrido por 9,56 puntos de AP; el híbrido
tampoco supera al GNN individual en AP media (-0,03 puntos). En PaySim el
híbrido alcanza la mayor AP media observada, +23,65 puntos frente al baseline
y +2,20 frente al GNN. Estas diferencias no prueban significancia estadística
ni superioridad universal: las semillas comparten un mismo período.

En el punto operativo de PaySim, el híbrido alcanza Recall 66,56 % con
46,30 FP por 1.000 legítimas. **El presupuesto de 10 fijado en validación
no se mantuvo en test.** No se mezcla este Recall con las métricas del umbral F1.

## Alcance de SHAP

Se explica el híbrido real de la semilla 42, con ocho referencias de train,
tres permutaciones y 16 transacciones aleatorias de test por dataset. Cuatro
casos locales adicionales ilustran TP, FP, FN y TN; no se agregan al ranking
global. El ranking representa esa muestra pequeña, no toda la población.

Se explican grupos de atributos transformados del nodo objetivo con contexto
histórico y aristas fijos. No son explicaciones de aristas, causalidad ni
simulaciones de transacciones económicamente coherentes. Las dependencias
entre variables ingenierizadas no se preservan en todos los grupos.
Los errores máximos de aditividad auditados fueron 1,19e-7 en IEEE y 2,98e-8
en PaySim.

## Cómo presentar y reproducir

Con `.venv` activo en la raíz del proyecto, sin volver a entrenar:

```powershell
python show_results.py --thesis-run thesis_v1_20261012_154354 --details
python audit_explainability.py --run thesis_v1_20261012_154354
python dashboard_app.py --no-prepare --port 8766
```

El panel actualizado lee las métricas y SHAP de esta corrida. La caché de
animaciones se reconstruyó desde sus predicciones; su fuente se identifica
en pantalla. Al seleccionar otra corrida, las tablas cambian pero la caché
visual mantiene su origen. `--rebuild-cache` permite regenerarla.

La guía actual para exponer es `docs/Guia_exposicion_proyecto_completo.docx`:
15 páginas revisadas visualmente con Word, con recorrido de archivos,
explicación oral, cantidades, resultados, costos y SHAP. Reemplaza la guía
anterior que describía el avance incompleto.

Para una nueva ejecución de todo el protocolo:

```powershell
.\project.ps1 thesis-complete
```

Este comando entrena de nuevo y puede tardar. No lo ejecutes simultáneamente
con otra corrida porque sus etapas finales utilizan `latest`. Python 3.12 y
`requirements-tested.txt` documentan las versiones locales; no garantizan
identidad numérica entre equipos. Los CSV se copian aparte al cambiar de PC.

Los datos y la corrida actual se conservan íntegros. Se retiraron del proyecto
13 carpetas de corridas anteriores y cinco reportes/guías obsoletos, con
respaldo recuperable en `../_respaldo_resultados_anteriores_20261012/`.
En `reports/thesis_v1_20261012_154354/` se incluyen copias exactas de los
reportes numéricos, protocolo, configuraciones, auditorías y gráficos actuales
para publicar junto con el código y la guía Word. `.venv`, datos crudos y
artefactos completos de `outputs` continúan excluidos de Git.
La ficha y el artículo todavía deben alinearse con los cortes reales, la
codificación enriquecida, el escenario estricto y el alcance de SHAP;
no fueron modificados.
