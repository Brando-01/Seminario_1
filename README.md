# Detección de fraude con GNN y Autoencoder

Un proyecto común con dos escenarios: IEEE-CIS (relaciones inferidas entre
transacciones) y PaySim (relaciones por cuentas). En ambos se comparan baseline
tabular, Autoencoder, GraphSAGE e híbrido GNN-AE. Los datos no se mezclan.

## Resultados actuales

La única corrida activa de referencia es `thesis_v1_20261012_154354`.
Se ejecutaron 24 modelos: cuatro enfoques × dos datasets × tres semillas.
Las auditorías de resultados y SHAP pasaron; la suite aprobó 61 pruebas.

- [Reporte de métricas y costos](reports/thesis_v1_20261012_154354/REPORT.md).
- [Estado, particiones y limitaciones](ESTADO_PROYECTO_VERIFICADO.md).
- [Evidencia publicada y alcance del respaldo](reports/README.md).
- [Guía Word de exposición](docs/Guia_exposicion_proyecto_completo.docx).
- [Metodología y recorrido del código](PROYECTO_COMPLETO.md).
- SHAP del híbrido: [IEEE-CIS](reports/thesis_v1_20261012_154354/ieee/explanations/REPORT_SHAP.md)
  y [PaySim](reports/thesis_v1_20261012_154354/paysim/explanations/REPORT_SHAP.md).

| Dataset | Mejor AP media observada | AP del híbrido | F1 del híbrido |
|---|---|---:|---:|
| IEEE-CIS | Baseline: 53,98 % | 44,43 % | 45,19 % |
| PaySim | Híbrido: 29,41 % | 29,41 % | 31,75 % |

Son medias de las semillas 42, 52 y 62 sobre un solo período por dataset.
No prueban superioridad estadística ni validez en producción. En PaySim el
híbrido no mantuvo en test el presupuesto operativo de falsas alarmas fijado
en validación. No se utilizó la totalidad de los CSV.

Las corridas y reportes anteriores se retiraron del proyecto a un respaldo local
fuera del repositorio. El código de experimentación anterior permanece disponible,
pero no es la ruta recomendada para exponer o reproducir los resultados actuales.

## Instalación en otra computadora

Necesitas Git, Python 3.12 de 64 bits y conexión para instalar dependencias.
La corrida publicada se ejecutó en CPU; una GPU no es obligatoria.

```powershell
git clone https://github.com/Brando-01/Seminario_1.git
cd Seminario_1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests -q
```

`requirements-tested.txt` registra las versiones del equipo original (CPU);
`requirements.txt` define los rangos de instalación. Los resultados numéricos
pueden variar entre versiones y equipos. Las pruebas no requieren los datasets.

Para leer las tablas, el Word y los gráficos publicados no necesitas entrenar.
Para ejecutar el modelo, auditar los artefactos completos o usar el panel local,
aporta los CSV y genera una corrida completa, o transfiere de manera separada
la carpeta original `outputs/thesis_v1_20261012_154354` y su caché visual.

## Datos necesarios para entrenar

Crea las carpetas:

```powershell
New-Item -ItemType Directory -Force -Path data\raw\ieee_cis,data\raw\paysim | Out-Null
```

Copia estos archivos descomprimidos:

```text
data/raw/ieee_cis/train_transaction.csv
data/raw/ieee_cis/train_identity.csv
data/raw/paysim/PS_20174392719_1491204439457_log.csv
```

El test oficial de Kaggle IEEE-CIS no se usa para calcular métricas porque no
tiene etiquetas. El test del proyecto se toma de un período posterior del
CSV de entrenamiento etiquetado.

## Ejecutar el protocolo completo

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe thesis_experiment.py --stage all
.\.venv\Scripts\python.exe audit_explainability.py --run latest
.\.venv\Scripts\python.exe characterize_thesis.py --run latest
.\.venv\Scripts\python.exe show_results.py --thesis-run latest --details
```

También puedes usar `.\project.ps1 thesis-complete`.
El entrenamiento puede tardar. No lances corridas simultáneas si usas `latest`.
Cada ejecución nueva conserva su identificador; los hashes protegen las
configuraciones, fuentes, datos y modelos congelados.

Con los artefactos originales ya presentes, muestra la corrida registrada sin
volver a entrenar:

```powershell
.\.venv\Scripts\python.exe show_results.py --thesis-run thesis_v1_20261012_154354 --details
.\.venv\Scripts\python.exe audit_explainability.py --run thesis_v1_20261012_154354
.\.venv\Scripts\python.exe dashboard_app.py --no-prepare --port 8766
```

Abre http://127.0.0.1:8766/#resultados. El panel presenta predicciones guardadas,
subgrafos reales y animaciones esquemáticas del funcionamiento de los modelos.
No entrena modelos. Si falta la caché, inicia sin `--no-prepare`; requiere
los CSV y una corrida completa. `--rebuild-cache` reconstruye los ejemplos.

## Distribución del código

```text
configs/                  Configuraciones de las rutas anteriores
data/raw/                 CSV originales (no se publican)
src/data/                 Carga, particiones temporales y preprocesamiento
src/graph/                Relaciones causales y subgrafos
src/models/               Baseline, AE, GraphSAGE e híbrido
src/training/             Pérdidas, entrenamiento y mini-lotes exactos
src/evaluation/           Métricas
src/explainability/       SHAP del híbrido real con contexto fijo
tests/                    Pruebas automáticas
dashboard/                Interfaz y animaciones
docs/                     Guía de exposición y scripts de generación
reports/                  Únicos resultados publicados de referencia
outputs/                  Artefactos completos locales (no se publican)
thesis_experiment.py       Protocolo actual: congelar, entrenar, evaluar, auditar
characterize_thesis.py     Caracterización de los períodos utilizados
audit_explainability.py    Auditoría de resultados y SHAP
show_results.py           Lectura de métricas sin reentrenar
dashboard_app.py           Servidor local del panel
project.ps1               Comandos simplificados para PowerShell
```

AP es Average Precision, no accuracy. Precision, Recall y F1 usan el umbral
elegido en validación; el punto operativo de falsas alarmas es otro umbral.
La media de F1 no se calcula a partir de Precision y Recall promedios.

SHAP explica atributos transformados del nodo objetivo con sus vecinos y
aristas fijos. El ranking usa 16 casos aleatorios por dataset; cuatro casos
adicionales ilustran TP, FP, FN y TN. No es atribución causal ni de aristas.

Git excluye `.venv`, CSV, matrices, predicciones completas, cachés y checkpoints.
Se publican código, documentación, reportes numéricos y gráficos actuales.
La ficha y el artículo académico aún deben alinearse con el protocolo ejecutado.
