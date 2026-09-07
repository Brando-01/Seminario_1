# Deteccion de fraude con GNN y Autoencoder

## Evaluacion actual: protocolo v2

Para comparar ambos datasets con ventanas temporales completas, tres semillas y resultados aislados:

```powershell
python -m pytest -q
python validate_experiments.py --epochs 30 --seeds 42 52 62
python show_results.py --validation-run latest --details
```

Consultar [VALIDACION_V2.md](VALIDACION_V2.md) para el protocolo, limites y archivos de evidencia. Esta ruta excluye los saldos posteriores de PaySim, evita historial entre operaciones del mismo instante y selecciona las redes supervisadas por AP de validacion. No ejecuta la totalidad de los datasets ni demuestra superioridad estadistica. Los comandos de abajo se conservan para los experimentos anteriores.

## Implementacion

Implementacion experimental para comparar cuatro enfoques de deteccion de fraude:

1. Baseline tabular supervisado.
2. Autoencoder como detector de anomalias.
3. GraphSAGE para aprender relaciones entre transacciones.
4. Modelo hibrido GraphSAGE-Autoencoder.

Se ejecutan dos experimentos independientes. IEEE-CIS usa entidades inferidas (tarjeta, correo y dispositivo); PaySim usa las cuentas de origen y destino. Los datasets no se mezclan y las conclusiones se obtienen dentro de cada uno.

Las configuraciones antiguas de PaySim incluyen una variante con cuatro saldos e historial y otra `strict` sin saldos. No deben confundirse con el protocolo v2 ni interpretarse como una ablacion controlada si cambian otras variables a la vez. Ninguna de esas configuraciones usa `isFlaggedFraud` ni errores de consistencia.

## Estructura principal

```text
fraud_detection_gnn_ae/
|-- configs/              # configuracion de cada experimento
|-- data/raw/             # archivos originales, sin modificar
|-- outputs/              # EDA, metricas y checkpoints
|-- src/                  # carga, preprocesamiento, grafos, modelos y evaluacion
|-- tests/                # pruebas automaticas
|-- eda.py                # analisis exploratorio
|-- run_experiment.py     # ejecutor de modelos
|-- show_results.py       # visor e interprete de resultados
`-- project.ps1           # comandos simplificados para PowerShell
```

## Instalacion

Para instalar en otra computadora, necesitas Git, Python 3.12 de 64 bits (la serie utilizada en la validacion local), conexion para descargar las dependencias y los CSV originales. Los modelos pueden ejecutarse en CPU; una GPU no es obligatoria. El consumo de memoria y el tiempo dependen de la ventana de datos y del equipo.

En PowerShell:

```powershell
git clone https://github.com/Brando-01/Seminario_1.git
cd Seminario_1
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force -Path data\raw\ieee_cis,data\raw\paysim | Out-Null
```

Copia los archivos descomprimidos a estas rutas:

```text
data/raw/ieee_cis/train_transaction.csv
data/raw/ieee_cis/train_identity.csv
data/raw/paysim/PS_20174392719_1491204439457_log.csv
```

Estos tres archivos bastan para la evaluacion v2. Los archivos test y sample_submission de Kaggle no son necesarios para ese comando. Debes aportar los CSV por separado: los datos, `.venv`, `outputs` y los pesos de redes no estan incluidos en Git.

Sin activar el entorno ni cambiar la politica de PowerShell, puedes ejecutar:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe validate_experiments.py --epochs 30 --seeds 42 52 62
.\.venv\Scripts\python.exe show_results.py --validation-run latest --details
.\.venv\Scripts\python.exe audit_validation.py --run latest
```

Las pruebas no requieren los datasets. El entrenamiento si los requiere. El visor y la auditoria requieren una corrida terminada en `outputs`; en un clon nuevo, primero hay que entrenar. El resumen historico se puede leer sin entrenar en `RESULTADOS_VALIDACION_V2.md`. Las dependencias de `requirements.txt` tienen limites minimos, no versiones fijadas: los resultados pueden variar entre versiones y equipos.

Alternativamente, desde la raiz del proyecto puedes activar el entorno:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si PowerShell bloquea `project.ps1`, habilitalo solamente para la terminal actual:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Flujo recomendado

### 1. Validar el proyecto

```powershell
.\project.ps1 validate
```

La suite actual contiene 15 pruebas.

### 2. Explorar los datos

```powershell
.\project.ps1 eda-ieee -Rows 100000
.\project.ps1 eda-paysim -Rows 100000
```

El EDA muestra desbalance, monto, fraude por tipo de operacion o producto y datos faltantes. Los JSON quedan en:

- `outputs/experiment_1_ieee/eda/eda_summary.json`
- `outputs/experiment_2_paysim/eda/eda_summary.json`

### 3. Hacer una prueba tecnica rapida

```powershell
.\project.ps1 train-ieee -Rows 10000 -Epochs 2 -Seed 42
.\project.ps1 train-paysim -Rows 30000 -Epochs 2 -Seed 42
.\project.ps1 results
```

Estas corridas verifican que todo funciona, pero no son resultados finales de tesis.

### 4. Entrenar una corrida de desarrollo

```powershell
.\project.ps1 train-ieee -Rows 100000 -Epochs 30 -Seed 42
.\project.ps1 train-paysim -Rows 300000 -Epochs 30 -Seed 42
.\project.ps1 results
```

Para reproducir la ablacion estricta sin saldos:

```powershell
.\project.ps1 train-paysim-strict -Rows 300000 -Epochs 30 -Seed 42
.\project.ps1 results-strict
```

### 5. Evaluar estabilidad para la tesis

Cada benchmark repite el experimento con semillas 42, 52 y 62, y presenta media y desviacion estandar:

```powershell
.\project.ps1 benchmark-ieee -Rows 100000 -Epochs 30
.\project.ps1 benchmark-paysim -Rows 300000 -Epochs 30
```

Puede tardar considerablemente en CPU. Para resultados definitivos, aumenta filas y epocas segun la memoria disponible y documenta esos valores. El pipeline actual trabaja con grafos completos en memoria; no se debe intentar cargar los 6.3 millones de filas de PaySim sin implementar muestreo de vecindarios o entrenamiento por bloques.

## Como interpretar los resultados

La metrica principal es **AUC-PR**, porque el fraude es muy poco frecuente. Tambien deben reportarse:

- `Recall`: porcentaje de fraudes reales detectados.
- `Precision`: porcentaje de alertas que realmente eran fraude.
- `F1`: equilibrio entre precision y recall.
- `FP/1000`: falsas alertas por cada 1000 operaciones legitimas; menor es mejor.
- `AUC-ROC`: separacion global de clases, como metrica secundaria.

No elijas un modelo solamente por recall: puede detectar muchos fraudes generando demasiadas falsas alarmas. Para la conclusion, prioriza AUC-PR y F1, y revisa el costo operativo con FP/1000.

## Controles contra fuga de informacion

- Division cronologica 70/15/15 para entrenamiento, validacion y prueba.
- Preprocesamiento ajustado solo con entrenamiento.
- Umbral de clasificacion elegido solo en validacion.
- Metricas calculadas una vez sobre prueba.
- Grafos causales con aristas de transacciones pasadas hacia la actual.
- Validacion y prueba conservan el contexto historico anterior sin enviar informacion futura hacia entrenamiento.
- En PaySim se excluyen `isFlaggedFraud` y los saldos del conjunto principal de caracteristicas.
- Las inconsistencias de saldo de PaySim quedan desactivadas en el experimento principal: pueden revelar artefactos del simulador y producir resultados perfectos que no generalizan.
- Los archivos oficiales de prueba de IEEE-CIS no se usan para evaluar porque no tienen etiquetas.

## Archivos producidos

Cada experimento guarda:

- `metrics/prepare_summary.json`: filas, fraude, particiones y tamano del grafo.
- `metrics/*_metrics.json`: resultado individual de cada modelo.
- `metrics/comparison_latest.json`: ultima comparacion completa.
- `metrics/comparison_seed_<semilla>.json`: corrida reproducible por semilla.
- `checkpoints/*.pt`: pesos entrenados de Autoencoder, GraphSAGE e hibrido.

Los archivos de `data/raw` nunca son sobrescritos.
