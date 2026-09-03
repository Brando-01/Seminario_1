# Deteccion de fraude con GNN y Autoencoder

Implementacion experimental para comparar cuatro enfoques de deteccion de fraude:

1. Baseline tabular supervisado.
2. Autoencoder como detector de anomalias.
3. GraphSAGE para aprender relaciones entre transacciones.
4. Modelo hibrido GraphSAGE-Autoencoder.

Se ejecutan dos experimentos independientes. IEEE-CIS usa entidades inferidas (tarjeta, correo y dispositivo); PaySim usa las cuentas de origen y destino. Los datasets no se mezclan y las conclusiones se obtienen dentro de cada uno.

PaySim incluye dos variantes para una ablacion transparente: la principal usa los cuatro saldos brutos disponibles en la transaccion y variables historicas causales; la variante `strict` excluye completamente los saldos. Ninguna usa `isFlaggedFraud` ni los errores de consistencia que delatan el simulador.

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

Desde la raiz del proyecto:

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

El resultado esperado es `7 passed`.

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
