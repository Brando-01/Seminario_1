# Caracterización PAYSIM

Descriptive characterization only; no model/threshold selection from test.

Filas en ventana: 1,030,563. Columnas antes de transformación: 29.

## Particiones

| Partición | Operaciones | Fraudes | Tasa % | Tiempo mínimo | Tiempo máximo |
|---|---:|---:|---:|---:|---:|
| train | 348191 | 197 | 0.0566 | 3600 | 57600 |
| validation | 226064 | 74 | 0.0327 | 61200 | 86400 |
| test | 456308 | 619 | 0.1357 | 90000 | 259200 |

## Faltantes en train

| Columna | Faltantes % |
|---|---:|
| step | 0.00 |
| type | 0.00 |
| amount | 0.00 |
| nameOrig | 0.00 |
| oldbalanceOrg | 0.00 |
| newbalanceOrig | 0.00 |
| nameDest | 0.00 |
| oldbalanceDest | 0.00 |
| newbalanceDest | 0.00 |
| isFraud | 0.00 |
| isFlaggedFraud | 0.00 |
| amount_log1p | 0.00 |
| destination_is_merchant | 0.00 |
| origin_previous_count_log1p | 0.00 |
| origin_previous_amount_sum_log1p | 0.00 |
| origin_previous_amount_mean_log1p | 0.00 |
| origin_time_since_previous | 0.00 |
| origin_is_new | 0.00 |
| origin_amount_vs_previous_mean | 0.00 |
| destination_previous_count_log1p | 0.00 |

Las estadísticas de test son descriptivas posteriores a la congelación; no modifican el diseño. El preprocesador se ajusta exclusivamente con train.