# Caracterización IEEE

Descriptive characterization only; no model/threshold selection from test.

Filas en ventana: 136,884. Columnas antes de transformación: 440.

## Particiones

| Partición | Operaciones | Fraudes | Tasa % | Tiempo mínimo | Tiempo máximo |
|---|---:|---:|---:|---:|---:|
| train | 61720 | 2386 | 3.8658 | 2592014 | 4319966 |
| validation | 28118 | 1241 | 4.4135 | 4320017 | 5183966 |
| test | 47046 | 2045 | 4.3468 | 5184045 | 6479999 |

## Faltantes en train

| Columna | Faltantes % |
|---|---:|
| id_24 | 99.49 |
| id_07 | 99.45 |
| id_08 | 99.45 |
| id_23 | 99.45 |
| id_21 | 99.45 |
| id_22 | 99.45 |
| id_27 | 99.45 |
| id_25 | 99.45 |
| id_26 | 99.45 |
| dist2 | 94.21 |
| id_18 | 94.19 |
| D7 | 93.87 |
| id_33 | 92.24 |
| id_04 | 91.80 |
| id_03 | 91.80 |
| id_34 | 91.77 |
| id_30 | 91.76 |
| id_32 | 91.76 |
| V156 | 91.36 |
| V157 | 91.36 |

Las estadísticas de test son descriptivas posteriores a la congelación; no modifican el diseño. El preprocesador se ajusta exclusivamente con train.