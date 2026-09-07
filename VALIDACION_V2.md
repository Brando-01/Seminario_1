# Comparacion reproducible de los dos experimentos

Desde la carpeta del proyecto, con el entorno virtual activo:

```powershell
python -m pytest -q
python validate_experiments.py --epochs 30 --seeds 42 52 62
```

Cada ejecucion crea `outputs/validation_v2_FECHA_HORA/`, sin sobrescribir corridas anteriores. Para ver el ultimo informe terminado en PowerShell:

```powershell
python show_results.py --validation-run latest --details
python audit_validation.py --run latest
```

Este comando avisa si la corrida mas reciente esta incompleta. Alternativamente, para leer el ultimo informe terminado:

```powershell
$report = Get-ChildItem .\outputs\validation_v2_*\REPORT.md | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Get-Content -LiteralPath $report.FullName
```

Si no aparece un informe, revisar la terminal: la corrida puede seguir en curso o haber fallado. `results.json` guarda resultados parciales despues de cada modelo, no prueba que terminaron todos.

## Protocolo

- IEEE: todas las operaciones con TransactionDT entre 30 y 60 dias, extremo superior excluido. Particiones cronologicas aproximadamente 70/15/15, sin cortar timestamps iguales.
- PaySim: todos los registros de los pasos 1 a 24. Entrenamiento 1 a 16, validacion 17 a 20, prueba 21 a 24. No es una evaluacion de los 6.3 millones de registros ni de todo el horizonte.
- Entrenamiento independiente por dataset; no se mezclan datos ni pesos aprendidos.
- Baseline de gradient boosting, Autoencoder, GraphSAGE con rama tabular e hibrido GNN-Autoencoder con rama tabular. Misma particion y variables disponibles para los cuatro.
- Transformaciones ajustadas solo con entrenamiento. Logaritmo de importes/saldos configurados y recorte a 10 desviaciones estandar; se excluye elapsed_days.
- PaySim conserva saldos anteriores, excluye saldos posteriores, reglas de fraude preexistentes y variables de consistencia que utilizan saldos posteriores.
- Historial por horas completas: ninguna transaccion observa otra de su misma hora. Grafos dirigidos de pasado a presente, con contexto historico acumulado sin etiquetas de validacion/prueba. Esto representa inferencia secuencial, no operaciones futuras disponibles en entrenamiento.
- Redes supervisadas: checkpoint por mayor average precision en validacion. AE: minibatches de operaciones legitimas, checkpoint por menor MSE de validacion legitima. El AE aislado usa etiquetas para filtrar normales, pero no aprende a clasificar fraudes como los modelos supervisados.
- Semillas 42, 52 y 62; hasta 30 epocas, parada temprana. Hidden 64, latent 16, dropout 0.15, peso positivo limitado a 20 y peso de reconstruccion hibrida 0.01. Son parametros fijos de esta evaluacion, no una busqueda exhaustiva.
- Baseline conserva su parada interna dentro del conjunto de entrenamiento; el conjunto temporal externo se utiliza para sus umbrales. No todos los modelos reciben igual numero de actualizaciones ni igual capacidad.

## Interpretacion

`auc_pr` en los JSON significa **average precision (AP)** de sklearn, no area trapezoidal. Mayor es mejor. Comparar modelos dentro de cada dataset; no concluir que un dataset es mas facil solo por su AP, porque cambia la proporcion de fraude.

F1, precision y recall usan un umbral elegido en validacion. Las metricas `operating_metrics` usan otro umbral, seleccionado con presupuesto de falsas alarmas en validacion. Ese presupuesto no esta garantizado en prueba.

El informe presenta media y desviacion estandar entre semillas. No son intervalos de confianza ni replicaciones temporales independientes: todas comparten el mismo test. Un mayor promedio no demuestra superioridad estadistica.

Si `best_epoch` coincide con el limite de epocas y la AP de validacion sigue creciendo, el entrenamiento puede estar limitado por presupuesto: no concluir que se alcanzo el potencial del modelo. Una investigacion posterior debe fijar la busqueda de hiperparametros usando train/validacion y reservar otro periodo externo antes de consultar sus resultados. No prolongar corridas hasta obtener el test deseado.

Para afirmar que el hibrido es el mejor debe superar al baseline y a los componentes, no solo al AE. Una conclusion negativa tambien es un resultado valido. Comparaciones con las corridas antiguas no aislan el efecto de las mejoras porque cambiaron periodos, variables y protocolo.

## Evidencia guardada

- `protocol.json` y configuraciones por dataset: parametros de la ejecucion.
- `DATASET/prepared/metrics/prepare_summary.json`: tamanos, prevalencia, tiempos, variables y aristas.
- `DATASET/prepared/checkpoints/preprocessor.pkl`: transformador ajustado solo en train (cargar solo archivos propios confiables).
- `DATASET/seed_N/metrics/`: metricas y curvas de entrenamiento de redes.
- `DATASET/seed_N/checkpoints/`: pesos de las redes.
- `DATASET/seed_N/predictions/`: scores y etiquetas de validacion/test para auditoria.
- `results.json` e `REPORT.md`: comparacion global.

Antes de una afirmacion final de tesis faltan multiples ventanas temporales, analisis de variabilidad temporal y ablaciones de grafo/AE/capacidad. SHAP no se implementa ni se valida con este experimento. Aplicar explicabilidad no garantiza mejorar la deteccion.
