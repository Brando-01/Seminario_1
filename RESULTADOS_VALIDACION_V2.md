# Resultados de validacion: 6 de septiembre de 2026

Corrida: `outputs/validation_v2_20260906_212206/`. Cuatro modelos, dos datasets y tres semillas (42, 52, 62): **24 entrenamientos terminados**. Maximo de 30 epocas por red. Protocolo y limitaciones: [VALIDACION_V2.md](VALIDACION_V2.md).

## Conclusion

**No se demuestra que el hibrido sea el mejor.** El baseline de gradient boosting obtuvo la mayor AP promedio en ambos experimentos. La comparacion corresponde a estas ventanas, variables y presupuesto; no establece una superioridad universal ni el potencial maximo de las arquitecturas.

AP significa average precision; no es accuracy. Valores mayores son mejores. Media +/- desviacion estandar muestral de tres semillas, todas sobre el mismo test:

| Modelo | AP IEEE-CIS | AP PaySim | F1 IEEE-CIS | F1 PaySim |
|---|---:|---:|---:|---:|
| Baseline | 0.6008 +/- 0.0002 | 0.0785 +/- 0.0243 | 0.5862 | 0.0868 |
| Autoencoder | 0.2970 +/- 0.0086 | 0.0097 +/- 0.0077 | 0.3326 | 0.0328 |
| GNN | 0.4649 +/- 0.0013 | 0.0018 +/- 0.0009 | 0.4237 | 0.0033 |
| Hibrido | 0.4702 +/- 0.0088 | 0.0032 +/- 0.0022 | 0.4205 | 0.0025 |

En IEEE, el hibrido gana 0.0053 de AP promedio a la GNN, pero pierde 0.1306 frente al baseline; su F1 promedio tambien es ligeramente inferior al de la GNN. No gana a la GNN en todas las semillas. En PaySim supera a la GNN en AP promedio, pero queda por debajo del AE y del baseline. Esto no respalda afirmar que fusionar los componentes mejora necesariamente el sistema.

## Cobertura real

| Dataset | Filas train / validacion / test | Fraudes train / validacion / test |
|---|---|---|
| IEEE-CIS | 62,886 / 13,476 / 13,476 | 2,453 / 577 / 597 |
| PaySim | 348,191 / 184,917 / 41,147 | 197 / 38 / 36 |

Se procesaron 89,838 operaciones de IEEE y 574,255 de PaySim. No se entreno con la totalidad de ninguno de los datasets. IEEE usa el intervalo [30, 60) dias de TransactionDT. PaySim usa pasos 1..24; su validacion y prueba cubren apenas cuatro horas cada una.

PaySim tiene muy pocos fraudes en validacion y prueba para sostener una conclusion robusta. Ademas, la prevalencia cambia: aproximadamente 0.0566% en train, 0.0205% en validacion y 0.0875% en test. No comparar las AP entre datasets como si tuvieran igual dificultad.

## Umbral operativo y limitaciones

- En PaySim, el hibrido obtuvo recall de prueba **0 en las tres semillas** con el umbral seleccionado para un presupuesto de 10 falsas alarmas por 1000 legitimas en validacion. No debe recomendarse para uso practico con este resultado. La tabla anterior usa otro umbral: el que maximiza F1 en validacion.
- Varias redes IEEE eligieron la epoca 30 y su AP de validacion aun crecia: hay indicios de limitacion por presupuesto, no prueba de convergencia.
- Tres semillas no son tres periodos independientes ni proporcionan por si mismas significancia estadistica.
- Se modificaron ventanas, variables y entrenamiento respecto de las corridas antiguas. No atribuir los cambios de AP exclusivamente a las mejoras de codigo.
- No se implemento SHAP en esta intervencion, ni se verifico novedad academica o calidad de explicaciones.

## Cambios implementados y verificados

- Particiones sin timestamps compartidos; historial y aristas estrictamente anteriores al instante actual.
- Ventanas completas aun cuando no se especifica limite de filas.
- Logaritmos configurables, tratamiento de infinitos y recorte numerico ajustado con train.
- AE con minibatches normales y restauracion por menor error de validacion; GNN/hibrido con restauracion por mayor AP de validacion.
- Configuracion separada v2, exclusiones de saldos posteriores en PaySim, resultados por semilla y preservacion de corridas antiguas.
- Scores, etiquetas, checkpoints de redes, preprocesador e historiales guardados.
- **15 pruebas automaticas aprobadas**; auditoria de los 24 modelos reprodujo metricas y umbrales desde sus predicciones, verificando particiones comunes y scores finitos.

## Ver sin reentrenar

```powershell
python show_results.py --validation-run validation_v2_20260906_212206 --details
python audit_validation.py --run validation_v2_20260906_212206
```

## Siguiente etapa de investigacion

Ampliar la cobertura temporal y cantidad de fraudes de PaySim, fijar otra evaluacion externa y realizar busqueda de hiperparametros solo con entrenamiento/validacion. Evaluar mas presupuesto de entrenamiento y ablaciones controladas: sin grafo, sin AE y con capacidad comparable. No reutilizar estos tests para seleccionar una configuracion ganadora. Solo despues evaluar la aportacion de explicabilidad, sin presentarla como garantia de mayor precision.
