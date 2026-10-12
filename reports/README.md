# Evidencia publicada de la corrida actual

Única referencia actual: `thesis_v1_20261012_154354`.

Esta carpeta contiene copias exactas de los reportes, configuraciones,
protocolo, métricas por modelo/semilla, caracterización, conteos de preparación,
auditorías y gráficos SHAP generados en la corrida local completa.
Los reportes no se recalcularon ni se alteraron para publicarlos.

`results.json` contiene 24 registros. Las auditorías de resultados y SHAP
pasaron en el equipo original con todos los artefactos disponibles.
Las copias de esas auditorías registran lo ocurrido, pero no sustituyen una
auditoría nueva ni permiten verificar checkpoints que aquí no se incluyen.

No se publican CSV, matrices, scores por transacción, checkpoints ni valores
SHAP individuales. La carpeta `outputs/thesis_v1_20261012_154354` original se
conserva íntegra localmente. Para reproducir desde un clon, sigue el README.

Se retiraron 13 carpetas de corridas anteriores y cinco reportes/guías obsoletos
del proyecto, con respaldo recuperable fuera del repositorio en
`../_respaldo_resultados_anteriores_20261012/`.
El respaldo no se publica. La exploración previa formó parte del desarrollo:
esta limpieza no elimina su existencia ni convierte los resultados en prueba
de superioridad estadística o de ausencia universal de fuga de información.

El código de experimentación anterior se mantiene por trazabilidad. Las rutas
antiguas no son necesarias para reproducir la comparación actual.
