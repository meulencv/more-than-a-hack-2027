# Auditoría de referencias externas y robustez

5 de noviembre de 2026. No se han leído etiquetas públicas ni privadas del partido
del concurso. No se han modificado `reference.py`, los modelos ni los candidatos
congelados. Todos los archivos de esta auditoría son adicionales.

## Error reproducible en la referencia débil

`reference._one_touch` examina `rows[i:]`. Cuando el evento actual es `BALL LOST`,
la propia fila satisface inmediatamente `row.team == team and row.type == "BALL LOST"`.
La función devuelve `True` sin mirar eventos posteriores. Por tanto, todos los
eventos `BALL LOST` de un rival se excluyen como posibles cambios de posesión,
aunque `CONTROL` y la documentación indican que deben contribuir evidencia de
tenencia previa al perder el balón.

Se demuestra con una secuencia sintética: local pasa; rival tiene una pérdida
registrada durante dos segundos; cien fotogramas después el local vuelve a pasar.
La referencia actual nunca concede posesión al rival. La alternativa inicia la
inspección en `rows[i + 1:]`, conservando la supresión de recuperaciones seguidas
por pérdida en cinco fotogramas.

Esta corrección concuerda mejor con la intención del código. **No convierte los
eventos en ground truth perfecto**: el significado de control y las fronteras
temporales continúan siendo aproximados. No se ha ajustado nada contra imágenes
del partido del reto ni contra sus etiquetas.

| Partido externo | Pérdidas marcadas touch originalmente | Tras corrección | Frames cambiados | Posesiones original → alternativa |
| --- | ---: | ---: | ---: | ---: |
| 1 | 257/257 | 23/257 | 3.172 (2,23 %) | 254 → 361 |
| 2 | 233/233 | 24/233 | 4.366 (3,12 %) | 237 → 334 |
| 3 | 288/288 | 38/288 | 669 (0,47 %) | 378 → 407 |

La diferencia en posesiones es grande pese al pequeño porcentaje de fotogramas:
esto es precisamente lo que `spellF1` penaliza y un acierto por frame puede ocultar.

## Efecto de la referencia con predicciones idénticas

Se vuelven a puntuar las mismas probabilidades de v3 almacenadas para los seis
casos externos (tres partidos, dos degradaciones), usando la configuración
`config_external_selected.json`. No se vuelve a entrenar ni se cambia una etiqueta
predicha. El spellF1 medio cambia de **0,6250165** a **0,5566767** al sustituir solo
la referencia débil. Estos números no deben confundirse con el score del reto.

Los resultados están en `data/work/audit_reference/cached_scores.json`, con
ambas referencias explícitas. No comparar una versión nueva contra la referencia
alternativa y una antigua contra la original como si fuera una mejora del modelo.

## Prueba más difícil, sin usar el estado para generar ruido

El generador actual `degrade` usa las etiquetas débiles externas para decidir dónde
sintetizar observaciones de balón muerto a partir de la posición del reinicio.
Eso es legal para construir datos externos, pero crea una distribución artificial
condicionada al objetivo; su validación puede sobreestimar la transferencia a ruido
desconocido. **No es una prueba de fuga de etiquetas del concurso.**

Se añade una prueba distinta, `audit_stress.blind_corruption`, que no lee eventos
ni etiquetas para producir observaciones:

- Huecos del balón en bloques de ocho segundos, con probabilidad 40 % por bloque.
- Ruido autocorrelacionado de 1,1 m en el balón.
- Ruido independiente de 0,35 m en cada trayectoria de jugador.
- Permutación de slots dentro de cada equipo cada veinte segundos, preservando la
  geometría colectiva pero fragmentando las identidades usadas para velocidades.

Un test cambia arbitrariamente todas las etiquetas manteniendo las posiciones y
comprueba que las observaciones corrompidas son exactamente iguales. Se usan los
modelos v3 ya entrenados dejando fuera el partido que se evalúa.

| Partido | spellF1 referencia original | spellF1 referencia alternativa |
| --- | ---: | ---: |
| 1 | 0,4636 | 0,3815 |
| 2 | 0,4793 | 0,4038 |
| 3 | 0,4566 | 0,4413 |

No es una simulación demostrada del concurso ni permite estimar su score. La
disponibilidad del balón de origen en el tracking externo sigue correlacionada
con el estado. Es una prueba de sensibilidad a oclusiones largas y cambios de ID,
que el generador previo no representaba completamente.

## Reproducción y recomendación

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_audit_*.py' -v
.venv/bin/python -m possession.audit_reference
OMP_NUM_THREADS=4 .venv/bin/python -m possession.audit_cached
OMP_NUM_THREADS=4 .venv/bin/python -m possession.audit_stress
```

Pasan cuatro pruebas nuevas. Los datos detallados permanecen en `data/work/`,
excluido de Git. La referencia alternativa se obtiene mediante una copia aislada
de los globals de la función: no hay monkeypatch global ni cambios en los baselines.

Recomendación: conservar las referencias anteriores con nombre explícito, evaluar
los nuevos métodos contra ambas para sensibilidad, y migrar de forma versionada
la referencia después de verificar el significado de los eventos. Priorizar otro
dataset con posesión nativa y referencias independientes en vez de seguir elevando
métricas sobre las mismas tres referencias débiles.

## Comprobación adicional del modelo temporal TCN

Se han aplicado exactamente las mismas observaciones de la prueba de estrés a los
checkpoints TCN entrenados dejando fuera cada partido (500 pasos, TCN48x6).
No hay entrenamiento adicional ni ajuste del estrés para el TCN.

| Referencia débil | HGB v3: spellF1 medio | TCN: spellF1 medio |
| --- | ---: | ---: |
| Original | 0,4665 | 0,4927 |
| Alternativa sin auto-coincidencia | 0,4089 | 0,4299 |

TCN mejora en los partidos 1 y 2 y retrocede en el partido 3. En la referencia
original, el macroF1 medio pasa de 0,5289 a 0,6207. Esto respalda estudiar el modelo
temporal sin demostrar rendimiento en el partido del concurso. Detalle en
`data/work/audit_stress/tcn_results.json`. Reproducción:

```bash
OMP_NUM_THREADS=4 .venv/bin/python -m possession.audit_stress_sequence
```

Antes de migrar referencias, invalidar las cachés de entrenamiento e incluir hashes
del código de referencias y extracción en su identidad. La caché actual se nombra
solo con la configuración y podría reutilizar etiquetas antiguas silenciosamente
tras corregir `reference.py`.
