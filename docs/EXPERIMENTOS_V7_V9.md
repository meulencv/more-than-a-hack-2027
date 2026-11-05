# Transferencia de dominio, objetos temporales y memoria de control

## Resultado de estas tres rondas

Se entrenaron **nueve variantes adicionales** después de las trece de V6: cuatro
de transferencia, tres de objetivos temporales y dos de memoria de control. Se
compararon 37 configuraciones adicionales de fusión/decodificación. La suma de
V6–V9 es 22 variantes entrenadas, más de 60 comparaciones y dos reentrenamientos
exactos del candidato V6. Los experimentos nuevos **no justifican sustituir V6**:
sus pequeñas ventajas de desarrollo son inestables o empeoran el peor caso.

Ninguna ronda V7–V9 abrió ni evaluó el ground truth público del concurso. Tampoco
reabrió J03WR9: su confirmación ya se había consumido en V6. Las cifras siguientes
son desarrollo exploratorio sobre tres partidos y tres condiciones de observación,
no nueve partidos independientes ni una estimación imparcial del leaderboard.

## V7: aprender equipos desde retransmisiones

Se localizaron datos abiertos de la A-League australiana 2024/2025 en el repositorio
oficial de [SkillCorner](https://github.com/SkillCorner/opendata), fijado en el commit
`4340d274572876239c154c90bc507a9b3250a656`. La documentación describe tracking de
retransmisión, posesión y detecciones frente a extrapolaciones. Ninguno de estos
partidos es el Barcelona–Athletic del concurso.

Se descargaron ocho partidos: seis de entrenamiento y dos de desarrollo. Dos
partidos adicionales quedaron reservados y **no se descargaron ni evaluaron**.
Los restantes de la fuente tampoco se usaron. Protocolo previo: `PROTOCOLO_V7.json`,
commit `3ebd793`. El repositorio publica licencia MIT y solicita acreditar a
SkillCorner; la licencia, README y archivos originales se conservan localmente.

Precauciones de referencia y muestreo:

- `possession.group` es un **profesor algorítmico**, no ground truth corregido.
  Los valores reales `home team` y `away team` se convierten a H/A. Cualquier
  ausencia queda desconocida; nunca se convierte a D.
- No se usó el complemento de las fases de juego como etiqueta de balón parado.
- Las coordenadas se interpolan de 10 a 25 Hz sin atravesar observaciones ausentes;
  los atributos finales a 5 Hz coinciden con observaciones originales alternas.
  Las etiquetas se mantienen desde el tick anterior y las partes se separan.
- Se probaron tracking completo, solo posiciones `is_detected`, y ambas vistas
  mezcladas con los tres partidos nativos de entrenamiento.
- Una TCN binaria aprende H/A. La fusión solo redistribuye la masa H/A del experto
  V6 y deja **P(D)** intacta. El estado D final sí puede variar al decodificar,
  porque compite con las nuevas probabilidades H/A.

| Transferencia | Peso del experto binario | spellF1 nativo medio | Peor caso |
| --- | ---: | ---: | ---: |
| Referencia V6 | 0 | 0,6453 | 0,5898 |
| Tracking completo | 0,25 | 0,6484 | 0,5723 |
| Solo detecciones | 0,25 | 0,6459 | 0,5800 |
| Mixto, semilla 2027 | 0,25 | 0,6512 | 0,5868 |
| Mixto, semilla 2028 | 0,25 | 0,6444 | 0,5792 |

Se probaron también pesos 0,5 y 1,0: empeoraron. La coincidencia H/A con el profesor
en los dos partidos nuevos pasa de aproximadamente 80–81 % con V6 a 84–86 % con
las redes adaptadas. Solo el 53–54 % de sus instantes tiene referencia H/A, por
lo que **esta coincidencia no equivale al score oficial H/A/D**. La mejora de
spellF1 completo no resiste la repetición de semilla. No se adopta la adaptación.

## V8: entrenar objetos temporales

Protocolo registrado antes de entrenar en `PROTOCOLO_V8.json` (`cfc7ba2`). Se
conservan los 156 atributos del experto físico, arquitectura, 800 actualizaciones,
semilla y partición de V6. Las pérdidas nuevas se aplican únicamente a referencias
externas de entrenamiento:

- CE ponderada por la inversa de la longitud del segmento, para igualar segmentos.
- IoU suave por posesión H/A completa, usando como región local la posesión y sus
  dos segmentos vecinos. Evita que otra posesión legítima del mismo equipo infle
  el denominador. Se excluyen segmentos truncados y vecinos desconocidos.
- CE focal más la misma IoU, para concentrarse en ejemplos difíciles.

La IoU suave es un sustituto diferenciable, **no el evaluador oficial**. Todas las
comparaciones finales se realizan con el evaluador oficial de intervalos.

| Objetivo | Solo red, decodificador fijo | Mejor mezcla con V6 | Peor caso de esa mezcla |
| --- | ---: | ---: | ---: |
| Inversa de longitud | 0,5965 | 0,6253 | 0,5587 |
| IoU temporal | 0,6413 | 0,6475 | 0,5803 |
| Focal + IoU | 0,6409 | 0,6467 | 0,5834 |

Se compararon seis formas prefijadas de decodificación por red: fija, fronteras
propias, fronteras anteriores y tres mezclas iguales. La ponderación inversa rindió
claramente peor; enfatizar mucho los tramos cortos no basta para reconstruir
posesiones. Las pérdidas por IoU quedaron cerca de V6, sin ventaja relevante ni
mejor peor caso. No se añade otra red por unas milésimas de desarrollo.

## V9: memoria del último control fiable

Protocolo previo: `PROTOCOLO_V9.json` (`98a708b`). Se añadieron 23 atributos a los
156 anteriores: equipo y jugador del último/próximo control estable, edad de los
anclajes, acuerdo entre ellos, proximidad actual a esos jugadores, continuidad e
indicio cinemático de vuelo. Los anclajes se apoyan en distancia y velocidad relativa
balón-jugador sostenidas al menos seis frames. Los parámetros se fijaron antes del
entrenamiento y no se buscaron contra etiquetas del concurso.

Los anclajes son atributos que el modelo puede rechazar, no etiquetas ni
correcciones impuestas. La falta de anclaje nunca significa balón parado. La
memoria se reinicia al cambiar de parte y es simétrica al intercambiar equipos.

| Modelo | spellF1 medio | Peor caso |
| --- | ---: | ---: |
| TCN con anclajes, fijo | 0,6334 | 0,5267 |
| Mejor mezcla con V6 | 0,6438 | 0,5826 |
| HGB con anclajes | 0.5945 | 0.5007 |

No supera la representación anterior. Que una hipótesis sea coherente con la
definición de posesión no garantiza que sus observaciones automáticas sean mejores:
los anclajes también heredan ruido y errores de proyección. Esa es una limitación
del mecanismo, no una causa del resultado aislada experimentalmente.

## Conservación y reproducción

Modelos y cachés permanecen bajo `data/work/v7`, `v8` y `v9`, ignorados por Git.
Manifiesto de los 16 archivos externos usados: `skillcorner_v7_manifest.json`.
Resúmenes y hashes: `V7_V9_RESULTS.json`. Los dos CSV anteriores siguen intactos:

- `evaluation/results/sequence_quality_gated.csv`: SHA
  `793075ee994c33296e138f632116e136b306393702f37408262da87d29837d99`.
- `evaluation/results/v6_selected.csv`: SHA
  `fd002b4fe859c36f1b608b2483b9bccbd0f80c425d0b241ddf67ef2f592e20e1`.

No se exportó otro CSV ni se usó el tramo público para escoger V7–V9. No hubo
publicación, envío, subida de datos ni servicios pagados. Los comandos siguientes
son para una carpeta de trabajo nueva; algunos scripts rechazan sobrescribir pesos:

```bash
.venv/bin/python scripts/fetch_v7_skillcorner.py
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v7_transfer --prepare --kind all
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v7_transfer --kind detected
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v7_transfer --kind mixed
PYTHONPATH=. OMP_NUM_THREADS=6 .venv/bin/python scripts/repeat_v7_mixed.py
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v8_objects inverse
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v8_objects iou
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v8_objects focal_iou
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v9_control --prepare --kind tcn
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v9_control --kind hgb
OMP_NUM_THREADS=6 .venv/bin/python -m unittest discover -s tests
```

La principal incertidumbre sigue siendo la transferencia al tracking de Metrica.
Tres partidos nativos reutilizados para desarrollo no resuelven esa incertidumbre.
Partidos adicionales independientes con referencias fiables permitirían distinguir
mejor las mejoras reales de la variación entre partidos y semillas.
