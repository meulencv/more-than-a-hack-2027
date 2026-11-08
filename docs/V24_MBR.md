# V24: decodificación MBR para spellF1 (branch `exp/mbr-spellf1`)

## Dónde falla el modelo (resumen de V17–V23)

1. **Posesiones breves.** Recall a IoU 0,5: 10 % en las de menos de 1 s, 43 % de
   1 a 3 s, 72 % de 3 a 10 s y 86 % a partir de 10 s. Casi la mitad de las
   posesiones duran menos de 3 s, y cada una cuenta como un objeto. El modelo
   casi nunca emite rachas ≤0,8 s (1 frente a 74 reales en limpio).
2. **Por qué las reglas no lo arreglan.** Insertar una posesión breve es una
   apuesta asimétrica: si es falsa, suma un FP y además puede partir una posesión
   larga que ya estaba bien, con lo que se pierde también un TP. Por eso las reglas
   fijas de V23 (confirmación, márgenes, fusiones) empeoran el total: insertan y
   borran sin tener en cuenta el coste de cada caso concreto.
3. **Cambio de dominio.** En el partido del reto falta el balón en el 37,9 % de
   los frames puntuables, y en la condición de estrés el spellF1 cae a 0,57. Donde
   no hay balón las probabilidades son planas, y un decodificador con costes fijos
   trata igual las zonas seguras y las inciertas.
4. **Falsos D.** Un D falso dentro de una posesión la parte en dos. V23 conservaba
   los D a propósito, así que nadie ha atacado este error.

## La idea: decodificar por el riesgo esperado, no por el camino más probable

Viterbi devuelve el camino MAP, que maximiza la verosimilitud de la secuencia. La
métrica del ranking es otra: un F1 por objetos que no se descompone por frames. La
decodificación de **riesgo bayesiano mínimo** (MBR, estándar en reconocimiento de voz:
segmental MBR, Goel & Byrne 2000; Kumar & Byrne 2002) elige el etiquetado con el
**spellF1 esperado** más alto bajo la distribución posterior del propio modelo.

`possession/mbr_decoder.py`:

- Toma la **misma CRF** que `decode_fine` (mismas emisiones a 25 Hz y mismos costes
  modulados por la cabeza de límites) y la usa como distribución `P(s) ∝ exp(β·score)`.
- Muestrea M=32 caminos exactos con forward filtering y backward sampling (FFBS).
- Genera candidatos con Viterbi a escalas de coste 0,6–1,6. La escala 1 reproduce el
  decodificador congelado bit a bit (hay un test que lo comprueba), así que MBR siempre
  puede quedarse con él.
- Elige el candidato con mayor spellF1 medio contra las muestras. Opcionalmente lo
  refina con edición local: borrar posesiones breves, insertar las que aparecen en
  ≥20 % de las muestras y mover cada límite a la mediana de las muestras. Una edición
  solo se acepta si sube el spellF1 esperado.
- Usa un spellF1 propio con barrido O(n+m). Es idéntico a la métrica oficial en 30
  pruebas aleatorias, hasta 12 decimales.

Así se resuelve directamente el dilema del punto 2: una posesión breve se inserta solo
cuando su ganancia esperada compensa el riesgo de romper una larga. En las zonas sin
balón las muestras discrepan, y el decodificador se vuelve conservador allí sin que
haya que fijarlo a mano (punto 3).

## Evidencia sintética

La primera prueba fue **sintética** (`python -m scripts.v24_synthetic_check`, sin datos del partido),
con posteriores mal especificadas, oclusiones del 35 % y ruido correlacionado:

| Decodificador | spellF1 (5 semillas) |
| --- | ---: |
| Viterbi congelado | 0,7959 |
| MBR solo candidatos, β=4 | **0,8104** (no empeora en ninguna semilla) |
| MBR con refinado, β=4 | 0,8063 |
| MBR con refinado, β=8 | 0,8066 |

Con β=1 la posterior es demasiado difusa: sus muestras tienen unas 10 veces más
posesiones que la verdad. **β es el hiperparámetro que hay que elegir por validación
cruzada.**

## Para medirlo de verdad

Con las predicciones fuera del entrenamiento de V13 en `data/work/v13/fold*/`:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m possession.v24_mbr --workers 6
.venv/bin/python -m unittest tests.test_mbr_decoder
```

El script aplica el protocolo de adopción de V17–V23: elige en J03WOH/J03WPY y exige
que el elegido mejore cada uno de J03WOY y J03WQQ, sin bajar el macroF1 más de 0,002.
No lee las etiquetas del reto. Si se adopta, basta con cambiar `decode_fine` por
`decode_mbr` en `competitive_bundle`.

## Reproducción externa del 8 de noviembre de 2026

Se evaluó `ec38a8d` frente al Viterbi congelado de `90a56a5`. Cada método
recibió las mismas probabilidades y fronteras de las cuatro redes V13, fuera
del entrenamiento en cada partido. Son seis partidos Sportec con tres
condiciones cada uno. Las 18 observaciones comparten seis referencias de
partido y ya forman parte del desarrollo. La evaluación se limitó a esas
referencias externas y mantuvo cerrado J03WR9.

| Configuración | spellF1 medio | Peor caso | macroF1 medio | Δ selección | Δ J03WOY | Δ J03WQQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Viterbi congelado | 0,675775 | 0,574096 | 0,921978 | — | — | — |
| MBR β=3 | 0,676893 | 0,577227 | 0,921949 | −0,001636 | +0,001243 | −0,003119 |
| MBR β=5 | 0,675837 | 0,578547 | 0,921902 | −0,001662 | −0,000299 | −0,000732 |
| MBR β=8 | 0,675286 | 0,575814 | 0,921877 | −0,001365 | −0,000650 | −0,003205 |
| MBR β=4 con refinado | 0,676514 | 0,585109 | 0,921951 | −0,002190 | +0,001760 | −0,001287 |
| MBR β=8 con refinado | 0,675379 | 0,571163 | 0,921938 | −0,001584 | +0,000369 | −0,001264 |
| MBR β=4, ganador sintético añadido | 0,677114 | 0,582609 | 0,921966 | −0,001357 | +0,002403 | −0,001457 |

La selección usa la media de J03WOH y J03WPY. En el barrido de la rama, β=8
sin refinado es la mejor variante MBR de esa partición. Pierde frente a
Viterbi en selección y también en ambos partidos de confirmación. El β=4
sin refinado se probó por separado porque fue el mejor en sintético y no
estaba en el barrido externo. Tampoco supera el criterio. Se conserva Viterbi.

### Diagnóstico para la siguiente iteración

Se añadieron dos controles exploratorios. Aplican a ambas mitades el mismo
Viterbi con costes de transición multiplicados por 0,6 o 0,8. MBR β=4
eligió precisamente esas escalas en las 36 mitades evaluadas, 11 veces 0,6 y
25 veces 0,8.

| Decodificador | Limpio | Moderado | Estrés | Media |
| --- | ---: | ---: | ---: | ---: |
| Viterbi congelado | 0,730688 | 0,661762 | 0,634875 | 0,675775 |
| Viterbi fijo 0,6 | 0,729232 | 0,662504 | 0,639195 | 0,676977 |
| Viterbi fijo 0,8 | 0,730081 | 0,661714 | 0,635645 | 0,675813 |
| MBR β=4 | 0,729730 | 0,663315 | 0,638298 | 0,677114 |

- **Coste de transición.** El control 0,6 reproduce casi toda la ganancia media
  de MBR β=4. La selección por muestras añade 0,000137 sobre ese control y
  pierde 0,000897 en estrés. En un nuevo ensayo, compara MBR con ese control
  y con Viterbi congelado para aislar lo que aporta la selección.
- **Posesiones breves.** A IoU 0,5, Viterbi recupera 136 de 1.029 posesiones
  de menos de 1 s, β=4 recupera 140 y el control 0,6 recupera 148. Entre 1 y
  3 s los recuentos son 1.200, 1.246 y 1.273 sobre 2.499. Son recuentos
  acumulados sobre las tres condiciones, que repiten las referencias de
  cada partido. Conviene medir también falsas divisiones de posesiones largas
  al variar los costes H↔A.
- **Calibración del objetivo.** El β=4 aumentó el `spellF1` esperado contra
  sus propias muestras en ambas mitades de los 18 casos. Frente a la
  referencia externa, perdió en 8 de esos 18 casos. El margen interno sirve
  para elegir dentro de la posterior, pero en estos datos no garantiza una
  mejora externa. Una prueba posterior puede congelar la calibración por
  condición antes de medirla en partidos reservados.
- **Balón muerto.** El β=4 cambia la máscara D en 1.653 frames acumulados y
  el F1 vivo/muerto medio pasa de 0,981230 a 0,981023. Un control con D fijo
  y costes H↔A ajustados por separado permitiría distinguir recuperación de
  control y cortes causados por D.

### Archivos y comandos

- [Barrido de la rama](v24_results/branch_sweep.json) con los 108 registros,
  métricas, decisión y diagnósticos de la posterior por mitad.
- [β=4 sin refinado](v24_results/beta4.json) con la comprobación externa de
  la configuración que ganó en sintético.
- [Controles y duración](v24_results/diagnostics.json) con métricas por
  condición, posesiones recuperadas por duración y cambios en la máscara D.

Desde una copia con los caches externos V10 y V13 preparados:

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_mbr_decoder -v
OMP_NUM_THREADS=1 python -m scripts.v24_synthetic_check
OMP_NUM_THREADS=1 python -m possession.v24_mbr --workers 4
OMP_NUM_THREADS=1 python -m scripts.v24_diagnostics --workers 4
```

El último comando reproduce también el β=4 sin refinado. Los resultados
se escriben bajo `data/work/`. Se usó Python 3.12.15 en arm64 con NumPy 2.5.4,
pandas 3.1.1, PyTorch 2.15.0 y joblib 1.6.0. Si se reutiliza un caché V10 desde
otra copia, su firma guarda una clave con la ruta absoluta del módulo original.
La prueba aislada trasladó esa clave conservando la comprobación de todos los
hashes de contenido. Con un caché generado en la misma copia no hace falta
ese ajuste.
