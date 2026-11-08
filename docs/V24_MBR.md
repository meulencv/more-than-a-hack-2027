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

## Evidencia hasta ahora

Solo es **sintética** (`python -m scripts.v24_synthetic_check`, sin datos del partido),
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
