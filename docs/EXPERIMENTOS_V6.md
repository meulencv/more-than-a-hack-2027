# Ronda V6: experimentos, resultados y límites

## Conclusión

Se entrenaron 13 variantes/inicializaciones nuevas, se compararon más de 30
configuraciones de modelo y decodificación y se repitieron desde cero los dos
entrenamientos seleccionados. Hay avances fuertes y confirmados en los partidos
externos; **no hay una mejora demostrada en los cinco minutos públicos**.

El candidato anterior se conserva en `evaluation/results/sequence_quality_gated.csv`
(SHA `793075ee994c33296e138f632116e136b306393702f37408262da87d29837d99`).
V6 se conserva separado en `evaluation/results/v6_selected.csv`
(SHA `fd002b4fe859c36f1b608b2483b9bccbd0f80c425d0b241ddf67ef2f592e20e1`).
No se ha publicado ni enviado ninguno. No se afirma que V6 vaya a ganar el concurso.

## Protocolo previo a los experimentos

`PROTOCOLO_V6.json` se registró en Git (`3eac267`) antes de ampliar la ronda:

- Entrenamiento nativo: J03WMX, J03WOH, J03WOY.
- Desarrollo: J03WN1, J03WPY, J03WQQ.
- Confirmación reservada: J03WR9, sin cargar hasta congelar la selección (`fa13d90`).
- Evaluador oficial y spellF1 medio como criterio; resultados desglosados por partido.
- Tres condiciones por partido: limpio, ruido moderado con desapariciones y estrés
  con huecos de ocho segundos e intercambios de slots. Las alteraciones no consultan
  etiquetas para producir ruido. Tres condiciones no son tres partidos independientes.
- Ninguna etiqueta del partido del concurso en el entrenamiento, ajuste o selección.

Se fijaron modelo y selección antes de la confirmación. El CSV concreto se congeló
en `76d9c42`, antes de una única comprobación pública. No se volvió a ajustar después.

## Desarrollo: comparación real en los mismos nueve casos

Las cifras siguientes son medias sobre tres partidos y tres condiciones, no
puntuaciones del partido del concurso ni intervalos de confianza independientes.

| Familia | Decodificador | spellF1 medio | Peor condición | macroF1 medio |
| --- | --- | ---: | ---: | ---: |
| baseline | fixed | 0.5091 | 0.4202 | 0.8424 |
| hgb_native | fixed | 0.5883 | 0.4847 | 0.9012 |
| hgb_mixed | fixed | 0.5779 | 0.4766 | 0.8980 |
| tcn_global | fixed | 0.5781 | 0.4888 | 0.8963 |
| tcn_local | fixed | 0.5889 | 0.5066 | 0.8980 |
| tcn_segment | fixed | 0.6193 | 0.5322 | 0.8975 |
| tcn_boundary | boundary | 0.6317 | 0.5740 | 0.9001 |
| tcn_boundary | fixed | 0.6280 | 0.5595 | 0.8996 |
| tcn_multistage | fixed | 0.5968 | 0.5134 | 0.9002 |
| tcn_mixed | fixed | 0.5673 | 0.4577 | 0.8903 |
| physics_hgb | fixed | 0.5923 | 0.5034 | 0.9034 |
| physics_tcn | boundary | 0.6378 | 0.5538 | 0.9037 |
| physics_tcn | fixed | 0.6406 | 0.5525 | 0.9045 |
| tcn_ballfree | fixed | 0.4690 | 0.3616 | — |

La mejor combinación predefinida fue la media de las redes de fronteras y física,
con fronteras del primer experto: **0,6453** de spellF1 medio y **0,5898** en la
peor condición, frente a 0,5091 y 0,4202 del candidato anterior.

Las combinaciones geométricas, por entropía, mediana, calidad y experto sin balón
se probaron, no solo se propusieron. Ninguna superó esa combinación. Los resultados
completos se conservan en `V6_RESULTS.json` y `data/work/v6/ensembles/`.

## Qué aportó y qué no

- **Datos adicionales:** se completaron los siete partidos de Sportec. El entrenamiento
  usa tres partidos y referencias nativas. Una nueva ablation mezclando referencias
  débiles Metrica rindió peor en este desarrollo que usar solo nativas.
- **Normalización:** LayerNorm por instante superó a GroupNorm dependiente de toda
  la secuencia en la comparación de arquitectura equivalente. Reduce el desajuste
  entre entrenar ventanas cortas e inferir partes completas.
- **Segmentos:** se dio mayor peso a tramos breves y la suavidad se aplicó dentro de
  estados reales, sin castigar sus fronteras de entrenamiento.
- **Fronteras aprendidas:** una cabeza binaria predice cambios; Viterbi reduce el
  coste de transición cuando hay evidencia de frontera. El mínimo rígido de un
  segundo se elimina en esta variante; puede representar cambios más breves.
- **Física de control:** distancia y velocidad relativa balón-jugador, movimiento
  conjunto, aproximación, impulsos y giro de trayectoria. No basta con ser el más
  cercano al balón. Estas observaciones no usan altura ni etiquetas en inferencia.
- **Hipótesis alternativas negativas:** ignorar el balón por completo, refinar en dos
  etapas, añadir etiquetas débiles al entrenamiento nativo y varias mezclas no
  justificaron su adopción. Se conservan código, modelos locales y resultados.

Las redes son implementaciones propias pequeñas inspiradas en segmentación temporal,
no reproducciones de las arquitecturas completas de [ASRF](https://github.com/yiskw713/asrf),
[MS-TCN](https://arxiv.org/abs/1903.01945) ni [ASFormer](https://github.com/ChinaYi/ASFormer).
No se descargaron ni usaron pesos de esos repositorios.

## Otra inicialización

Se repitieron las dos redes más prometedoras con semilla 2028. La combinación obtuvo
0,6320 frente a 0,6453 con 2027; ambas superan claramente al anterior en desarrollo.
Promediar las cuatro redes dio 0,6439 y no mejoró el peor caso. Se conserva la pareja
original, más sencilla, en vez de añadir modelos sin mejora medida.

## Confirmación reservada, después de congelar

J03WR9 no entró en entrenamiento ni desarrollo. Una vez evaluado no se usó para
retocar nada. Las tres condiciones siguen siendo variantes del mismo partido.

| Condición | Anterior: spellF1 | V6: spellF1 | Anterior: macroF1 | V6: macroF1 |
| --- | ---: | ---: | ---: | ---: |
| clean | 0.5687 | 0.7464 | 0.9009 | 0.9520 |
| moderate | 0.5384 | 0.6574 | 0.8337 | 0.9044 |
| stress | 0.4549 | 0.6269 | 0.7972 | 0.9044 |

## Comprobación pública: resultado desfavorable

| Métrica, mismos cinco minutos | Candidato anterior | V6 |
| --- | ---: | ---: |
| spellF1 | **0,7067** | 0,6595 |
| macroF1 | **0,8768** | 0,8641 |
| Accuracy | **88,88 %** | 88,35 % |

El tramo contiene 17 posesiones. V6 predice 20 y el anterior 13. Estas cifras se
registran sin usarlas para ajustar o elegir una variante posterior. No se concluye
que V6 sea mejor en el concurso, ni que su caída en una muestra pequeña demuestre
que sea peor en el partido completo. Ambos archivos y todos los avances anteriores
se preservan. La discrepancia entre dominios sigue sin explicación demostrada.

La [descripción original de Sportec](https://www.nature.com/articles/s41597-025-04505-y)
explica que posesión se define por control y estado inactivo por interrupción del
árbitro. Por tanto no se atribuye la divergencia a una supuesta definición de
«último toque»: esa explicación no está demostrada. Diferencias de distribución,
calidad de tracking, anotación y el tamaño de la muestra son posibilidades, no
causas identificadas. No se consultaron segmentos públicos para corregirlos a mano.

## Reproducibilidad y fuentes

Se verificaron tamaños y SHA-256 de los 14 XML públicos y timestamps de todos los
partidos. Todos los intervalos comprobados son compatibles con 25 Hz. Fuente:
Bassek, Rein, Weber y Memmert (2025), *An integrated dataset of spatiotemporal and
event data in elite soccer*, [DOI](https://doi.org/10.6084/m9.figshare.28196177.v1),
CC BY 4.0. Manifiesto: `sportec_v6_manifest.json`.

Las dos redes seleccionadas se volvieron a entrenar desde cero en otra carpeta y
produjeron checkpoints idénticos byte a byte. La inferencia CPU y MPS produjo
exactamente el mismo CSV de 199.319 filas. Pasan el validador oficial y las pruebas
automáticas. Los pesos y los datos permanecen en carpetas ignoradas por Git.

```bash
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
.venv/bin/python scripts/fetch_v6_native.py
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v6_data
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v6_experiment tcn_boundary
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v6_physics --prepare --kind tcn
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v6_bundle --export --device cpu
.venv/bin/python evaluation/validate_submission.py evaluation/results/v6_selected.csv
# Repetir entrenamiento sin sobrescribir los congelados:
OMP_NUM_THREADS=6 .venv/bin/python scripts/retrain_v6_selected.py
```

El entrenamiento comprobado utiliza MPS; la inferencia también funciona en CPU.
No se garantiza que entrenar en otra plataforma produzca los mismos pesos. La
confirmación se conserva, no se reabre para elegir parámetros. Para otra ronda
con confirmación independiente harán falta nuevos partidos reservados. Más datos
externos de Metrica con posesión corregida permitirían investigar la transferencia
al dominio objetivo sin recurrir a las respuestas del partido del concurso.
