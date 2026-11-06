# Modelo final: resultado y alcance de la mejora

El candidato actual es `evaluation/results/final_candidate.csv`. Combina cuatro
redes entrenadas con seis partidos externos, observaciones de las dos cámaras y
decodificación a 25 Hz. Configuración exacta: `FINAL_SELECTION.json`.

**spellF1 del método en validación externa por partidos: 0,6758, frente a 0,6421
de la referencia comparable.** No es una puntuación del partido del concurso.
Los checkpoints finales se ajustaron después con los seis partidos: no se atribuye
a esos pesos una evaluación independiente sobre sus propios datos de entrenamiento.

## Comparación principal, mismos datos de entrenamiento por partición

Se hicieron tres particiones, con dos partidos excluidos en cada una. Los cuatro
componentes y sus normalizadores se entrenaron desde cero con los otros cuatro
partidos. Cada partido recibe predicciones fuera de su propio entrenamiento, en
limpio, con ruido moderado y con estrés. Son seis partidos, no dieciocho muestras
independientes. J03WR9 permaneció cerrado y no entró en el ajuste final.

| Método | spellF1 medio | Peor condición | macroF1 medio |
| --- | ---: | ---: | ---: |
| Arquitectura V6, reentrenada por partición | 0,6421 | 0,5280 | 0,9078 |
| Cuatro redes, decodificación anterior | 0,6705 | 0,5685 | 0,9202 |
| Cuatro redes, interpolación y decodificación a 25 Hz | **0,6758** | **0,5741** | **0,9220** |

La media por partido mejora en los seis casos. Las decisiones de arquitectura
ya habían utilizado estos partidos durante el desarrollo histórico y la elección
del decodificador reutiliza sus predicciones externas. Por tanto son resultados
de desarrollo fuera del ajuste de cada partición, **no una confirmación intacta
del sistema seleccionado**. El resumen exacto conserva todos los valores y las
particiones, no solo los casos favorables.

## Representación y segunda cámara: comparación anterior al ajuste final

Esta ablation usa tres partidos para entrenar y tres para desarrollar, con las
mismas tres condiciones por partido. Los números no corresponden a los pesos
finales ajustados con seis partidos.

| Configuración | spellF1 medio |
| --- | ---: |
| V6, una vista | 0,6453 |
| Cuatro redes con trayectorias individuales, una vista | 0,6668 |
| Cuatro redes y segunda vista sintética | 0,6851 |
| Lo anterior y decodificación a 25 Hz | 0,6892 |

La segunda vista externa se genera con ruido independiente, cortes y repeticiones
de tramos de los otros partidos. La sincronización recupera más del 99,3 % de los
frames elegibles, rechaza los replays del ensayo y supera el 99,97 % de alineaciones
aceptadas dentro de cinco frames. **No son tasas de acierto del partido real**.
Las oclusiones reales de ambas cámaras pueden estar correlacionadas, por lo que
la ganancia de esta simulación puede sobreestimar el beneficio real de la fusión.

En el partido del reto, sin consultar sus etiquetas, la sincronización aporta:

- 80.443 frames alineados, en 84 tramos aceptados.
- 63.474 pares con balón observado por ambas cámaras, reservados para diagnóstico.
- Discrepancia mediana entre esos balones: 0,747 m.
- 4.466 observaciones nuevas donde faltaba el balón táctico.
- Ausencia de balón en frames puntuables: 40,69 % → 37,88 %.

La alineación usa solo constelaciones de jugadores por equipo. El balón no entra
en su búsqueda y se reserva para diagnosticarla. Ningún desplazamiento temporal
ni corrección espacial se fijó manualmente en frames específicos del partido.

## Transferencia a tres partidos de Metrica

Ninguno de los checkpoints de esta tabla se entrenó con los tres partidos Metrica
evaluados. Sus referencias proceden de eventos y son aproximadas; estos partidos
ya habían aparecido en otras etapas del proyecto. No son un nuevo test intacto.

| Checkpoint | spellF1 con referencia corregida | Con referencia histórica |
| --- | ---: | ---: |
| V6, entrenamiento nativo | 0,6576 | 0,6329 |
| V12, entrenamiento nativo | 0,6882 | 0,6618 |
| Final, seis partidos nativos y 25 Hz | **0,6918** | **0,6648** |

Cada columna mantiene las mismas referencias y los nueve casos para todos los
modelos. No se comparan estos valores con los del concurso o los nativos como si
fueran una misma evaluación. La configuración final se congeló antes de esta prueba
(`b2581ca`) y no se retocó después de leer sus resultados.

## Experimentos que no entraron

Se revisaron doce repositorios y fuentes primarias de investigación. Esta ronda
entrenó once variantes/inicializaciones exploratorias, doce componentes para
validación por partidos y cuatro para el ajuste final: **27 entrenamientos nuevos**.
Con las rondas V6–V9 anteriores, el registro suma 49 entrenamientos/variantes,
además de los reentrenamientos de comprobación anteriores.

Las redes de reconstrucción auxiliar, atención completa, Bi-LSTM y CNN+Bi-LSTM
se ejecutaron y midieron; no superaron la evidencia de la configuración elegida.
Más capacidad o una técnica publicada no fueron suficientes para adoptarlas.
Los pesos, métricas y resultados negativos se conservan localmente.

## Reproducción y restricciones

El CSV final contiene 199.319 filas y pasa el validador oficial. CPU y MPS generan
exactamente el mismo archivo en esta máquina:

`75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e`.

V12 también se reprodujo desde cero, incluida la alineación sin caché, con el mismo
CSV byte a byte. La entrega local empaqueta código y los cuatro checkpoints,
excluyendo vídeos, tracking, alineaciones precalculadas y ground truth. La
comprobación del paquete se registra en `FINAL_ARTIFACT.json`.

```bash
# Con los cuatro pesos congelados y los datos proporcionados en data/:
OMP_NUM_THREADS=6 .venv/bin/python -m possession.competitive_bundle
.venv/bin/python evaluation/validate_submission.py evaluation/results/final_candidate.csv
```

Para preparar el entrenamiento en una copia nueva y repetirlo sin sobrescribir los
pesos congelados:

```bash
.venv/bin/python scripts/fetch_v6_native.py
OMP_NUM_THREADS=6 .venv/bin/python scripts/prepare_final_training.py
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v13_cv --final-only --output-root data/work/retrained_final
# Validación por partidos completa, en una raíz separada:
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v13_cv --output-root data/work/repeated_cv
```

El entrenamiento reproducido utiliza MPS y dependencias fijadas; no se promete
igualdad de pesos al entrenar en otra plataforma. Para reproducir exactamente la
predicción se usan los checkpoints congelados y la inferencia CPU indicada.

V5, V6 y V12 siguen conservados. V5 tuvo spellF1 público 0,7067 y V6 0,6595;
son comprobaciones históricas, no criterios usados para elegir este candidato.
**Esta ronda no abrió ni evaluó las etiquetas públicas o privadas del concurso.**
La puntuación del nuevo archivo en el partido completo sigue siendo desconocida.

## Ronda posterior V17–V22

Se ejecutaron 41 alternativas y 17 ajustes o continuaciones adicionales, incluidos
especialistas a 25 Hz, transiciones dirigidas, duración explícita y correctores
de errores cruzados. Ninguno supera de forma consistente el criterio registrado
de adopción. El candidato y sus métricas anteriores se conservan. Véase
[el informe completo](EXPERIMENTOS_V17_V22.md), que distingue sus subconjuntos
de desarrollo del resultado global y conserva los resultados negativos.
