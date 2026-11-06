# Ronda V17–V22: diagnóstico de errores y especialistas

**No se adopta ninguna variante nueva.** Se ejecutaron 41 configuraciones
alternativas y 17 ajustes, incluidos dos entrenamientos de continuación. El
candidato conservado mantiene su comparación externa por partidos: **spellF1
0,675775 y macroF1 0,921978**. No son puntuaciones del partido del concurso.

Resultados exactos, hashes de pesos y decisiones:
[EXPERIMENTOS_V17_V22.json](EXPERIMENTOS_V17_V22.json).
Los resultados negativos y los pesos permanecen en `data/work/v17` a `v22`.
El CSV final y su configuración no se modificaron.

## Qué falla

Se diagnosticaron predicciones fuera del entrenamiento de dos partidos externos,
J03WMX y J03WN1. Sus 1.011 posesiones de referencia se examinaron en tres
condiciones de observación; las tres condiciones repiten las mismas referencias,
no son partidos independientes. Se emparejaron posesiones mediante el código
oficial, con IoU 0,5 para este diagnóstico.

| Duración de la posesión real | Posesiones, sumando las tres condiciones | Proporción emparejada |
| --- | ---: | ---: |
| Menos de 1 s | 486 | 10,49 % |
| De 1 a 3 s | 990 | 43,33 % |
| De 3 a 10 s | 1.059 | 72,05 % |
| 10 s o más | 498 | 86,14 % |

Estas proporciones son recall de emparejamiento a IoU 0,5, **no F1**. Casi la
mitad de las posesiones únicas duran menos de tres segundos. Aunque afectan a
pocos fotogramas, cada una cuenta como un objeto en la métrica oficial.

Entre los 3.033 casos hay 570 sin ningún solapamiento con una predicción del mismo
equipo y 588 que la heurística diagnóstica clasifica como absorbidos por una
posesión más larga. Las categorías son aproximaciones sobre intervalos, no
anotaciones humanas de duelos o contactos.

Se generaron seis gráficos de fallos externos y se inspeccionaron cuatro de
ellos, confrontando referencia, predicción, probabilidades, límites, distancias
y velocidad. Están en `data/work/v17/visual_audit/`. En los seis ejemplos el balón
está observado: las oclusiones no explican todos los fallos. Aparecen controles
breves entre posesiones de un mismo equipo y secuencias de cambios rápidos que
el modelo fusiona. En un ejemplo de 1,68 s, el equipo de referencia es el más
cercano al balón solo en el 35,7 % de sus fotogramas; cercanía y control no son
intercambiables. No se dispone del vídeo de estos partidos externos: se
inspeccionan señales y trayectorias, no se afirma haber revisado imágenes de juego.

## Evaluación de las nuevas ideas

Todos los modelos excluyen del entrenamiento los partidos de su evaluación.
Se fijaron configuraciones antes de ejecutarlas. Para cada familia se seleccionó
en J03WOH/J03WPY y solo la opción elegida pasó a J03WOY/J03WQQ. Se exige mejora
de spellF1 en ambos partidos de comprobación y macroF1 medio no inferior en más
de 0,002. Una variante que no supera la selección no se entrena ni puntúa
innecesariamente en la segunda partición.

**Son particiones de desarrollo reutilizadas históricamente y entre hipótesis**,
no un test final intacto. Sus resultados tienen sesgo de selección acumulado.
Los tres regímenes por partido tampoco son seis muestras independientes.

El conjunto actual obtiene **0,709897** en la partición de selección y
**0,688752** en la segunda partición. Estos valores usan otros subconjuntos que
el 0,675775 global; no representan una subida del candidato.

| Familia realmente ejecutada | Mejor alternativa por spellF1 en selección | spellF1 de selección | Resultado |
| --- | --- | ---: | --- |
| Penalizaciones de cambio y agregación por límites, idea ASRF | Coste ×0,5 | 0,708753 | Inferior al conjunto actual |
| Especialistas de posesiones breves, balón muerto y equipo | Mezclar 25 % del especialista de balón muerto | 0,710482 | Mejora pequeña, inconsistente entre partidos |
| Experto temporal a 25 Hz, 800 actualizaciones | Mezcla del 10 % | 0,708224 | Inferior |
| Mezclas por visibilidad, incertidumbre y tipo de decisión | Dar más peso al TCN para balón muerto y a trayectorias para equipo | 0,708048 | Inferior; tampoco supera la condición de macroF1 |
| Red con seis tipos de transición y estado sin transición | Mezclar 25 % de su clasificación | 0,712463 | Mejora un partido y empeora el otro |
| Experto de 25 Hz prolongado hasta 4.000 actualizaciones | Mezcla del 25 % | 0,710359 | Empeora en la segunda partición |
| Duración explícita aprendida del entrenamiento | Corrección de riesgo ×0,5 | 0,709499 | Inferior |
| Corrector de errores y puerta de intervención aprendida | Intervención máxima 25 % | 0,706117 | Inferior |
| Corrector temporal residual sobre probabilidades previas | Residuo ×0,25 | 0,706401 | Inferior |

Las tres opciones que pasaron la selección se comprobaron por separado:

| Opción | spellF1 en segunda partición | Cambio en J03WOY | Cambio en J03WQQ |
| --- | ---: | ---: | ---: |
| Conjunto actual | 0,688752 | — | — |
| Especialista de balón muerto | 0,689528 | −0,000979 | +0,002532 |
| Clasificador con transiciones dirigidas | 0,689124 | +0,004555 | −0,003810 |
| Experto a 25 Hz prolongado | 0,686414 | +0,002033 | −0,006708 |

Son cambios demasiado pequeños e inconsistentes para justificar una sustitución.
Las medias favorables de dos variantes no se presentan como una mejora sólida.

## Cómo se entrenaron los correctores de errores

Para cada partición exterior, solo sus cuatro partidos de entrenamiento generan
los ejemplos del corrector. Dos TCN interiores se entrenan con las primeras o
segundas mitades y predicen la mitad opuesta. El corrector ve errores de
predicciones fuera del ajuste de esa mitad, junto a señales geométricas,
probabilidades, entropía y límites. Una puerta HGB estima cuándo intervenir;
otro HGB propone el estado. También se probó un corrector TCN residual con
muestreo concentrado en errores, penalización de residuos y coherencia temporal.

Esto conserva la exclusión de los partidos exteriores en todas las fases.
Tiene una limitación explícita: el modelo interior es un TCN entrenado con
medias partes y el modelo exterior tiene cuatro expertos. La distribución de
errores cambia. La evaluación externa no mostró que estos correctores resolvieran
ese desajuste ni que mejorasen los objetos de posesión.

## Investigación consultada

- [ASRF, código oficial](https://github.com/yiskw713/asrf): separar clasificación
  y detección de límites. Se probó una adaptación propia de agregación por picos,
  no una reproducción exacta de su modelo ni de sus resultados.
- [UVAST, código oficial](https://github.com/boschresearch/UVAST): secuencias de
  segmentos y duraciones. Se consultó; no se ejecutó su implementación.
- [ASM-Loc, código oficial](https://github.com/boheumd/ASM-Loc): compensar acciones
  breves mediante muestreo por segmentos. Se consultó el enfoque; no se atribuyen
  sus métricas ni su implementación a nuestro especialista.
- [PathCRF, código oficial](https://github.com/hyunsungkim-ds/pathcrf): estados de
  posesión y consistencia entre transiciones. La nueva red dirigida y el
  decodificador son implementaciones propias, no PathCRF completo.
- [Ball Radar, código oficial](https://github.com/hyunsungkim-ds/ballradar):
  inferencia de balón y poseedor a partir de jugadores, revisado también en V10.
- [Sportec/DFL, publicación del dataset](https://www.nature.com/articles/s41597-025-04505-y):
  confirma disponibilidad de BallPossession/BallStatus y ausencia de vídeo por
  restricciones de licencia. Es la referencia nativa externa utilizada aquí.

No se comparan accuracy de poseedor, mAP de detección o F1 de eventos con el
spellF1 oficial como si fueran la misma métrica. No se usaron pesos externos
entrenados con nuestros partidos de comprobación.

## Reproducción y conservación

Los protocolos `PROTOCOLO_V17*.json` y `PROTOCOLO_V18.json` a `V22.json` fijan
entrenamiento, candidatos, particiones y criterio de adopción. Los módulos
`possession.v17_errors`, `v17_specialists`, `v17_highrate`, `v17_gates`,
`v18_typed`, `v19_highrate_long`, `v20_duration`, `v21_correction` y `v22_refiner`
ejecutan sus respectivas pruebas. `possession.v22_report` consolida los archivos.

Se conserva el SHA-256 del CSV actual:
`75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e`.
No se abrió el ground truth público del concurso ni J03WR9. No se subieron datos,
pesos o resultados a servicios externos. Todos los procesos de esta ronda han
terminado; no queda un entrenamiento ejecutándose en segundo plano.

Verificación final: 75 pruebas unitarias correctas y validación de formato del
CSV de 199.319 filas. Esa validación de formato no mide la exactitud de etiquetas.
