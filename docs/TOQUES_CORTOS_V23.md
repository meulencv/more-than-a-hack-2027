# Toques breves y márgenes de confirmación

Se conserva el candidato actual. La revisión V23 probó **16 alternativas
temporales**; ninguna mejora el spellF1 de selección. La prueba no entrenó pesos
nuevos ni utilizó el ground truth del concurso. El estado previo estaba guardado
en `f00057c`; el primer prototipo V23 también quedó guardado en `6bc8003`.

## Qué significa un toque breve

Según [las reglas oficiales](../CHALLENGE.md), el cambio exige control del rival.
Una desviación sin control conserva el equipo anterior. No existe una duración
mínima oficial: un control auténtico de 0,2 s puede separar tres posesiones en
H-A-H. H-D-H también son dos posesiones, aunque la parada sea breve.

Un margen del algoritmo es tiempo para reunir evidencia, no una autorización
para eliminar cualquier posesión menor que ese margen. Tampoco cambia la
evaluación oficial, que sigue emparejando intervalos por IoU 0,3–0,7.

## Diagnóstico reproducido

En J03WMX y J03WN1, excluidos del ajuste de sus respectivas redes, se cuentan
interrupciones H-A-H/A-H-A en las etiquetas externas y en las predicciones.
El emparejamiento es global, uno a uno, mediante el código oficial a IoU 0,5.

| Condición | Reales ≤0,8 s | Predichas ≤0,8 s | Reales ≤2 s | Predichas ≤2 s | Predichas ≤2 s emparejadas con una posesión real |
| --- | ---: | ---: | ---: | ---: | ---: |
| Limpio | 74 | 1 | 220 | 58 | 43 |
| Ruido moderado | 74 | 0 | 220 | 44 | 29 |
| Estrés | 74 | 0 | 220 | 39 | 25 |

Las referencias se repiten entre condiciones; no son ejemplos independientes.
La última columna puede emparejar con una posesión real más larga o con otros
vecinos: al exigir también una interrupción real ≤2 s, los recuentos son 39, 25
y 22. No son métricas de toques físicos ni del partido del concurso.

El modelo ya suprime muchas interrupciones breves. Una regla que borre todas las
rachas inferiores a un segundo no resuelve ese fallo y puede destruir aciertos.

## Políticas implementadas

1. **Confirmación retrospectiva:** esperar 0,12 / 0,24 / 0,40 / 0,80 s con evidencia
   del equipo rival. Al confirmar, asignar el cambio desde el primer fotograma
   de esa evidencia, sin desplazar el límite al momento de confirmación.
2. **Margen adaptativo:** 0,12 s con apoyo de control físico, margen normal con
   balón observado y 0,80 s durante ausencia de balón. El apoyo físico combina
   proximidad y velocidad relativa, y sigue siendo un proxy de tracking 2D.
3. **Excepciones para evidencia fuerte:** probabilidad ≥0,95 permite confirmación
   inmediata. Se probaron también dos variantes con probabilidad ≥0,80 más
   control físico ≥0,80 y ventaja de control ≥0,50. No son pruebas de control real.
4. **Fusión débil:** solo H-A-H breve con evidencia débil de A, observación
   suficiente y sin apoyo físico. La debilidad se comprueba también condicionada
   a H/A para no confundir incertidumbre de balón muerto con equipo incierto.
5. **Recuperación de pulsos:** buscar evidencia del rival dentro de una posesión
   larga, entre anclas del mismo equipo, con probabilidad y apoyo físico/límites.
   Sus ventanas máximas son 0,4 / 0,8 / 1,2 s. Esta propuesta exige tres fotogramas
   de apoyo: no puede recuperar controles de uno o dos fotogramas.
6. **Fusión solo por duración:** umbrales 0,2 / 0,4 / 0,8 s como control diagnóstico,
   excluido de la selección de candidato porque ignora la evidencia de control.

Todas las políticas conservan exactamente D/X y separan mitades. La ausencia de
balón por sí sola no cambia el estado a D. La fusión rechaza conflictos cuando
una de sus anclas también sería candidata a desaparecer; evita invertir segmentos
en cadenas H-A-H-A-H.

El primer prototipo se conserva en `data/work/v23/` y en el commit indicado.
Después de revisarlo se registró otra ronda antes de ejecutar las correcciones
de conflictos y la excepción física conjunta. Su código lleva un hash en
[PROTOCOLO_V23_REVIEWED.json](PROTOCOLO_V23_REVIEWED.json). Los resultados
revisados están en `data/work/v23/reviewed/results.json`, con resumen versionado
en [RESULTADOS_V23.json](RESULTADOS_V23.json).

## Resultado y decisión

Selección: J03WOH/J03WPY, tres condiciones por partido, mismas probabilidades
fuera de entrenamiento para todas las políticas. Se elige por spellF1 oficial,
con macroF1 no inferior en más de 0,002. El ganador debe después mejorar cada
partido de la segunda partición J03WOY/J03WQQ. Como ningún candidato supera al
original en selección, no se abre esa segunda partición en esta ronda.

| Política | spellF1 de selección |
| --- | ---: |
| Conjunto actual | **0,709897** |
| Fusiones ≤0,8 s | 0,709897; no modifican ningún fotograma |
| Mejor confirmación fija, 0,24 s | 0,693537 |
| Mejor confirmación adaptativa/conjunta, 0,24 s | 0,692909 |
| Mejor recuperación de pulsos, máximo 0,4 s | 0,708664 |

Las políticas de confirmación introducen otros cambios y cortes además de intentar
recuperar controles omitidos. El balance global es peor; no se adopta el filtro.
La conservación de D es una limitación deliberada: esta prueba no corrige falsos
estados D del sistema base.

El 0,709897 corresponde a estos dos partidos de selección, **no a una mejora del
0,675775 global** del método actual. Los partidos ya participaron en el desarrollo
histórico; son pruebas de desarrollo, no un test independiente del sistema final.

Se corrigió la documentación antigua que atribuía un mínimo de un segundo al
candidato actual. El Viterbi actual no impone ese mínimo duro. Pasan **87 pruebas**,
incluidas confirmación sin retardo, preservación de controles muy breves fuertes,
fusiones solapadas, simetría y conservación de paradas/mitades.

```bash
.venv/bin/python -m possession.v23_touch_margins
.venv/bin/python -m unittest discover -s tests
```

No se modificaron los pesos ni el CSV canónico. Su SHA-256 sigue siendo
`75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e`.
