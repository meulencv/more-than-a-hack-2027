# Estimación de posesión por trayectorias individuales y dos cámaras

## Objetivo y salida

El sistema produce `frame,label` para los 199.319 fotogramas de la grabación táctica,
a 25 fps. H significa Barcelona con el balón vivo, A Athletic con el balón vivo,
D juego parado y X ausencia de respuesta. La selección de métodos usa spellF1,
el emparejamiento de intervalos de posesión a los cinco umbrales IoU oficiales.

El candidato reproducible es `evaluation/results/final_candidate.csv`, con
configuración y hashes en `docs/FINAL_SELECTION.json`. El estado del trabajo y los
resultados completos están en `docs/RESULTADOS_MODELO_FINAL.md`.

## Datos y separación entre observaciones y referencias

Los inputs del partido del concurso son las posiciones automáticas de jugadores y
balón de las cámaras táctica y broadcast. Se convierte la posición normalizada a
metros usando 105,3 × 68 m. Las velocidades se recalculan desde las posiciones:
no se depende de la unidad del canal de velocidad exportado. Los slots se tratan
como trayectorias automáticas fragmentadas, no como identidades humanas permanentes.

Se usa la validez de la homografía broadcast para excluir planos sin geometría.
La homografía táctica se verifica en la huella de los inputs pero no aporta
atributos al clasificador. El candidato no usa píxeles, narración, Smart Tagging,
un LLM ni etiquetas humanas del partido. La visión se aprovecha a través del
tracking automático suministrado para las dos grabaciones.

Para entrenar se emplean seis partidos del dataset de Bassek, Rein, Weber y Memmert
(2025), *An integrated dataset of spatiotemporal and event data in elite soccer*,
[DOI](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0. En sus XML nativos,
BallStatus 0 se convierte a D; BallStatus 1 con BallPossession 1/2 se convierte a
H/A; el resto se ignora. La propiedad mientras el balón está parado se descarta.

Los partidos son J03WMX, J03WN1, J03WOH, J03WPY, J03WOY y J03WQQ. J03WR9, usado en
una confirmación anterior, permaneció cerrado en estas rondas. Las referencias
nativas y los eventos externos nunca entran como atributos. No se emplean eventos,
tracking corregido ni otras anotaciones humanas sobre el Barcelona–Athletic.

## Sincronización de cámaras

Las grabaciones tienen cronologías distintas. Se comparan constelaciones de
jugadores por equipo, sin usar el balón, a un frame por segundo:

1. Cuantiles de posición permiten recuperar candidatos por vecinos próximos.
2. Una distancia Chamfer recortada por equipo verifica la geometría sin identidades.
3. Votos de desplazamiento y programación dinámica encuentran tramos consistentes.
4. Se refina el desplazamiento al fotograma y se rechazan tramos que retroceden
   en el tiempo de partido. Se consideran cuatro reflexiones de ejes automáticamente.

Solo se rellena un balón táctico ausente cuando existe una correspondencia aceptada
y el balón aparece en la otra vista. No se sobrescriben las observaciones tácticas
existentes. Los parámetros están fijados en `PROTOCOLO_V11.json` y la alineación
puede regenerarse sin cachés o tablas manuales.

En el reto se aceptan 80.443 correspondencias y se añaden 4.466 observaciones. La
discrepancia mediana de los balones vistos por ambas cámaras, no usados para
sincronizar, es 0,747 m. Esto comprueba coherencia de observaciones; no es una
evaluación de posesión ni una afirmación de corrección en todos los frames.

## Modelo de cuatro componentes

Se promedian con peso 1/4 las probabilidades de cuatro redes:

- Una CNN temporal sobre 123 atributos de posición, movimiento y contexto.
- La misma familia con 33 atributos adicionales de control físico: proximidad,
  velocidad relativa, aproximación, impulsos y giro del balón.
- Dos redes de conjuntos y trayectorias, entrenadas con semillas 2027 y 2028.

En las dos últimas redes, cada jugador tiene 14 observaciones de posición,
velocidad, relación con el balón y máscaras de disponibilidad. Un encoder compartido
procesa su trayectoria corta antes de agregar cada equipo mediante media, máximo
y atención. Una CNN temporal de siete dilataciones combina ese contexto con un
token del balón. La normalización es local por instante. No existe un orden semántico
de slots que el modelo deba memorizar; se comprueba invariancia al permutarlos.

Los modelos reciben posiciones ausentes como desconocidas, con máscaras explícitas.
Ni perder el balón ni perder un anclaje fuerza D. La memoria temporal permite
mantener control durante pases y oclusiones, pero no se impone un cambio por un
simple contacto cercano. Las diferencias exactas entre proveedores siguen siendo
una limitación de la transferencia.

Las ideas se inspiraron en los encoders por objeto de
[Borghesi et al. (2023)](https://iris.polito.it/handle/11583/2976470),
[Ball Radar](https://github.com/hyunsungkim-ds/ballradar) y
[PathCRF](https://github.com/hyunsungkim-ds/pathcrf). La implementación es propia;
no usa sus checkpoints ni datos propietarios y no reproduce sus arquitecturas
completas. La revisión y los resultados negativos están documentados aparte.

## Entrenamiento y decodificación

Las redes de atributos se entrenan 800 actualizaciones; las de trayectorias, 3.000
con descenso coseno de la tasa de aprendizaje. Se usan ventanas, intercambio de
equipos y tres condiciones de observación, incluidas oclusiones y cambios de slots.
Las alteraciones de sensor no consultan las etiquetas para generar el ruido.

La pérdida combina clasificación, ponderación de segmentos, suavidad dentro de
estados reales de entrenamiento y una cabeza de fronteras. Los normalizadores se
ajustan exclusivamente con los partidos de entrenamiento de cada partición.

Las probabilidades a 5 Hz se interpolan linealmente a 25 Hz dentro de cada parte.
El coste por frame se divide por cinco, conservando la escala temporal anterior.
Viterbi usa costes H↔A de 4 y cambios con D de 2,5. La cabeza de fronteras del primer
experto reduce esos costes cuando hay evidencia de cambio. No se buscó otra tabla
de costes ni se impuso una duración mínima manual. Las partes nunca se conectan.

## Evaluación y límites

En tres particiones, cada una entrena con cuatro partidos y deja dos completamente
fuera. El método pasa de 0,6421 a 0,6758 de spellF1 medio respecto a la arquitectura
anterior reentrenada con los mismos datos. La media mejora en los seis partidos;
macroF1 pasa de 0,9078 a 0,9220. Son condiciones correlacionadas de seis partidos,
no dieciocho partidos independientes. Se reutilizaron durante el desarrollo y la
elección del decodificador: no se presentan como una confirmación intacta.

Tras fijar el método, los cuatro componentes finales se ajustan con los seis
partidos. Sus pesos no tienen una evaluación independiente en esos mismos partidos.
Una comprobación posterior en tres partidos externos Metrica, no usados para ajustar
esos pesos, mejora la referencia corregida aproximada de 0,6576 a 0,6918; tampoco
equivale al score del concurso ni a ground truth nativo por frame.

La fusión de cámaras se evaluó además con vistas sintéticas de otros partidos,
introduciendo cortes y replays. Sus pérdidas de balón pueden ser menos correlacionadas
que las reales y sobreestimar el beneficio. La sincronización puede fallar con
formaciones parecidas, pocas detecciones o cambios de plano. Los balones en vuelo
y las proyecciones erróneas siguen siendo casos difíciles.

No se abrió ni puntuó el ground truth público o privado del reto en estas rondas.
No se conocen el spellF1 del nuevo candidato sobre el partido completo ni su posición
en la clasificación. Las comprobaciones públicas anteriores son históricas.

## Reproducción

Con el paquete local de código y pesos, las dependencias fijadas y los datos del
concurso en `data/tactical` y `data/broadcast`:

```bash
OMP_NUM_THREADS=6 .venv/bin/python -m possession.competitive_bundle
.venv/bin/python evaluation/validate_submission.py evaluation/results/final_candidate.csv
```

La ejecución predeterminada recalcula la sincronización y comprueba hashes de inputs
y pesos. El archivo validado tiene SHA-256
`75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e`.
CPU y MPS generaron el mismo CSV en esta máquina. Se entregan los checkpoints
congelados: no se presupone igualdad de entrenamiento entre plataformas distintas.

Los vídeos, tracking, cachés, pesos y salidas permanecen fuera de Git. El paquete
local contiene código y pesos entrenados en los otros partidos, y excluye datos
del concurso, alineaciones precalculadas y cualquier ground truth.
