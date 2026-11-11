# V24: fusión con la "vista del operador de cámara"

Candidato actual: `evaluation/results/v24_camera_fusion.csv`
SHA-256 `aac4079298817bc79b26de485a90103bb170ac1586ff25fbe6e0d60ffef08096`.
Código: `possession/v24_camera_fusion.py`. Resumen de la ejecución: `RESULTADOS_V24.json`.

## Qué problema ataca

Las cuatro redes se entrenaron con tracking de otro proveedor. En sus partidos el
balón parado se detecta casi sin error (F1 vivo/parado 0,993 en limpio); en el
tracking automático del reto no, porque aquí una parada suele aparecer sin balón
o con un balón que alguien lleva al punto de saque y que parece juego. Los fallos
observados son de tres tipos: paradas enteras no detectadas, reanudaciones
adelantadas e inicios de parada retrasados. Cada uno desplaza los límites de las
dos posesiones vecinas, que es justo lo que mide spellF1.

Además, el candidato anterior marcaba como juego el 44 % de los 2.934 frames
puntuables anteriores al saque inicial de la segunda parte, que la documentación
oficial (FAQ.md) declara balón parado.

## La señal nueva

Las dos grabaciones contienen una evidencia que las redes nunca ven: lo que
hicieron los operadores de cámara.

- El realizador de televisión abandona la cámara principal (repetición, primer
  plano; la homografía broadcast deja de ser válida) sobre todo con el balón
  parado.
- Ambas cámaras panean y hacen zoom de forma distinta durante una parada.

La grabación broadcast es continua dentro de cada parte. La alineación automática
por constelaciones de jugadores (`possession.camera_sync`, sin cambios) da un
desplazamiento por parte (−4014 y −26920 frames, la mediana de sus tramos
aceptados), y con él se proyecta **toda** la línea de tiempo broadcast sobre la
táctica, incluidos los tramos de repetición que la alineación geométrica no puede
emparejar. Un frame broadcast pertenece a una sola parte; el arranque de la
segunda, sin metraje, cuenta como "fuera de la cámara principal".

## Método

1. **Profesor**: las probabilidades congeladas del conjunto de cuatro redes,
   decodificadas igual que el candidato anterior.
2. **Atributos solo de cámara** (19): estado de corte broadcast, frames desde y
   hasta el cambio de plano, duración de los tramos vecinos, error de
   reproyección, punto de enfoque y anchura de campo visible de cada cámara
   (homografía inversa del centro de la imagen), velocidad de paneo a 0,5/2/6 s,
   ritmo de zoom y distancia entre los focos de ambas cámaras. No entra ninguna
   posición de jugador o balón ni ninguna salida del profesor.
3. **Alumno**: gradient boosting (scikit-learn, semilla fija) que aprende
   parado / no parado de las decisiones del propio profesor — pseudoetiquetas
   producidas por un algoritmo, sin intervención humana. Se ajusta de forma
   cruzada: cada quinto contiguo de cada parte se predice con un modelo que no lo
   ha visto, de modo que no puede copiar la decisión del profesor en el tramo
   que puntúa.
4. **Fusión** como evidencias independientes sobre vivo/parado:
   `logit P(D) = logit P_profesor(D) + logit P_alumno(D) − logit prior(D)`.
   H y A conservan la proporción del profesor. La confianza del alumno se acota
   a [0,02; 0,98]: un profesor muy seguro no se revierte solo por la cámara.
5. Los frames puntuables anteriores al saque publicado de la segunda parte se
   fijan a D (dato de la organización, no una observación propia).
6. Decodificador de 25 Hz sin cambios.

Todas las constantes se fijaron a priori. Ninguna se ajustó con etiquetas del
partido del reto.

## Resultado

| | Candidato anterior | V24 |
| --- | ---: | ---: |
| Posesiones predichas (partido completo) | 409 | 379 |
| Fracción de balón parado | 43,9 % | 45,0 % |
| D antes del saque de la 2.ª parte | 56,4 % | 100 % |

Cambian 7.872 de los 158.857 frames puntuables. El alumno reproduce al profesor
con AUC 0,911 fuera de su partición y está calibrado frente a él.

**Comprobación del pipeline en el tramo público** (5 minutos, 17 posesiones; no
se usó para elegir ni ajustar nada y su banda de ruido es de varios puntos):

| | macroF1 | spellF1 | F1 vivo/parado |
| --- | ---: | ---: | ---: |
| Candidato anterior | 0,841 | 0,626 | 0,951 |
| V24 | 0,902 | 0,653 | 0,973 |

La dirección es la esperada (recupera una parada entera que antes se perdía y
corrige una reanudación adelantada 167 frames), pero el tramo público es más
fácil que el resto del partido: allí el balón falta en el 12 % de los frames,
frente al 41 % global. No se conoce la puntuación en el partido completo.

## Uso de las etiquetas públicas en esta ronda

Se abrieron para lo que permiten las reglas: comprobar que el pipeline es
correcto y entender qué considera la referencia. Al inspeccionar los errores del
candidato anterior se vio que coincidían con paradas y que los cortes de la
realización coincidían con ellas; esa observación motivó el diseño de atributos.
Ningún parámetro, umbral o peso procede de esas etiquetas, y el alumno se
entrena exclusivamente con pseudoetiquetas del profesor.

## Estabilidad y reproducción

- Cambiar la partición cruzada (3 u 8 tramos por parte en lugar de 5) mantiene
  el 98,7–99,1 % de los frames.
- El CSV es idéntico byte a byte con 3 y 6 hilos.

```bash
OMP_NUM_THREADS=6 .venv/bin/python -m possession.competitive_bundle \
    --work-dir data/work/final_run --output evaluation/results/final_candidate.csv
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v24_camera_fusion \
    --probabilities data/work/final_run/probabilities.npz \
    --alignment data/work/final_run/alignment.json
.venv/bin/python evaluation/validate_submission.py evaluation/results/v24_camera_fusion.csv
```

El primer comando recalcula la sincronización y las probabilidades congeladas;
el segundo tarda unos 12 segundos.

## Lo que se probó y no entró

- **Costes de cambio del decodificador.** Barrido H↔A ∈ {4, 6, 8, 12, 16} y
  cambio con D ∈ {2,5; 4} sobre los tres partidos abiertos de Metrica (nueve
  condiciones, dos referencias). Diferencias ≤ 0,02 y de signo opuesto según la
  referencia: sin evidencia, se conservan 4 y 2,5.
- **Smart Tagging.** En el tramo inspeccionado las fases etiquetan al equipo
  contrario durante posesiones largas y los intervalos de balón parado abarcan
  varias jugadas. No se usa.
- **Detector propio de balón en vídeo.** Viable (el balón se distingue en la
  cámara táctica) pero no cabía en esta ronda; es la siguiente mejora natural.

## Límites

- La independencia entre cámaras y tracking es una aproximación. Una repetición
  emitida mientras el juego ya se ha reanudado empuja hacia D; el profesor solo
  lo resiste cuando está seguro. En el tramo público retrasa 43 frames una
  reanudación que el profesor tenía bien.
- Las paradas que la realización muestra con la cámara principal (saques
  rápidos) siguen dependiendo solo del tracking.
- El alumno hereda los sesgos sistemáticos del profesor; solo corrige sus
  incoherencias respecto a la vista de cámara.
- No mejora la asignación de equipo: los toques sin control que la referencia no
  cuenta como cambio de posesión siguen produciendo cambios espurios.
