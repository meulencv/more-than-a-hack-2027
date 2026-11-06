# Criterio de posesión y revisión por casos

Las definiciones describen el candidato actual. La tabla de diagnóstico y las
comprobaciones antiguas se conservan como evidencia histórica de un candidato
anterior; no representan las métricas del conjunto final. No se consultan etiquetas
del partido del reto. Base oficial: [CHALLENGE.md](../CHALLENGE.md).

## Definiciones

- En juego: H o A. Una disputa o un balón suelto conserva el último equipo con
  **control**, no necesariamente el último equipo que tocó el balón.
- Un pase/vuelo no implica cambio de equipo. Un rebote, una desviación o un duelo
  aéreo sin control posterior tampoco bastan por sí solos.
- Parada: D. No se pregunta a quién corresponde el balón muerto. H-D-H son dos
  posesiones, no una posesión continua. Para un resumen de porcentajes, el denominador
  de posesión viva es H+A; D se informa por separado, no se asigna a un equipo.
- Reinicio: D hasta la reanudación; después H/A según control. El candidato aprende
  estas fronteras, pero no tiene un detector explícito de cada tipo de saque.
- **No existe un mínimo numérico oficial**. El decodificador actual
  (`v14_resolution.decode_fine`) usa Viterbi a 25 Hz con costes de cambio y no
  impone un mínimo duro. El mínimo de 1 s pertenecía a un decodificador anterior.
  Los márgenes de confirmación se han probado separadamente en
  [V23](TOQUES_CORTOS_V23.md), sin adoptarlos por falta de mejora.
- Balón ausente/oculto: no prueba parada ni cambio de equipo. El conjunto actual
  combina cuatro redes con pesos fijos; la segunda cámara aporta observaciones
  donde falta el balón táctico. No garantiza conservar siempre al último controlador. El candidato no usa vídeo ni altura Z: infiere contexto desde
  tracking 2D, por lo que vuelo, rebote y control siguen siendo ambiguos.

## Diagnóstico externo medido

Fuente histórica: partido nativo Sportec J03WN1, excluido del ajuste de aquel
candidato pero ya utilizado como desarrollo. Se reutilizan las probabilidades
congeladas de aquel candidato, no del conjunto actual.
Las máscaras se superponen y no son muestras independientes. Las cifras son
**acierto por fotograma dentro de cada caso**, no spellF1 ni accuracy del concurso.
No se calcula spellF1 por máscara porque se inventarían fronteras al quitar frames.

| Caso | Limpio | Con ruido, oclusiones e intercambios de ID |
| --- | ---: | ---: |
| Balón en juego y observado | 76.7 % (68377 frames) | 86.5 % (43564 frames) |
| Balón en juego y ausente | sin muestras | 73.6 % (24813 frames) |
| Balón parado pero observado | 99.8 % (73184 frames) | 63.3 % (44998 frames) |
| Hasta 2 s de un reinicio | 81.7 % (10556 frames) | 63.6 % (10556 frames) |
| Hasta 1 s de un cambio de equipo | 60.4 % (17698 frames) | 56.6 % (17698 frames) |
| Geometría compatible con disputa (proxy) | 67.4 % (9404 frames) | 80.5 % (6812 frames) |

El caso limpio no contiene balón ausente. El estrés oculta observaciones sin
consultar las etiquetas; el balón parado visible resulta especialmente delicado
bajo ese ruido. Reinicios y cambios de equipo también muestran mayor dificultad.
La máscara de disputa usa dos equipos a menos de 3 m del balón y margen inferior
a 1 m; no confirma una disputa real. Tampoco se identifica la causa física de una
oclusión (jugador, cámara o fallo del detector). Son hipótesis, no etiquetas humanas.

El diagnóstico no permite afirmar que estos sean los mayores errores del partido
del concurso. Sí aporta prioridades externas: distinguir balón muerto visible de
juego, conservar control durante disputas y localizar el instante de reinicio.
Los resultados detallados y matrices de confusión están en `data/work/case_audit.json`.

## Comprobaciones añadidas

Tres casos sintéticos comprueban que una parada separa posesiones del mismo equipo,
que la ausencia de balón no obliga a D y que un reinicio puede cambiar al equipo
contrario sin arrastrar la propiedad anterior. Ver `tests/test_possession_cases.py`.
Son comprobaciones del mecanismo con evidencia sintética, no una demostración de
que el modelo reconozca correctamente cada situación real.

```bash
.venv/bin/python -m unittest discover -s tests -q
OMP_NUM_THREADS=6 .venv/bin/python -m possession.audit_cases
```
