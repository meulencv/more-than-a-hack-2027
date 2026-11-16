# V30: corrección del sesgo temporal de los límites (16/11/2026)

Estudio hecho solo con los tres partidos abiertos de Metrica y los pesos congelados de
`artifacts/frozen`. No se ha leído ningún dato ni etiqueta del partido del reto, y nada de esto
se ha ejecutado sobre él: **el candidato sigue siendo `v29_final`**.

## Qué se encontró

Los límites que decodifica el conjunto congelado están desplazados respecto a las referencias
de eventos, en el mismo sentido en los tres partidos y las tres condiciones de sensor
(`python -m possession.v30_boundaries offsets`, posesiones emparejadas a IoU ≥ 0,3):

| Límite | n | Cuartiles del desfase (fotogramas) |
|---|---:|---|
| posesión → balón parado | 375 | +6 / +17 / +30 (empieza tarde) |
| posesión → posesión | 2495 | −1 / +5 / +13 (cambia tarde) |
| balón parado → posesión | 635 | −10 / −4 / +1 (empieza pronto) |

El desfase no depende de la lectura de las intercepciones de un toque: las dos referencias
comparten esos límites. Si todos los emparejamientos a IoU 0,3 lo fueran también a 0,7, el
spellF1 subiría entre 0,07 y 0,12 con cualquiera de las dos lecturas; una constante por tipo
recupera una parte pequeña de eso.

## Resultado

`possession.v30_boundaries.shift` mueve cada límite una constante según su tipo, después del
verificador. Dejando un partido fuera (el verificador y las constantes se ajustan con los otros
dos), spellF1 por partido, media de las tres condiciones
(`python -m possession.v30_boundaries logo`):

| Variante | Consistente | Alternativa | Media |
|---|---|---|---:|
| Decodificador congelado | 0,657 / 0,667 / 0,647 | 0,716 / 0,703 / 0,681 | 0,678 |
| Verificador 0,5 (V29 `final`) | 0,691 / 0,737 / 0,703 | 0,709 / 0,712 / 0,683 | 0,706 |
| + desplazamiento, constantes elegidas sin el partido | 0,691 / 0,747 / 0,715 | 0,711 / 0,722 / 0,693 | **0,713** |
| + desplazamiento solo posesión → posesión (−4) | 0,692 / 0,740 / 0,709 | 0,711 / 0,715 / 0,688 | 0,709 |

Constantes elegidas en cada exclusión (→parado, parado→, equipo→equipo): (−16, +4, −4),
(−12, +4, −4), (−12, 0, −4). El macroF1 sube de 0,885 a 0,889.

**Cómo leerlo.** +0,007 de media sobre el verificador, sin pérdida en ningún partido ni
lectura, pero en el partido 1 la ganancia es prácticamente nula y todo queda muy por debajo
del ruido de 0,04 que `EVALUATION.md` atribuye al spellF1.

**Salvedades.**

- La condición `moderate` de estas cifras usa un perfil de huecos de balón *aproximado*: el
  perfil real sale del tracking del reto, que no estaba en la máquina del estudio. `clean` y
  `stress` reproducen `docs/evidence/measure.json` con cuatro decimales. Hay que repetir
  `logo` con los `data/work/v16` originales.
- Junto a un balón parado los partidos de Metrica no tienen balón en el tracking, y en el
  partido del reto P(D) se fusiona con la evidencia de cámara, que aquí no existe. Las dos
  constantes de balón parado pueden no transferirse; la de equipo → equipo (`TT_ONLY`)
  depende solo de las redes y del decodificador.

## Probado y descartado en esta ronda

| Idea | Resultado (ganancia esperada, dejando un partido fuera) |
|---|---|
| Promedio de fases de muestreo (5 rejillas de 5 Hz desplazadas, intercaladas o promediadas) | −0,000 / +0,001 con los costes actuales: sin efecto |
| Refinador de límites aprendido (GBM, curvas de probabilidad ±2 s) | +0,015 solo; tras el verificador +0,008, igual que las tres constantes |
| El mismo con distancias balón-jugador | +0,023, pero el extra viene solo de los límites con balón parado (donde Metrica no tiene balón); restringido a equipo → equipo, +0,016. Descartado por el mismo motivo que `live_only` en V28a |
| Variables de estabilidad en el verificador (persistencia al subir costes, acuerdo entre fases, forma de la curva) | +0,027 → +0,027/+0,028: sin efecto |

Techo del verificador con sus candidatos actuales (oráculo con el objetivo 0,5/0,5): +0,076
frente a +0,027 conseguido. Las variables derivadas de las salidas de las redes no acercan esa
cifra; haría falta información nueva.
