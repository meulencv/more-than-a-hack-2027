# Estado actual: candidato `v29_final`

El candidato que se entrega es `evaluation/results/v29_final.csv`
(SHA-256 `19cf15870dec4d7ae88a507128e23a8d04dec3e8d9cc5cc3dbcbf748be10203c`, 199.319 filas).
Un solo comando lo regenera desde el paquete de datos de la organización y los pesos de
`artifacts/`:

```bash
.venv/bin/python scripts/reproduce_submission.py
.venv/bin/python evaluation/validate_submission.py evaluation/results/v29_final.csv
```

Con `--team-name NOMBRE` escribe además `evaluation/results/NOMBRE.csv` con los mismos bytes (la
organización pide el CSV con el nombre del equipo). Entorno, datos, tiempos, hardware y origen de
cada peso: [REPRODUCE.md](../REPRODUCE.md). Método y resultados:
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## Qué es

1. Tracking automático, con el balón que falta completado desde el vídeo táctico por un detector
   entrenado sin etiquetas humanas (V26b, sin filtro de vuelo).
2. Conjunto congelado de cuatro redes entrenadas con seis partidos abiertos DFL/Sportec y sus
   etiquetas nativas de posesión.
3. Fusión sin etiquetas con el comportamiento de los operadores de las dos cámaras (V24) y el
   saque inicial publicado de la segunda parte.
4. Decodificador de Viterbi a 25 Hz.
5. Verificador de posesiones cortas (variables de V27, gradient boosting) ajustado solo con los
   tres partidos abiertos de Metrica. Objetivo con prior equiprobable: 0,5 × ganancia con la
   referencia «consistente» + 0,5 × ganancia con la «alternativa»; se elimina una posesión si la
   puntuación llega a 0.

En el partido del reto el verificador elimina 65 posesiones cortas (18 absorbidas entre dos
posesiones del rival, 44 pasan a balón parado y 3 al equipo anterior): de 373 posesiones quedan
290 (media 11,8 s, mediana 7,4 s, 5,1 por minuto de juego, 19 % de menos de 3 s, ninguna de menos
de 1 s) y cambia el 2,1 % de los fotogramas evaluados.

## El prior del verificador y el tramo público

Ni el comando ni los entrenamientos leen etiquetas del partido del reto, públicas o privadas. El
tramo público de 5 minutos sí se usó durante el desarrollo, y la sección 6 de
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) dice exactamente cómo:

- **Puntuado.** Se puntuaron candidatos con el evaluador oficial varias veces para comprobar el
  pipeline; una de ellas, el 16 de noviembre, con todos los candidatos que existían entonces. El
  fichero que se entrega (prior 0,5) no existía en ese momento y no se ha puntuado en el tramo
  público.
- **Inspeccionado.** Se compararon los tramos de etiquetas del tramo público con las predicciones
  del candidato de entonces (desde la ronda V24, véase
  [V24_FUSION_CAMARAS.md](V24_FUSION_CAMARAS.md)). De ahí salieron las variables de cámara, dejar
  apagado el componente del silbato y la idea de un verificador de posesiones.
- **Lo que no depende de él.** Ningún modelo, umbral, peso ni prior se ajustó con el tramo público
  ni se dedujo de él. El verificador se ajusta solo con los partidos abiertos de Metrica y su
  prior entre las dos lecturas es 0,5.
- **Historia.** Hasta la auditoría final del 16 de noviembre el candidato usaba un prior de 0,7
  para la lectura consistente. Ese 0,7 venía de comparar ventanas de los partidos abiertos con el
  resumen publicado del tramo (17 posesiones de 14,2 s de media) y con la duración de sus dos
  posesiones más cortas, conocida por la inspección. `RULES.md` dice que el tramo público no es
  un conjunto de ajuste; la auditoría señaló esa dependencia y por eso la entrega pasó al prior
  equiprobable. Las puntuaciones de las otras variantes en el tramo ya se conocían al hacer el
  cambio y el cambio no las sigue: los tres ficheros que más puntúan allí (0,709) no se entregan. Un
  hecho va en el otro sentido y se dice para que conste: el prior 0,5 elimina menos posesiones
  que el 0,7 (65 frente a 88), lo que acerca el fichero al pipeline sin verificador, la dirección
  en la que los ficheros puntuados salen mejor en ese tramo. El candidato anterior sigue
  disponible como `--variant prior70` (`evaluation/results/v29_prior70.csv`, SHA-256
  `ed3e87d911e3070662b96550d3759a0801422790622c7a252998ab878dfaea7a`).

## Por qué esta variante

En los tres partidos abiertos de Metrica, dejando un partido fuera cada vez, el cambio de spellF1
oficial respecto al decodificador congelado (0,645 / 0,659 / 0,650 con la referencia consistente,
0,708 / 0,694 / 0,673 con la alternativa) es:

| Lectura de la referencia | `final` (0,5) | `prior70` (0,7) | Solo consistente (1,0) | `blend_hedge` |
|---|---|---|---|---|
| Las intercepciones de un toque no cuentan (consistente) | +0,032 / +0,061 / +0,060 | +0,063 / +0,073 / +0,084 | +0,059 / +0,083 / +0,079 | +0,085 / +0,095 / +0,086 |
| Mezcla q = 0,5 | +0,023 / +0,039 / +0,034 | +0,015 / +0,026 / +0,041 | −0,007 / +0,014 / +0,023 | +0,001 / +0,025 / +0,020 |
| Todas cuentan (alternativa) | +0,004 / +0,015 / +0,001 | −0,042 / −0,021 / −0,014 | −0,083 / −0,055 / −0,047 | −0,082 / −0,047 / −0,055 |

`final` no pierde en ningún partido con ninguna de las tres lecturas (por condición sí hay dos
valores negativos de nueve con la lectura alternativa: −0,012 y −0,004). A cambio renuncia a
parte de la ganancia si la lectura correcta es la consistente (+0,051 de media frente a +0,073 de
`prior70`). 0,5 es el prior de indiferencia entre las dos lecturas de la definición escrita y no
usa información del partido del reto. Se mantiene el conjunto congelado y no la mezcla con las
redes reentrenadas porque la mezcla pierde entre 0,047 y 0,090 con la lectura alternativa, donde
`final` no pierde, su regla de reetiquetado se calibró con los mismos partidos de Metrica y sus redes
nunca se validaron con un balón completado desde vídeo. Las cifras están en
`docs/evidence/hedge.json` (copia de `data/work/v29/hedge.json`; fila base `frozen`, familia
`v27`, objetivo `w0.50`). La puntuación en el partido del reto es desconocida.

Las variantes `prior70`, `blend_hedge`, `blend_bet` y `safe` siguen disponibles con `--variant`,
pero no son la entrega. En el código de estudio (`possession/v29_final.py`,
`possession/v29_lab.py`) y en `data/work/v29` los candidatos se llaman `hedge_frozen`
(= `prior70`), `hedge`, `bet` y `safe`; `evaluation/results/v29_hedge_frozen.csv` tiene los mismos
bytes que `v29_prior70.csv`. La variante `final` corresponde al objetivo `w0.50` de
`possession.v29_lab` sobre la base `frozen`; la ejecución de estudio no escribió un CSV para ella.

## Estado anterior (V24)

Hasta esta ronda el candidato era `evaluation/results/v24_camera_fusion.csv`
(SHA-256 `aac4079298817bc79b26de485a90103bb170ac1586ff25fbe6e0d60ffef08096`): el mismo conjunto
congelado y la misma fusión de cámaras, sin el balón del vídeo ni el verificador. Sigue
documentado en [V24_FUSION_CAMARAS.md](V24_FUSION_CAMARAS.md); el conjunto congelado
(`final_candidate.csv`), en [RESULTADOS_MODELO_FINAL.md](RESULTADOS_MODELO_FINAL.md) y
[MEMORIA_TECNICA.md](MEMORIA_TECNICA.md); las rondas V17–V23, en
[EXPERIMENTOS_V17_V22.md](EXPERIMENTOS_V17_V22.md) y [TOQUES_CORTOS_V23.md](TOQUES_CORTOS_V23.md).
Los CSV y pesos de las rondas anteriores se conservan.
