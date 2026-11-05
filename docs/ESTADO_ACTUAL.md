> Actualización V7–V9: nueve entrenamientos y 37 comparaciones adicionales. La transferencia, las pérdidas por intervalos y la memoria de control no aportan una ventaja estable frente a V6. No se han usado nuevas comprobaciones públicas ni se han sustituido los CSV congelados. Ver [EXPERIMENTOS_V7_V9.md](EXPERIMENTOS_V7_V9.md).

> Actualización V6: hay una familia nueva con mejoras externas, pero su comprobación pública da 0,6595 frente a 0,7067 del candidato conservado. No se presenta como mejora demostrada del concurso. Ver [EXPERIMENTOS_V6.md](EXPERIMENTOS_V6.md). Ambos CSV se preservan.

# Estado actual: candidato por calidad de observación

El candidato actual es `evaluation/results/sequence_quality_gated.csv`, congelado
antes de la comprobación pública en el commit local `8402a96`. No se ha publicado
ni enviado nada a la organización. Método y reproducción: [GATED_CANDIDATE.md](GATED_CANDIDATE.md).

## Comprobación pública única después de congelar

| Métrica, mismos cinco minutos | Candidato anterior conservado | Nuevo candidato |
| --- | ---: | ---: |
| spellF1, principal | 0,6581 | **0,7067** |
| macroF1 | 0,8475 | **0,8768** |
| Accuracy | 86,23 % | **88,88 %** |

Solo son 7.500 fotogramas y 17 posesiones reales; el nuevo sistema predice 13.
No se ha entrenado ni ajustado con esta comprobación. No se probaron todos los
CSV contra el tramo público para elegir el mejor. No se extrapola el resultado
al partido completo ni se garantiza una posición en la clasificación.
Los resultados completos están en `data/work/frozen_gate_public/` y su resumen
verificable en [COMPROBACION_PUBLICA_FINAL.json](COMPROBACION_PUBLICA_FINAL.json).

## Evidencia externa y limitaciones

- TCN temporal y mezcla fija mejoran el desarrollo de Metrica sobre seis casos
  (tres partidos × dos simulaciones), también al revisar sus referencias débiles.
- Se encontró y documentó un error en la detección de toques de las referencias
  antiguas. Se conservan para comparaciones históricas; HGB nuevo usa la versión
  corregida. Cambiar una referencia cambia la puntuación aunque el modelo no cambie.
- Se añadieron dos partidos Bundesliga con etiquetas nativas de posesión y balón
  vivo/parado, adquiridos con licencia CC BY 4.0 y fuentes verificadas. Uno se usa
  para entrenar; el otro está excluido del ajuste de pesos, pero se inspeccionó
  como desarrollo para seleccionar la estrategia. No es ya un test intacto.
- En el segundo partido nativo, spellF1 pasa de 0,4912 a 0,5260 en limpio y de
  0,2711 a 0,4202 con oclusiones, ruido y cambios de identidad. Son condiciones
  de un solo partido, no partidos independientes ni una estimación del leaderboard.
- El clasificador HGB mixto conserva el caso limpio; la TCN mixta aporta durante
  oclusiones. El peso lo determina la fracción de balón ausente en cinco segundos
  dentro de la misma parte, sin referencias como entrada.
- CatBoost, dos clasificadores jerárquicos y bosques aleatorizados se probaron y
  no justificaron su inclusión. No se mezclan modelos que empeoran la evidencia.

Investigación y fuentes: [INVESTIGACION_METODOS.md](INVESTIGACION_METODOS.md).
Auditoría de referencias: [AUDIT_REFERENCIAS_Y_ROBUSTEZ.md](AUDIT_REFERENCIAS_Y_ROBUSTEZ.md).
El vídeo broadcast, los comentarios humanos y el ground truth privado no se usan.
