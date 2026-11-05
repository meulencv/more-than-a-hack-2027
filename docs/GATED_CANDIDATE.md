# Candidato congelado: mezcla según calidad de observación

## Resultado y criterio de elección

Se conserva el HGB mixto cuando el balón está observado y se da más peso a una
red temporal especializada en oclusiones durante las ventanas donde falta.
El pesoTCN es exactamente la fracción de fotogramas con balón ausente en una
ventana centrada de125fotogramas (5s), calculada por separado dentro de cada parte.
No recibe etiquetas, eventos humanos, equipos ni números absolutos de fotograma.
Las probabilidades se normalizan y se decodifican con la configuración original
v3 (`config_external_selected.json`), sin segunda pasada. No se hizo búsqueda de
ventanas ni ajuste con las respuestas del partido del concurso.

## Experimentos externos anteriores (seis condiciones)

Tres partidos Metrica, cada uno con dos semillas de degradación, dejando fuera
el partido evaluado durante el entrenamiento. Las referencias se derivan de
eventos y son débiles; se muestran tanto la versión antigua como una alternativa
que corrige un error de autorreferencia en la detección de toques.

| Referencia y método | spellF1 medio |
| --- | ---: |
| Referencia antigua /HGB v3 | 0,62502 |
| Referencia antigua /TCN | 0,64790 |
| Referencia antigua /mezcla fija50/50 | 0,65185 |
| Referencia corregida /HGB v3 | 0,55668 |
| Referencia corregida /TCN | 0,56902 |
| Referencia corregida /mezcla fija50/50 | 0,57432 |

Esas seis condiciones NO son seis partidos independientes y se han utilizado para
comparar métodos. No son un test final intacto ni equivalen a etiquetas de posesión
perfectas. Resultados por partido/semilla en `data/work/sequence_tcn/`.

## Nueva comprobación con anotaciones nativas de otro dominio

Fuente: [Sportec/DFL public data](https://doi.org/10.6084/m9.figshare.28196177),
CCBY4.0. Las etiquetas nativas BallStatus/BallPossession se almacenan únicamente
como referencias, nunca como variables de entrada. Se entrena con SportecJ03WMX
junto con Metrica1/2/3. **J03WN1 no entra en ningún entrenamiento**, pero ya se
ha inspeccionado como validación de desarrollo para elegir métodos. No se presenta
como test final intacto después de esa selección.

| SportecJ03WN1 | HGBsoloMetrica | HGBmixto | TCNmixta | Mezcla por calidad |
| --- | ---: | ---: | ---: | ---: |
| Limpio: spellF1 | 0,49121 | 0,52597 | 0,34060 | **0,52597** |
| Estrés sin etiquetas: spellF1 | 0,27109 | 0,35250 | 0,44161 | **0,42017** |
| Limpio: macroF1 | — | 0,85974 | 0,70237 | **0,85974** |
| Estrés sin etiquetas: macroF1 | — | 0,63936 | 0,84141 | **0,77217** |

El estrés añade ruido de posiciones, pérdidas de balón en bloques de8s y cambios
de identidad dentro del mismo equipo. No consulta las etiquetas para añadir ruido.
El gate conserva el resultado limpio del HGBmixto y mejora su comportamiento bajo
este estrés, aunque TCNsola es mejor que la mezcla en esa condición concreta.
No demuestra superioridad universal ni garantiza transferencia al partido del reto.
La media de los dos spellF1 es0,47307 frente a0,38115 del HGBsoloMetrica; es una
media de condiciones de un solo partido, no una estimación de leaderboard.

## Entrenamiento y arquitectura

HGB: Metrica1/2/3 con referencias débiles corregidas + SportecJ03WMX con etiquetas
nativas. TCN: Metrica1/2/3 con referencias débiles antiguas + SportecJ03WMX nativo.
La diferencia entre referencias se declara explícitamente, no se oculta.

TCN48canales ×6capas residuales dilatadas, inspirada en segmentación temporal por convoluciones y suavidad ([MS-TCN, autores](https://arxiv.org/abs/1903.01945)); no es una reproducción de su arquitectura multietapa. Entrenamiento fijo500pasos AdamW,
semilla2027. Normalización ajustada solo con los partidos de entrenamiento,
canales de ausencia y pérdida temporal suavizada. GroupNorm usa estadísticas de
la secuencia: inferencia offline por partes completas, sin cruzar el descanso.
La adaptación nativa TCN usa dos semillas de corrupción sin etiquetas y ambos
órdenes de equipo. Las etiquetas desconocidas se excluyen de la pérdida.

## Artefactos congelados y reproducción verificada

CSV: `evaluation/results/sequence_quality_gated.csv`.
SHA256 `793075ee994c33296e138f632116e136b306393702f37408262da87d29837d99`.

HGB: `data/work/domain_adaptation/mixed.joblib`.
SHA256 `fbf76ce5daae3fd4628de331545bb0dd1c7cc2a0affb85276cc20d141aec4c04`.

TCN: `data/work/sequence_native_mix/mixed_model.pt`.
SHA256 `b70f46229ef4188ca7d4972f91874f7b08716d34938ae1320493095f125da745`.

El CSV pasa el validador:199.319filas, H46.570/A27.690/D84.597/X40.462,
251posesiones. Dos ejecuciones con los modelos guardados produjeron exactamente
el mismo CSV, byte a byte. También la inferencia explícita en CPU ha producido
el mismo SHA256 y un nuevo entrenamiento MPS del TCNmixto ha generado pesos
idénticos byte a byte al checkpoint congelado. Los hashes, configuración completa y procedencia están
en `data/work/sequence_quality_gated/run.json`. Los datos/pesos/derivados quedan
fuera deGit. La selección y congelación se hicieron antes de la evaluación pública. La comprobación posterior del candidato congelado está en [ESTADO_ACTUAL.md](ESTADO_ACTUAL.md); no se usó para ajustar el modelo.

## Reproducción desde los modelos guardados

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_quality_gate --export-only --device auto
# Para inferencia sin GPUApple: --device cpu (CSV idéntico comprobado)
.venv/bin/python evaluation/validate_submission.py evaluation/results/sequence_quality_gated.csv
```

## Reproducción desde entrenamiento limpio

Con los archivos originales del reto en `data/tactical/`, Python3.12 y dependencias
fijadas (incluido PyTorch2.15.0 en la ejecución comprobada):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
.venv/bin/python scripts/fetch_external.py
.venv/bin/python scripts/fetch_sportec_native.py
# Genera las cachés de variables de Metrica y las particiones externas:
OMP_NUM_THREADS=8 .venv/bin/python scripts/prepare_sequence_cache.py
# Genera el HGB legado necesario como referencia de los experimentos:
OMP_NUM_THREADS=8 .venv/bin/python -m possession.run_v3
# Entrena el HGB con referencias corregidas + único Sportec de entrenamiento:
OMP_NUM_THREADS=8 .venv/bin/python -m possession.domain_adaptation
# Entrena TCN con Metrica + ese mismo Sportec, preservando el segundo como desarrollo:
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_native_mix
# Generación sin consultar referencias del reto:
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_quality_gate --export-only
.venv/bin/python evaluation/validate_submission.py evaluation/results/sequence_quality_gated.csv
```

Las cachés externas no deben reutilizarse si cambia el código de extracción,
las fuentes o las referencias. Para un entrenamiento limpio se utiliza una copia
nueva del código sin `data/work/`, descargando las fuentes verificadas mediante
los manifests y aportando los datos del reto. La reproducción bit a bit probada
para este candidato es desde sus modelos guardados; no se garantiza igualdad de
pesos entre plataformas distintas. El TCNsoloMetrica sí se reprodujo exactamente
al repetir su entrenamiento en este ordenador, pero no se extrapola esa prueba
al entrenamiento en una plataforma distinta. El entrenamiento mixto MPS se
ha repetido en una carpeta independiente y el checkpoint resultante sí coincide
exactamente con el congelado. No se ha probado entrenamiento CPU=MPS.

Comando de comprobación del nuevo entrenamiento sin sobrescribir el congelado:

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m possession.sequence_native_mix \
  --work-dir data/work/sequence_native_mix_retrained --train-only
cmp data/work/sequence_native_mix/mixed_model.pt \
  data/work/sequence_native_mix_retrained/mixed_model.pt
```

Pruebas del gate: pesos finitos dentro de[0,1], suma de probabilidades1,
simetría de equipos, partes aisladas y rechazo de entradas inválidas.

El HGB mixto también se volvió a entrenar desde cero en otra carpeta y produjo exactamente el mismo archivo de modelo (`fbf76ce5…`). La comprobación de inferencia CPU usa los modelos entrenados en MPS/CPU; no demuestra que entrenar la TCN en CPU produzca los mismos pesos que en MPS.
