# Estado actual: cuatro redes, dos cámaras y vista del operador (V24)

El candidato actual es `evaluation/results/v24_camera_fusion.csv`
(SHA-256 `aac4079298817bc79b26de485a90103bb170ac1586ff25fbe6e0d60ffef08096`).
Parte del conjunto congelado de cuatro redes y le añade una evidencia que este no
veía: los cortes de la realización de TV y el movimiento de ambas cámaras, fusionados
mediante un alumno auto-supervisado. Método, resultados y límites:
[V24_FUSION_CAMARAS.md](V24_FUSION_CAMARAS.md).

```bash
OMP_NUM_THREADS=6 .venv/bin/python -m possession.competitive_bundle --work-dir data/work/final_run
OMP_NUM_THREADS=6 .venv/bin/python -m possession.v24_camera_fusion \
    --probabilities data/work/final_run/probabilities.npz --alignment data/work/final_run/alignment.json
.venv/bin/python evaluation/validate_submission.py evaluation/results/v24_camera_fusion.csv
```

En esta ronda sí se abrió el tramo público, solo para comprobar el pipeline y
entender la referencia; no se ajustó nada con él. Lo que sigue describe el
conjunto congelado sobre el que se construye V24.

## Conjunto congelado anterior (`final_candidate.csv`)

El candidato actual es `evaluation/results/final_candidate.csv`.
Configuración: [FINAL_SELECTION.json](FINAL_SELECTION.json).
Resultados y límites: [RESULTADOS_MODELO_FINAL.md](RESULTADOS_MODELO_FINAL.md).
Memoria técnica: [MEMORIA_TECNICA.md](MEMORIA_TECNICA.md).

La comparación por partidos externos da **spellF1 0,6758**, frente a **0,6421** de
la arquitectura anterior reentrenada con los mismos datos por partición. Mejora
la media en los seis partidos. El método final se reentrenó con los seis después
de seleccionarlo; esos checkpoints no tienen un test independiente en los mismos
seis. La segunda cámara se valida por separado con vistas sintéticas y coherencia
geométrica de los inputs reales, sin etiquetas del partido del reto.

El CSV tiene 199.319 filas y SHA-256
`75e884aea21d68e0cc46c3e5a67798e799e4ceb59431e5872638080b38dc192e`.
CPU y MPS producen exactamente el mismo archivo. La reproducción del paquete
sin cachés se registra en [FINAL_ARTIFACT.json](FINAL_ARTIFACT.json).

```bash
OMP_NUM_THREADS=6 .venv/bin/python -m possession.competitive_bundle
.venv/bin/python evaluation/validate_submission.py evaluation/results/final_candidate.csv
```

No se ha puntuado el nuevo candidato contra el tramo público ni se ha usado ese
tramo para seleccionar modelos. Su puntuación en el partido completo y su puesto
en la competición son desconocidos. V5 conserva su comprobación histórica pública
de 0,7067 y V6 la de 0,6595; no son las puntuaciones públicas del modelo nuevo.

Se conservan los CSV y pesos de V5, V6 y V12. Nada se ha publicado o enviado.
Los datos y pesos permanecen fuera de Git; el paquete de código y pesos es local.
Investigación: [INVESTIGACION_V10.md](INVESTIGACION_V10.md).
Las rondas anteriores siguen en sus informes V6 y V7–V9 y en el historial de Git.

La ronda posterior V17–V22 ejecutó 41 alternativas y 17 ajustes/continuaciones.
Ninguna superó el criterio de adopción; el candidato se conserva. Diagnóstico y
resultados negativos: [EXPERIMENTOS_V17_V22.md](EXPERIMENTOS_V17_V22.md).

La prueba V23 de márgenes para controles breves conserva el candidato: ninguna
política temporal mejora la selección externa. Véase
[TOQUES_CORTOS_V23.md](TOQUES_CORTOS_V23.md). El decodificador actual no impone
un mínimo duro de duración; la documentación antigua se corrigió.
