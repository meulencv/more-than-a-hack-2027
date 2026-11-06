# Estado actual: modelo final de cuatro redes y dos cámaras

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
