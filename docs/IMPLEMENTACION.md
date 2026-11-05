> Documento histórico. Para el candidato actual, ver [ESTADO_ACTUAL.md](ESTADO_ACTUAL.md) y [GATED_CANDIDATE.md](GATED_CANDIDATE.md).

# Estado de la solución — 5 de noviembre de 2026

Hay tres estimadores implementados. Todos producen un CSV válido con 199.319 filas que pasa estrictamente `evaluation/validate_submission.py`.
No se han publicado datos, código ni entregas a servicios externos.

## Versiones

### Temporal v1 (Heurística base)
Distancias en metros, velocidad del balón, movimiento colectivo y Smart Tagging como evidencia blanda. Decodificador Viterbi que mantiene continuidad y duración mínima. Cada parte se procesa por separado; fotogramas fuera de los límites oficiales reciben `X`. No usa entrenamiento supervisado.
Comprobación única de 5 minutos: spellF1 **0,2235**, macroF1 **0,5255** (la referencia pública `smooth50` obtiene 0,3256; el 0,2433 publicado corresponde al partido completo).
Se conserva congelado en `evaluation/results/temporal_v1.csv`.

### Temporal externa v2 (Detector binario en juego / parado)
Sustituyó la heurística de balón parado por un clasificador de árboles entrenado con el Partido 1 externo. Su salida pasa el validador oficial.
Se conserva congelado en `evaluation/results/temporal_external_v2.csv`.
Diagnóstico (`transfer_audit.json`): brecha de representación entre datasets (inconsistencia en slots vs jugadores reales, distancia balón-jugador no calibrada a ruido del reto).

### Temporal externa v3 (Candidato final defendido)
Corrige íntegramente las inconsistencias de representación y transferencia:
1. **Representación común unificada:** Metros reales, recuento absoluto de jugadores visibles (frente a 22), velocidad recalculada a partir de posiciones a lag constante (0,2 s), dirección de ataque detectada automáticamente.
2. **Modelo de degradación sintética realista:** Calibrado con estadísticas no supervisadas del partido del concurso (longitudes de rachas de desaparición de balón, ruido autocorrelacionado $\sigma \approx 1,0$ m).
3. **Referencias externas completas:** Incorpora los 3 partidos de muestra de Metrica Sports (Games 1, 2 y 3, este último en formato EPTS FIFA igual al del reto), derivando referencias débiles de posesión basadas en eventos (excluyendo toques sin control y duelos aéreos).
4. **Predicción simétrica:** Invarianza de equipo promediando predicciones directas y permutadas.
5. **Decodificador temporal tipificado:** HSMM Viterbi con duración mínima obligatoria de 1,0 s (5 pasos a 5 Hz) y penalización asimétrica de pérdidas de posesión en juego abierto frente a transiciones a balón parado.

## Validación Externa y Benchmark

Validación cruzada dejando un partido fuera (Leave-One-Game-Out CV) sobre los partidos externos bajo degradación de ruido y oclusiones:

| Partido | Modelo | spellF1 | macroF1 | alive_dead_f1 | Posesiones Pred (GT) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Game 1 | Baseline Smooth 50 | 0,1890 | 0,5591 | 0,6953 | 531 (254) |
| **Game 1** | **v3 (CV)** | **0,6607** | **0,8667** | **0,9320** | **237 (254)** |
| Game 2 | Baseline Smooth 50 | 0,2129 | 0,6335 | 0,7242 | 541 (237) |
| **Game 2** | **v3 (CV)** | **0,6426** | **0,8767** | **0,9336** | **233 (237)** |
| Game 3 | Baseline Smooth 50 | 0,2169 | 0,5854 | 0,7188 | 638 (378) |
| **Game 3** | **v3 (CV)** | **0,5694** | **0,8560** | **0,9464** | **257 (378)** |
| **Promedio CV** | Baseline Smooth 50 | 0,2063 | 0,5927 | 0,7128 | 570 (290) |
| **Promedio CV** | **v3 Propuesta** | **0,6242** | **0,8665** | **0,9373** | **242 (290)** |

Resultados completos archivados en `data/work/external_benchmark.json`.

## Candidato Final (`temporal_v3.csv`)

Generado por `possession.run_v3`:
* Exactamente 199.319 filas, ordenadas 0..199318 (`H`: 49.545, `A`: 29.690, `D`: 79.622, `X`: 40.462).
* 250 posesiones de media 12,7 s.
* SHA-256 del CSV: `084621c9a22c192700c9dff508f4ff68cb24646b9d4a705184f7eb14c9d9781a`
* SHA-256 del modelo: `fa65d00760b1a0baedc3b62b5e13e09cf4f8037d19bf279c4fce2d21240e31ea`
* Reproducción exacta comprobada bit a bit re-ejecutando desde cero.

## Reproducción Rápida

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
.venv/bin/python scripts/fetch_external.py
.venv/bin/python -m unittest discover -s tests -v
OMP_NUM_THREADS=8 .venv/bin/python -m possession.benchmark_external
OMP_NUM_THREADS=8 .venv/bin/python -m possession.run_v3
.venv/bin/python evaluation/validate_submission.py evaluation/results/temporal_v3.csv
```
