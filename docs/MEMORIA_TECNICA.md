> Documento histórico. Para el candidato actual, ver [ESTADO_ACTUAL.md](ESTADO_ACTUAL.md) y [GATED_CANDIDATE.md](GATED_CANDIDATE.md).

# Memoria Técnica — More Than A Hack 2027

**Reto:** Inferencia de posesión frame a frame (`H`, `A`, `D`, `X`) para el partido FC Barcelona vs Athletic Club (LaLiga 2024-25, Jornada 2).  
**Referencia temporal:** Cámara táctica (199.319 fotogramas, 25 fps, 105,3 × 68,0 m).  
**Métrica de ordenación del concurso:** `spellF1` (recuperación de posesiones continuas mediante emparejamiento temporal 1 a 1 a umbrales IoU 0,3..0,7).  
**Primer desempate:** `macroF1` frame a frame (media no ponderada de F1 para H, A y D sobre fotogramas puntuados).  
**Fecha:** Noviembre 2026.

---

## 1. Resumen Ejecutivo y Enfoque

El reto de estimar posesión a partir de tracking automático presenta una asimetría fundamental documentada en las reglas:
1. **Pérdida severa del balón:** El balón desaparece en el 40,7% de los fotogramas en juego puntuados (158.857 fotogramas) debido a oclusiones, agrupaciones en el área y pases aéreos.
2. **Ruido de detección y fragmentación:** La distancia al jugador más cercano en el tracking automático tiene una mediana de ~1,89 m (frente a <0,3 m en datasets limpios), y asignar la posesión simplemente al jugador más cercano al balón genera miles de micro-fragmentaciones que destruyen `spellF1` (el baseline no suavizado obtiene un spellF1 de apenas 0,0486).
3. **Definición de posesión:** La posesión exige control. Un despeje, un rebote, una disputa aérea o una parada sin control no cambian la posesión del equipo que la ostentaba. Además, el balón parado (`D`) no es posesión de nadie, sino una detención del juego.

### Evolución de versiones:
* **Temporal v1 (Heurística):** Empleó distancias físicas ponderadas por velocidad y un suavizado temporal Viterbi sin entrenamiento supervisado. En la ventana pública de 5 minutos obtuvo `spellF1: 0,2235; macroF1: 0,5255`, por debajo del suelo publicado del baseline suavizado (`baseline_smooth50` = 0,2433).
* **Temporal v2 (Detector binario externo débil):** Entrenó un modelo `HistGradientBoosting` en el partido 1 de muestra de Metrica para distinguir en juego / parado. Presentó una brecha severa de transferencia (KS distance = 1,0 en jugadores visibles y 0,45 en distancias balón-jugador) por no unificar unidades ni degradar adecuadamente los datos de entrenamiento.
* **Temporal v3 (Solución final defendida):**
  - **Unificación rigurosa de espacio y representación:** Conversión de todos los datasets a coordenadas métricas reales, recomputación homogénea de velocidades a partir de posiciones a lag constante (0,2 s) y recuento absoluto de jugadores visibles en lugar de fracción de slots de tracking.
  - **Modelo de degradación sintética calibrado en el partido del concurso sin etiquetas:** Calibración de ruido espacial autocorrelacionado en el tiempo ($\sigma \approx 1,0$ m) y patrones de oclusión con las longitudes de racha reales observadas en el tracking automático del partido.
  - **Supervisión débil robusta en 3 partidos externos:** Generación de referencias aproximadas basadas en eventos en los 3 partidos públicos de Metrica Sports (Partidos 1, 2 y 3).
  - **Inferencia simétrica equipo-invariante:** Posesión evaluada promediando las probabilidades bajo la orientación original y la permutación espejo de equipos, garantizando neutralidad absoluta.
  - **Decodificador temporal HSMM con costes diferenciados por tipo de transición:** Modelo de Markov de duración mínima obligatoria (mínimo 1,0 s = 5 pasos de 5 Hz) con penalización asimétrica: cambiar de posesión en juego abierto ($H \leftrightarrow A$, pérdida de balón) tiene un coste superior ($c=4,0$) al de una interrupción a balón parado ($c=2,5$).

---

## 2. Datos y Fuentes Utilizadas

### 2.1 Datos del reto (Evaluación)
* **Archivo de tracking táctico:** `data/tactical/BAR-ATH_tactical_atd.txt` y metadata XML.
* **Alineación de fotogramas:** 
  - Fotogramas evaluados: 1ª parte (11.185 a 90.651) y 2ª parte (110.808 a 190.197), total 158.857 fotogramas.
  - Fotogramas fuera de juego (antes del inicio, descanso, tras el pitido final): etiquetados estrictamente con `X` (40.462 fotogramas).
* **Datos descartados:**
  - **Broadcast:** El vídeo y tracking broadcast están en una línea temporal distinta, con cortes, repeticiones y homografía incompleta (34,2% no utilizable). La sincronización manual o el warping no aportaban ventajas justificables frente a la continuidad temporal del feed táctico 100% calibrado.
  - **Gemma / Modelos de visión multimodal:** Se evaluó localmente `gemma-4-26b-a4b-it-4bit` con MLX-VLM en secuencias de 4 imágenes tácticas. Aunque generó JSON estructurado válido, predijo "live" en 12/12 secuencias sin discriminar paradas ni recuperaciones sutiles, a un coste de 1,5–4,2 s por secuencia (~17 GB VRAM). No se empleó en la tubería final.
  - **Ground truth público (5 minutos):** Empleado única y exclusivamente como comprobador de formato y sanidad de alineación de la tubería, nunca para optimizar hiperparámetros ni como conjunto de entrenamiento.

### 2.2 Datos externos (Entrenamiento y Validación)
* **Repositorio oficial de Metrica Sports:** `https://github.com/metrica-sports/sample-data`
* **Revisión fijada:** Commit `e706dd506b360d69d9d123d5b8026e7294b13996`
* **Partidos utilizados:**
  1. **Sample Game 1:** Tracking en CSV (Home + Away) y eventos (70.578 fotogramas en 1ª parte, 71.496 en 2ª parte).
  2. **Sample Game 2:** Tracking en CSV (Home + Away) y eventos (67.807 fotogramas en 1ª parte, 72.307 en 2ª parte).
  3. **Sample Game 3:** Formato EPTS estándar de la FIFA (`Sample_Game_3_metadata.xml`, `Sample_Game_3_tracking.txt` y `Sample_Game_3_events.json`), exactamente el mismo formato de entrega del concurso (143.761 fotogramas).
* Todos los hashes SHA-256 de los 9 archivos externos están formalizados en `docs/external_data_manifest.json` y se descargan/verifican mediante `scripts/fetch_external.py`.

---

## 3. Metodología de Validación Externa

Para no violar las reglas del concurso y no ajustar hiperparámetros contra los 5 minutos públicos (lo cual, como demuestra `EVALUATION.md`, sobreajusta y selecciona modelos peores para el partido completo), toda la validación se realizó mediante **Validación Cruzada Dejando Un Partido Fuera (Leave-One-Game-Out CV)** sobre los 3 partidos externos:

### Benchmark comparativo en tracking degradado (simulación del reto):

En cada pliegue se entrenó en 2 partidos externos y se evaluó a ciegas en el partido restante sometido a la degradación de oclusiones y ruido del concurso, computando las métricas oficiales con `evaluation/possession_metrics.py`:

| Partido Evaluado | Modelo | spellF1 | macroF1 | alive_dead_f1 | Posesiones Pred (GT) | Duración media pos |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Game 1** | Baseline Smooth 50 | 0,1890 | 0,5591 | 0,6953 | 531 (254) | 5,5 s |
| **Game 1** | **v3 (entrenado en G2 + G3)** | **0,6607** | **0,8667** | **0,9320** | **237 (254)** | **12,8 s** |
| **Game 2** | Baseline Smooth 50 | 0,2129 | 0,6335 | 0,7242 | 541 (237) | 5,8 s |
| **Game 2** | **v3 (entrenado en G1 + G3)** | **0,6426** | **0,8767** | **0,9336** | **233 (237)** | **13,2 s** |
| **Game 3** | Baseline Smooth 50 | 0,2169 | 0,5854 | 0,7188 | 638 (378) | 5,2 s |
| **Game 3** | **v3 (entrenado en G1 + G2)** | **0,5694** | **0,8560** | **0,9464** | **257 (378)** | **11,9 s** |
| **Promedio CV** | Baseline Smooth 50 | 0,2063 | 0,5927 | 0,7128 | 570 (290) | 5,5 s |
| **Promedio CV** | **v3 Propuesta** | **0,6242** | **0,8665** | **0,9373** | **242 (290)** | **12,6 s** |

### Conclusiones de la validación:
1. **Triplicación de spellF1:** v3 incrementa el spellF1 en más de **+0,41 puntos absolutos** frente al baseline de referencia oficial (`smooth50`).
2. **Eliminación de la fragmentación:** Mientras el baseline predice casi el doble de posesiones de las reales (570 frente a 290) debido a pérdidas de balón momentáneas, v3 predice 242 posesiones de duración media 12,6 s, emparejando fielmente la distribución real (11-14 s).
3. **Generalización entre formatos:** Entrenar en partidos CSV (Games 1 y 2) y predecir en formato EPTS (Game 3), o viceversa, mantiene un rendimiento sobresaliente (spellF1 > 0,56), demostrando transferencia real de las señales tácticas colectivas.

---

## 4. Reglas de Posesión y Arquitectura del Sistema

### 4.1 Extracción de Señales Tácticas (5 Hz)
Cada 0,2 s (5 fotogramas), se calculan 123 características centradas en física y geometría colectiva:
* **Dirección de ataque dinámica:** Identificada automáticamente por la posición del jugador más retrasado de cada equipo en cada bloque.
* **Márgenes y cercanía:** Distancia del 1º y 2º jugador más cercano de cada equipo al balón, número de jugadores en radio de 3 m y 8 m, y diferencia $d_A - d_H$.
* **Señales de inercia y oclusión:** Tiempo transcurrido desde la última observación del balón (`ball_since_s`) y tiempo hasta la siguiente reaparición (`ball_until_s`), conservando la posesión durante oclusiones en lugar de forzar balón parado.
* **Dinámica colectiva:** Velocidad mediana de jugadores, avance colectivo hacia la portería rival, compresión espacial de líneas defensivas y distancia del balón al centroide de cada equipo.

### 4.2 Modelo de Probabilidades Simétricas
Se entrena un ensemble `HistGradientBoostingClassifier` con 339.920 muestras extraídas de los 3 partidos externos bajo múltiples variantes de ruido y oclusiones.
Para evitar sesgo hacia el equipo local o visitante:
$$P(\text{clase} \mid x) = \frac{1}{2} \left[ P_{\text{model}}(y \mid x) + P_{\text{model}}(\text{swap}(y) \mid \text{swap}(x)) \right]$$

### 4.3 Decodificador de Secuencia Temporal (HSMM Viterbi)
Las emisiones logarítmicas de probabilidad se decodifican mediante un algoritmo de Viterbi con restricciones de duración mínima y matriz de transición tipificada:
* Duración mínima de 1,0 s (5 pasos a 5 Hz) para cualquier estado $H$, $A$ o $D$, impidiendo micro-etiquetas causadas por toques espurios.
* Penalización de cambio directo de posesión ($H \leftrightarrow A$) de 4,0 unidades logarítmicas, exigiendo evidencia continuada de control rival para validar una transición.
* Penalización de cambio a/desde balón parado ($H/A \leftrightarrow D$) de 2,5 unidades, permitiendo que una jugada muerta confirmada por falta de movimiento colectivo transicione naturalmente.

---

## 5. Resultados del Candidato Final (`temporal_v3.csv`)

El modelo final fue entrenado en los datos combinados de los 3 partidos externos (339.920 muestras) y ejecutado sobre el partido del concurso sin acceder en ningún momento a etiquetas del reto:
* **Filas generadas:** Exactamente 199.319 fotogramas, de 0 a 199.318.
* **Comprobación de formato:** `evaluation/validate_submission.py` devuelve **OK / Valid**.
* **Distribución de etiquetas:**
  - `H`: 49.545 fotogramas (24,9%)
  - `A`: 29.690 fotogramas (14,9%)
  - `D`: 79.622 fotogramas (39,9%)
  - `X`: 40.462 fotogramas (20,3%, exactamente los fotogramas no puntuados)
* **Estructura temporal predicha:**
  - 250 posesiones en juego con duración media de 12,7 s.
  - 143 fases a balón parado con duración media de 22,3 s.
  - Coherencia táctica con el desarrollo del encuentro: Barcelona (`H`) domina la cuota de posesión en juego (~62,5% frente a ~37,5% de Athletic).

---

## 6. Reproducibilidad y Verificación

El pipeline es 100% determinista y almacena sus hashes en `data/work/temporal_v3/run.json`:
* **SHA-256 del modelo entrenado:** `fa65d00760b1a0baedc3b62b5e13e09cf4f8037d19bf279c4fce2d21240e31ea`
* **SHA-256 del archivo de submission:** `084621c9a22c192700c9dff508f4ff68cb24646b9d4a705184f7eb14c9d9781a`

### Instrucciones de Reproducción:
```bash
# 1. Crear entorno e instalar dependencias fijadas
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt

# 2. Descargar y verificar datos externos (Sample Games 1, 2, 3)
.venv/bin/python scripts/fetch_external.py

# 3. Ejecutar suite de pruebas unitarias
.venv/bin/python -m unittest discover -s tests -v

# 4. Ejecutar benchmark de validación cruzada externa
OMP_NUM_THREADS=8 .venv/bin/python -m possession.benchmark_external

# 5. Entrenar y generar la submission final v3
OMP_NUM_THREADS=8 .venv/bin/python -m possession.run_v3 \
  --output evaluation/results/temporal_v3.csv \
  --work-dir data/work/temporal_v3

# 6. Validar el archivo resultante con el validador oficial
.venv/bin/python evaluation/validate_submission.py evaluation/results/temporal_v3.csv
```

---

## 7. Limitaciones y Trabajo Futuro

1. **Ambigüedad en balones divididos y duelos aéreos:** Sin señal 3D directa de altura del balón en el ATD, un balón que sobrevuela a un jugador puede parecer un control aéreo si los dos jugadores se encuentran cerca.
2. **Fueras de banda rápidos:** En saques de banda inmediatos sin apenas interrupción de movimiento de jugadores, el modelo en ocasiones concatena la posesión si el equipo que sacó es el mismo que envió el balón fuera.
3. **Visión por computador auxiliar:** La integración de detección visual de siluetas de árbitro levantando el brazo o pancartas de fuera de juego podría refinar los instantes exactos de transición a balón parado en jugadas dudosas.
