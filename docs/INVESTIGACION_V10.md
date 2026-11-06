# Investigación aplicada: repositorios de posesión y modelos temporales

Revisión: 6 de noviembre de 2026. Se archivaron README, commits y archivos técnicos
de doce repositorios públicos en `data/external/research_v10/`. El manifiesto
registra URLs y SHA-256. Se inspeccionó código; no se ejecutaron sus instaladores.

## Qué resultados son comparables

El reto ordena por spellF1 de intervalos H/A, no por detección de objetos, precisión
del poseedor individual, F1 de eventos ni porcentaje agregado de posesión. No se
puede trasladar una cifra publicada en otra tarea a nuestro marcador.

| Fuente primaria | Qué evalúa o implementa | Aplicación comprobada aquí |
| --- | --- | --- |
| [PathCRF](https://github.com/hyunsungkim-ds/pathcrf), [artículo](https://arxiv.org/html/2602.12080v1) | 69,64 % de exactitud de aristas; 75,69 % de F1 de eventos en su test. El evento exige tipo, jugador y tolerancia temporal; no es spellF1. | Interacciones entre jugadores, estados temporales y coherencia de secuencia. Se probó una red propia de atención completa; no superó la combinación seleccionada. |
| [Ball Radar](https://github.com/hyunsungkim-ds/ballradar) | Reconstrucción de trayectoria y poseedor con conjuntos y Bi-LSTM. Parte de sus datos Fitogether no es pública. | Se probaron representación por jugador, memoria Bi-LSTM y red híbrida. Se entrenó exclusivamente con fuentes públicas. |
| [Borghesi et al., 2023](https://iris.polito.it/handle/11583/2976470) | 86,2 % de accuracy de posesión y 89,2 % de vivo/parado; embeddings temporales por objeto y agregación invariante. | El cambio efectivo fue procesar cada trayectoria antes de resumir el equipo. No se reproducen ni se atribuyen sus cifras a nuestro modelo. |
| [Tryolabs](https://github.com/tryolabs/soccer-video-analytics) | Demostración de posesión desde vídeo. Sus autores describen su detector de balón como un modelo de juguete sobreajustado. | Revisado; sus pesos no se incorporaron al candidato. |
| [dattran1243](https://github.com/dattran1243/football-video-analytics) | Benchmark de detección; su README distingue proximidad visible de posesión oficial. | Gestión de calidad por plano; no se toma mAP como score de posesión. |
| [futtraffic](https://github.com/danrublop/futtraffic) | Reconstrucción de juego, detección y tracking; integra PathCRF como componente opcional. | Referencias a datasets y evaluación de componentes. No aporta un spellF1 transferible a este reto. |
| [SeyeSamba64](https://github.com/SeyeSamba64/football-match-analysis) | Asignación al jugador más cercano y porcentaje acumulado. | No sustituye la definición de control del reto. |
| [danendral](https://github.com/danendral/cv-football-analysis) | Detección, tracking y porcentajes por equipo. | Revisado como tubería de vídeo; no se encontró una evaluación comparable de intervalos H/A/D. |
| [pitch-vision](https://github.com/CarterFitzgerald/pitch-vision) | Proximidad y porcentaje; documenta pérdidas del balón, especialmente en vuelo. | Confirma la necesidad de tratar los huecos sin convertirlos automáticamente a D. |
| [donaldng05](https://github.com/donaldng05/football-cv) | Tracking, histéresis de control y redes de pases. | Revisado; las cifras de detección/tracking no se usan para ordenar modelos de posesión. |
| [SportsMOT analytics](https://github.com/mohamedkhaled110/sportsmot-tracking-analytics) | Proximidad por frame, métricas de tracking y aplicación de análisis. | No hay una cifra de spellF1 equivalente. |
| [DataME2](https://github.com/DataME2/football_video_analysis) | Detector y asignación al pie más cercano, con porcentajes. | Su benchmark del detector no valida la segmentación que necesitamos. |
| [sportvision](https://github.com/MohibShaikh/sportvision) | Detección RF-DETR y proximidad por equipo. | Se revisó el alcance; no se incorporó como estimador de control. |

La tabla contiene doce repositorios y un artículo adicional. «Revisado» no significa
que se haya reproducido su benchmark. Las implementaciones nuevas son propias,
inspiradas en esas estructuras; no son reproducciones completas de PathCRF o Ball Radar.
No se descargaron ni usaron sus checkpoints ni sus datos propietarios.

## Decisión importante de evaluación

Los checkpoints públicos de PathCRF se entrenaron con una partición de los mismos
siete partidos Sportec que usamos como referencia. Aplicarlos a cualquiera de sus
partidos de entrenamiento y presentar el resultado como validación habría mezclado
entrenamiento y evaluación. Sus parámetros publicados también plantean poseedor y
receptor, con slots estables y episodios preparados, no directamente H/A/D en
tracking fragmentado. Se tomaron las ideas y se entrenaron modelos propios con
particiones verificadas.

## Qué se probó de verdad

- Embedding de conjuntos por frame; trayectoria temporal por jugador antes del pooling.
- Fusión del conjunto aprendido con los 156 atributos anteriores.
- Dos capas de atención entre jugadores y un token del balón.
- Reconstrucción de observaciones del balón ocultadas artificialmente; etiquetas de
  posesión solo externas, coordenadas automáticas sin etiquetas humanas del reto.
- Bi-LSTM y variante híbrida con CNN temporal; inferencia recurrente por partes completas.
- Repeticiones de semilla, entrenamiento de 3.000 pasos y oclusiones aleatorias nuevas.
- Segunda cámara sincronizada sin usar el balón, y fusión de observaciones ausentes.
- Decodificación a 25 Hz mediante interpolación de probabilidades o log-probabilidades.
- Comparación por partidos excluidos y ajuste final con seis partidos externos.

La reconstrucción auxiliar, la atención completa y las variantes recurrentes no
justificaron entrar en el candidato final. La memoria por trayectoria, el promedio
de semillas, la segunda cámara y la resolución temporal sí aportaron evidencia.

## Fuentes adicionales y acceso

Se verificaron [SoccerTrack v2](https://huggingface.co/datasets/atomscott/soccertrack-v2)
y [SNGAR](https://huggingface.co/datasets/OpenSportsLab/SNGAR-Action-Spotting-Tracking).
Requieren acceso autorizado. No había sesión de Hugging Face guardada; no se
aceptaron términos, crearon cuentas ni accedió a datos restringidos. Se preguntó
al usuario por acceso existente y se continuó con los datos disponibles.

También se consultaron detectores públicos de fútbol de 2022/2023. Sus mAP de
objetos no se convierten en F1 de posesión y no se incorporaron pesos al modelo.
La edición 2026 de la hackathon tenía otro criterio y el chatbot ganador no aporta
un benchmark reproducible de este problema; véase `INVESTIGACION_METODOS.md`.
