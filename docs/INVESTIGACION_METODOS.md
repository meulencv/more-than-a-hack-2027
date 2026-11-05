# Investigación de métodos y edición anterior

Revisión: 5 de noviembre de 2026. Investigación de fuentes primarias. Las técnicas
se describen como opciones, no como mejoras de rendimiento ya demostradas.

## Qué ganó en 2026

Barça Innovation Hub identifica a **IpurdIAr** como ganador de More Than a Hack 2026.
Su sistema respondía preguntas tácticas combinando tracking, eventos inferidos y
fragmentos de narración transcritos con Whisper. Transformaba coordenadas en señales
como zonas de campo, presión y compactación antes de consultarlas con un agente.
No localizamos un repositorio verificable del equipo ganador durante esta búsqueda.
La fuente describe el producto y las decisiones de diseño, no publica una evaluación
reproducible de posesión fotograma a fotograma.

Fuente: [entrevista oficial de Barça Innovation Hub, 30 de abril de 2026](https://barcainnovationhub.fcbarcelona.com/blog/ipurdiar-football-analysis-chatbot/).

**No es el mismo formato de competición.** El repositorio de 2027 explica que la
edición de 2026 valoraba originalidad, uso de LLM y presentación; la fase actual se
clasifica por spellF1. Copiar el chatbot ganador no equivale a mejorar ese indicador.

Fuente: [OVERVIEW oficial del reto 2027](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/OVERVIEW.md).

## Repositorios técnicos útiles

### PathCRF: trayectorias de jugadores y consistencia de secuencia

El repositorio oficial de KDD 2026 plantea control y pases como estados de un grafo
dinámico; un CRF impide transiciones incompatibles. Incluye checkpoints y emplea
siete partidos públicos Sportec/DFL. Sus referencias procesadas combinan eventos
sincronizados y tracking. La estructura exige adaptar los slots fragmentados de
nuestro tracking; no es una sustitución inmediata del estimador actual.

Fuentes: [repositorio oficial PathCRF](https://github.com/hyunsungkim-ds/pathcrf),
[artículo de sus autores](https://arxiv.org/abs/2602.12080).

### Ball Radar: aprender el contexto temporal y colectivo

Ball Radar, KDD 2023, combina Set Transformers con Bi-LSTM: estima el poseedor y
condiciona en él la trayectoria del balón. Es una alternativa pertinente a árboles
por fotograma cuando faltan observaciones. Para el reto interesa especialmente la
idea de inferir desde movimiento colectivo y de entrenar con oclusiones; su
adaptación debe comprobarse con partidos externos y sin identidades estables.

Fuente: [repositorio oficial Ball Radar](https://github.com/hyunsungkim-ds/ballradar).

### Tryolabs: útil como demostración, no como modelo robusto listo

Su proyecto estima posesión desde vídeo con detección y tracking. Sus propios
autores advierten que el checkpoint de balón publicado es un modelo de demostración
sobreajustado a pocos vídeos. No se recomienda descargarlo y asumir que mejorará
este reto, que ya proporciona tracking automático.

Fuente: [repositorio de Tryolabs y advertencia de los autores](https://github.com/tryolabs/soccer-video-analytics).

## Hallazgo aplicado: referencias nativas Sportec/DFL

La colección **An integrated dataset of spatiotemporal and event data in elite
soccer** publica siete partidos alemanes, bajo **CC BY 4.0**. Descargamos dos:
Köln–Bayern y Bochum–Leverkusen, del 27 de mayo de 2023; no incluyen el partido
Barcelona–Athletic del concurso. Los XML de tracking contienen directamente
`BallStatus` y `BallPossession` por fotograma. Esto evita depender exclusivamente
de nuestras referencias débiles reconstruidas desde eventos de Metrica.

Fuentes: [colección original y licencia](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177),
[artículo de datos](https://www.nature.com/articles/s41597-025-04505-y).

El lector nuevo `possession/sportec_native.py` conserva esas etiquetas solo en
`Match.meta`, separadas de las observaciones. La correspondencia está contrastada
con el lector Sportec de kloppy: posesión 1 local, 2 visitante; estado 1 en juego,
0 parado. Los valores desconocidos se dejan sin etiquetar. Se inserta un separador
X entre partes para que el evaluador no cuente una transición artificial de descanso.
No se aplican correcciones humanas al partido del reto ni se lee su ground truth.

Fuente del mapeo: [SportecTrackingDataDeserializer oficial de kloppy](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/sportec/deserializer.py).

El manifiesto `sportec_native_manifest.json` registra URLs, tamaños y SHA-256 de los
cuatro XML necesarios. Los archivos quedan en `data/external/pathcrf/`, excluidos
de Git. Las etiquetas son nativas del proveedor: no se afirma que su definición de
control sea perfectamente idéntica a la del concurso. Sigue siendo necesario evaluar
transferencia y ruido entre dominios.

## Prioridades concretas

1. Evaluar el candidato congelado en las dos nuevas referencias nativas, sin volver
   a seleccionar métodos por los cinco minutos públicos del concurso.
2. Entrenar un modelo temporal con las observaciones de otros partidos y comparar
   por partido excluido. Investigar especialmente intervalos de balón perdido.
3. Si aporta valor externo, combinar la inferencia secuencial con el clasificador
   existente. Conservar las alternativas que no mejoren como experimentos descartados.
4. Tratar la visión como una fuente adicional para contactos y reinicios, no como
   una garantía automática de mejor posesión ni como una razón para añadir un LLM.

## Reproducción de las referencias nativas y auditoría de origen

Ejecutar con .venv/bin/python scripts/fetch_sportec_native.py. Para las pruebas:
.venv/bin/python -m unittest discover -s tests -p 'test_sportec*.py' -v.

El descargador verifica exactamente los cuatro XML del manifiesto. Conserva un
archivo existente solo si coinciden tamaño y SHA-256; las descargas se escriben
primero como .part y no reemplazan el destino sin verificación correcta.
Se utiliza la verificación TLS normal de curl, sin opciones para desactivarla.

La auditoría completa de los timestamps del balón está en
sportec_native_timestamp_audit.json: los **287.524 intervalos consecutivos dentro
de las partes** corresponden exactamente a **0,04 s por incremento de frame nativo**,
sin intervalos inválidos. Confirma 25 Hz en ambos partidos; no se comparan los
relojes entre partes como si fueran juego continuo.

| Partido | BallStatus 1, en juego | BallStatus 0, parado | Frames originales |
| --- | ---: | ---: | ---: |
| J03WMX | 85.369 | 60.598 | 145.967 |
| J03WN1 | 68.377 | 73.184 | 141.561 |

BallPossession nativo tiene únicamente códigos 1/2 en estos archivos: J03WMX,
74.558/71.409; J03WN1, 68.995/72.566. Esos recuentos incluyen balón parado y
**no son cuotas de posesión en juego**. El lector ignora la propiedad mientras
BallStatus es 0. El hogar/visitante se toma del atributo Role de cada equipo en
metadata, no de su orden: en los dos XML el visitante aparece primero.
Los tests comprueban desconocidos y mapeo, así como tamaño y hash del descargador.
