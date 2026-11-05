# Preparación y propuesta — More Than A Hack 2027


Estado actualizado de la implementación y resultados: [IMPLEMENTACION.md](IMPLEMENTACION.md).
Este documento conserva la propuesta inicial.

Revisión: 5 de noviembre de 2026. Repositorio oficial clonado en la raíz:
https://github.com/metrica-sports/more-than-a-hack-2027
Commit de referencia: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.

## Qué hay que resolver

En el partido Barcelona–Athletic hay que estimar, por algoritmo, la posesión en cada
fotograma del vídeo **táctico**. No basta con calcular un porcentaje total.

El resultado es un CSV UTF-8 con cabecera `frame,label` y **199.319 filas**,
numeradas de `0` a `199318`, a 25 fps:

| Etiqueta | Significado |
| --- | --- |
| H | Barcelona tiene la posesión y el balón está en juego |
| A | Athletic tiene la posesión y el balón está en juego |
| D | Balón parado / fuera de juego activo |
| X | Abstención o fotograma sin evaluación |

Durante un balón dividido o suelto se conserva la última posesión hasta que el rival
obtiene control. Un rebote o un toque aislado no equivalen a recuperar la posesión.

## Qué puntúa

La clasificación depende de **spellF1**: compara intervalos completos de posesión,
con emparejamiento uno a uno y solapamiento temporal a varios umbrales. La fragmentación
de una posesión en muchas pequeñas perjudica mucho. `macroF1` mide las clases por
fotograma y sirve como primer desempate. La referencia oficial a superar tiene
`spellF1 = 0,2433` en el partido completo; esa cifra no es una puntuación nuestra.

El ground truth público solo cubre los primeros cinco minutos de juego. Sirve para
verificar alineación y funcionamiento, **no para entrenar, seleccionar modelos ni
ajustar umbrales**. No hay clasificación pública en directo ni feedback previo a la entrega.

## Datos y estructura

```text
MoreThanAHack/
  data/
    tactical/        # 5 archivos originales, referencia de los resultados
    broadcast/       # 4 archivos originales, entrada opcional
    checks/          # comprobaciones y baselines regenerados; ignorados por Git
  starter/           # lectores y baseline oficiales
  evaluation/        # evaluador, validador y ejemplo etiquetado oficiales
  scripts/check_data.py
  docs/PROPUESTA.md   # este documento: propuesta, todavía sin modelo propio
  requirements-local.lock.txt
  .venv/             # entorno mínimo local
```

Los archivos se copian de Descargas conservando los originales. Los datos y vídeos
quedan excluidos de Git por el `.gitignore` oficial. La cámara de televisión tiene
su propio reloj, cortes y repeticiones: no se pueden juntar sus filas con las tácticas.
El tracking táctico se alinea como `video_frame = frame_id - 1`; la homografía ya
está indexada desde cero y el Smart Tagging usa segundos desde el inicio del vídeo.

Estado de la preparación: los nueve archivos están incorporados. El vídeo broadcast
ha terminado de descargarse y se ha copiado conservando el original. Hay una
discrepancia a investigar antes de utilizar esta cámara: su contenedor declara
**164.487 fotogramas**, mientras que el repositorio, el tracking y la homografía
indican **164.488**. El inventario señala por ello `video_frames: false` para
broadcast; no se han modificado los datos ni las constantes oficiales. Los datos
tácticos pasan todas las comprobaciones y permiten desarrollar el enfoque inicial.

## Propuesta técnica

Construir un estimador temporal de tres estados (`H`, `A`, `D`) con estas piezas:

1. **Evidencia geométrica.** Distancia en metros del balón a cada equipo, movimiento
   relativo, velocidad del balón, continuidad y calidad de las observaciones. Los
   slots del tracking no son identidades persistentes de jugadores.
2. **Evidencia táctica.** Usar Smart Tagging como señal probabilística: fases ofensivas,
   defensivas y reinicios. No copiar sus intervalos directamente como posesiones,
   y tratar explícitamente los equipos `N/A`.
3. **Control y memoria temporal.** Una máquina de estados inicial, seguida si aporta
   valor de un HMM o modelo semimarkoviano, que exija evidencia de control para cambiar
   de equipo y preserve la posesión durante pases, disputas y oclusiones. Evitar
   confundir «no veo el balón» con «balón parado»: falta en aproximadamente el 41 %
   de los fotogramas evaluados.
4. **Paradas y reinicios.** Inferirlos combinando geometría, movimiento colectivo y
   eventos automáticos. El vídeo táctico puede aportar evidencia cuando el tracking
   no sea suficiente; la homografía permite relacionar imagen y campo.
5. **Decodificación y exportación.** Resolver la secuencia por cada parte y generar
   el CSV determinista. Fuera de los intervalos oficiales evaluables, usar `X`.

Empezaría con la cámara táctica. La sincronización automática de broadcast y su
fusión solo se añadirían si aportan información útil frente al coste de implementarlas.
No hay todavía evidencia de que esta propuesta mejore el baseline.

Los parámetros se justificarán por criterios físicos, análisis sin etiquetas o
datos de otros partidos accesibles a los organizadores; no por el resultado en los
cinco minutos públicos. También conviene verificar robustez con oclusiones y ruido
sintéticos sin convertir esas pruebas en etiquetas manuales del partido.

## Entregables y restricciones

- CSV de posesión con nombre del equipo.
- Repositorio **público** con código, dependencias fijadas y comando que regenere
  exactamente el mismo CSV, sin publicar los datos del partido.
- Memoria técnica breve, PDF o Markdown: método, datos usados, entrenamiento,
  reglas de posesión, limitaciones y reproducción. Este documento es una propuesta
  inicial; no sustituye la memoria del método finalmente implementado.

Las etiquetas y las correcciones deben salir del algoritmo. No se permite etiquetar
o corregir a mano este partido, ni usar eventos humanos de este partido de proveedores
externos. Se permiten métodos y modelos externos inspeccionables, y etiquetas de
otros partidos. Equipos de **3 a 5 integrantes**.

## Fechas

Aunque la edición se llama 2027, esta fase termina en **noviembre de 2026**.
Según el correo aportado: todos los integrantes deben registrarse con el mismo
nombre de equipo antes del **10 de noviembre**; las instrucciones de envío llegarán
como máximo el **14 de noviembre**; cierre el **17 de noviembre a las 23:59 CET**
(hora de Madrid). El repositorio confirma el día 17, pero no especifica la hora.
El correo indica que pasan los 20 mejores; el repositorio anuncia resultados antes
del 26 de noviembre. El formulario de entrega se enviará por correo.

## Ejecución local

```bash
cd /Users/meulencv/development/projects/MoreThanAHack
source .venv/bin/activate
# En una instalación nueva: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
python scripts/check_data.py
python starter/baseline.py --out data/checks/baselines
python evaluation/run_evaluation.py --results data/checks/baselines
python evaluation/validate_submission.py data/checks/baselines/baseline_smooth50.csv
```

La ejecución de referencia reproduce las cuatro variantes oficiales, sin seleccionar
una variante por sus resultados públicos. Las librerías para notebooks (`kloppy`,
Jupyter, matplotlib) son opcionales y todavía no se han instalado.

Comprobado en este entorno: las cuatro salidas regeneradas son idénticas byte a byte
a las oficiales; el evaluador reproduce su tabla pública y el validador acepta
`baseline_smooth50.csv` con las 199.319 filas. Estos resultados verifican la instalación,
no constituyen una solución nueva. El inventario local se guarda en
`data/checks/data_inventory.json`; puede actualizarse ejecutando
`python scripts/check_data.py > data/checks/data_inventory.json`.

Fuentes oficiales: [problema](../CHALLENGE.md), [datos](../DATA.md),
[evaluación](../EVALUATION.md), [entrega](../SUBMISSION.md), [reglas](../RULES.md).
