> Documento histórico. El modelo actual y las comparaciones válidas están en [RESULTADOS_MODELO_FINAL.md](RESULTADOS_MODELO_FINAL.md). No deben compararse scores de distintos datasets como si pertenecieran al mismo test.

# Experimentos externos y Git local — 5 de noviembre de 2026

El estado anterior está conservado en el commit local `3f6c7ae`. No se ha publicado
ningún repositorio ni se ha hecho push. Los datos y derivados siguen excluidos,
y se excluyen también explícitamente formatos habituales de pesos de modelos.

## Comparación nueva

Se han entrenado modelos dejando fuera cada uno de los tres partidos externos.
Cada modelo se ha probado con dos degradaciones del partido excluido: seis casos.
Las semillas de entrenamiento y evaluación son distintas. Las mismas referencias
aproximadas derivadas de eventos se usan para comparar los métodos. Ninguna
etiqueta pública del partido del concurso se lee en estos experimentos.

| Método | spellF1 medio externo |
| --- | ---: |
| v3 actual | 0,6200 |
| Dos modelos: parado/en juego y equipo condicionado a juego | 0,6082 |
| Combinación de v3 y los dos especialistas | 0,6194 |
| ExtraTrees (160 árboles aleatorizados) | 0,5858 |
| Combinación de ExtraTrees y v3 | 0,6175 |

Las dos familias alternativas no justifican sustituir v3. El experimento no ha
usado Gemma ni descargado pesos adicionales: no hay evidencia para que un modelo
visual grande sea la siguiente mejora apropiada.

## Decodificación temporal

Sobre las probabilidades externas ya calculadas, se han comparado cuatro hipótesis:

| Hipótesis | spellF1 medio externo |
| --- | ---: |
| Actual: mínimo 1,4 s y segunda pasada | 0,6200 |
| Original: mínimo 1 s, costes 4/2,5/2,5, sin segunda pasada | 0,6250 |
| Mínimo 0,6 s con costes originales | 0,6243 |
| Menores permanencias y costes | 0,6119 |

La configuración original mejora modestamente la media en los tres partidos:
0,6561→0,6578; 0,6416→0,6483; 0,5624→0,5690. Esto NO demuestra una mejora en el
partido del concurso, ni una diferencia estadísticamente estable. Solo hay tres
partidos y dos degradaciones por partido; no son seis partidos independientes.
La selección sobre esta comparación es desarrollo externo, no un test final intacto.

Se guarda en `possession/config_external_selected.json`. No modifica la configuración
actual ni los CSV anteriores. El candidato está en
`evaluation/results/temporal_external_selected.csv` y reutiliza el modelo existente.
No se ha vuelto a puntuar contra los cinco minutos públicos para tomar esta decisión.

## Reproducción

```bash
OMP_NUM_THREADS=8 .venv/bin/python -m possession.experiment_v4
OMP_NUM_THREADS=8 .venv/bin/python -m possession.experiment_forest
OMP_NUM_THREADS=8 .venv/bin/python -m possession.experiment_decoder
OMP_NUM_THREADS=8 .venv/bin/python -m possession.run_v3 \
  --config possession/config_external_selected.json \
  --model data/work/temporal_v3/model.joblib \
  --output evaluation/results/temporal_external_selected.csv \
  --work-dir data/work/temporal_external_selected
```

Los resultados detallados y modelos de cada partición quedan en
`data/work/hierarchical_v4/`, ignorados por Git. Para reutilizar los experimentos
tras cambiar datos o extracción de variables hay que eliminar su caché de variables
y recalcularla; el identificador actual de caché solo incluye la configuración.
La ejecución forest y decoder utiliza la caché del experimento v4 previo.

## Limitación principal

Las etiquetas externas siguen siendo aproximadas y la calidad del tracking difiere.
No se ha demostrado que cambiar de clasificador resuelva esa limitación. Las
mejoras futuras necesitan mejores referencias externas y evidencia adicional de
control del balón, no seleccionar por resultados del tramo público del reto.
