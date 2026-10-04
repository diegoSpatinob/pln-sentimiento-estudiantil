# Fase 3A: auditoría y preparación de T1/T2/T3

Esta fase prepara la ejecución futura; no ejecuta experimentos, fine-tuning ni
evaluación predictiva de test. Fase 2B está fusionada y su smoke real exitoso
queda documentado en [colab_smoke_transformer.md](colab_smoke_transformer.md).
Sus métricas y artefactos siguen siendo **NO OFICIALES**.

## Hallazgos y correcciones

Auditoría inicial en `feat/transformer-experiments`, HEAD
`4630d19222c7d4e9b83af3c093c941cd5aa6910f`:

- La interfaz de Fase 0, la revisión original de BETO, el padding dinámico,
  las métricas y las barreras contra test ya estaban implementadas.
- `save_total_limit=2` podía borrar una de las tres épocas antes de seleccionar.
  Oficialmente se retienen tres checkpoints.
- El hook del smoke desempataba por F1 redondeado y accuracy, pero no había una
  función compartida con la selección global. Se extrae una única clave y selector.
- La API admitía cualquier LR del protocolo para cualquier ID, batch físico 8
  y CPU/fp32. Los runs oficiales ahora fijan ID/LR y CUDA/fp16/16/1.
- Las carpetas únicas por experimento impedían un rerun conservando intentos
  anteriores; se agrega un UUID completo por ejecución a ambas rutas.
- Faltaban persistencia oficial por época, estado final, runtime, hash del JSON,
  commit y resumen global. Se agregan sin producir resultados durante esta fase.
- El guard de resume no cubría el alias antiguo `train(model_path=...)` de
  Transformers 4.57.1. Se rechaza junto con cualquier argumento adicional antiguo.
- El notebook conserva documentación histórica de 2A; su sección 6.10 establece
  qué decisiones quedan sustituidas en 3A, sin añadir celdas de entrenamiento.

Se revisó el código instalado y la [API oficial de Trainer 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).
El checkpoint automático de HF y su rotación se ejecutan antes de `on_save`;
el callback oficial empareja la evaluación con ese checkpoint ya escrito.

## Protocolo oficial fijo

| Parámetro | Valor |
| --- | --- |
| Modelo | `dccuchile/bert-base-spanish-wwm-cased` |
| Revisión | `c4d86612f51b4f46759c8390d1798c2febe71b93`, exactamente la del JSON |
| T1 / T2 / T3 | LR `1e-5` / `2e-5` / `3e-5` |
| Entrada / etiqueta | `comentario_limpio` / `sentimiento_id` |
| Clases / num_labels | Negativo=0, Positivo=1 / 2 |
| Seed / max_length | 42 / 128 |
| Padding | Dinámico por batch, `DataCollatorWithPadding` |
| Épocas / weight decay | 3 / 0.01 |
| Batch físico / acumulación / efectivo | 16 / 1 / 16 |
| Dispositivo / precisión | Una GPU CUDA / fp16 |
| class_weight / oversampling / early stopping | None / desactivado / desactivado |
| Evaluación / guardado | Al terminar cada época |
| Optimizador / scheduler / warmup | AdamW de PyTorch / lineal / 0 pasos, como en 2A |

El JSON deja de tener campos operativos nulos: ahora registra 16/1/fp16/cuda.
`validate_config` protege esos valores. Los valores del smoke continúan siendo
explícitos y pueden reproducir su configuración técnica histórica sin convertirse
en una ejecución oficial. No hay fallback automático de batch o precisión oficial.

## Checkpoints y selección

Oficial: `save_strategy="epoch"`, `save_total_limit=3`,
`save_safetensors=True`, `num_train_epochs=3`, `max_steps=-1`.
Se conservan los checkpoints de las tres épocas, incluyendo pesos, estado de
Trainer y los estados que HF guarda por defecto (optimizador, scheduler, RNG y
scaler cuando corresponda). No se elimina ninguno al aplicar la selección.
Los checkpoints se llaman `checkpoint-<global_step>`, no `checkpoint-<epoch>`;
el registro de métricas contiene la correspondencia explícita.

Oficial: `load_best_model_at_end=False`, `metric_for_best_model=None` y el hook
`_determine_best_metric` no establece un ganador. HF conserva el modelo de la
última época en memoria. **Ese modelo en memoria no representa la selección
oficial**: la selección se hace sobre las tres épocas registradas, después de
completarlas, mediante `select_validation_candidate` y `validation_selection_key`:

1. Mayor `round(f1_macro, 4)` de validation.
2. Mayor accuracy original, sin redondearla.
3. Menor learning rate.
4. Época más temprana.

Se mantiene el redondeo Python existente: empates exactamente a mitad usan la
regla al par de `round`, aplicada al float original. No se redondean las métricas
persistidas. Dentro de un experimento el LR es constante. La selección global
aplica exactamente la misma función al ganador de cada T1/T2/T3. Un empate total
conserva el primer candidato; las épocas oficiales únicas y los LR distintos
hacen que ese caso no cree ambigüedad en las ejecuciones válidas.

El smoke conserva `load_best_model_at_end=True`, `metric_for_best_model="f1_macro"`
y límite 2, con su hook usando la misma clave para su recarga técnica. Su selección
no participa en la comparación oficial.

Para cargar posteriormente el checkpoint elegido, sin evaluar ningún dato:

```python
import json
from pathlib import Path
from scripts.transformer import load_selected_checkpoint

# Sustituir por la ruta real del resultado del experimento seleccionado.
result = json.loads(Path(result_path).read_text(encoding="utf-8"))
model = load_selected_checkpoint(result)
```

El helper valida el resultado completo, la regla, los hashes y la pertenencia del
checkpoint al run. Carga exclusivamente esa carpeta local mediante
`AutoModelForSequenceClassification.from_pretrained(..., local_files_only=True)`
y devuelve el modelo en modo eval en CPU. No entrena ni predice. Esta recarga de
consumo nunca pasa a `model_init` de otro experimento.

## Independencia y reserva de test

Antes de cargar configuración, preparar datos, descargar o inicializar BETO,
`run_official_experiment` exige que `git status --porcelain --untracked-files=all`
esté vacío. Se rechazan cambios versionados (incluidos los staged) y archivos no
rastreados no ignorados. Las rutas ignoradas data/processed/, models/ y results/
no bloquean el run. El builder oficial repite el control antes de inicializar el
modelo; al finalizar también se exige un árbol limpio y el mismo commit antes de
publicar resultados oficiales. El smoke conserva su registro técnico independiente.

Cada invocación oficial genera `RUN_<uuid de 32 hexadecimales>`. T1/T2/T3 y sus
reruns tienen carpetas distintas. `mkdir(exist_ok=False)` y escritura exclusiva
del resultado impiden sobrescribir ejecuciones; incluso una colisión de UUID
se rechaza. Cada `model_init` parte del ID y revisión original de BETO, con seed
42 y una cabeza de dos clases. Nunca toma el modelo smoke ni otro checkpoint.

`train(resume_from_checkpoint=False)` es la única llamada del ejecutor. El Trainer
rechaza resume con ruta, resume=True, `model_path`, trials y un segundo entrenamiento
del mismo Trainer. También rechaza cambios posteriores de sus argumentos preparados.
Un intento interrumpido mantiene sus checkpoints/épocas ya escritos y queda
`FAILED_OR_INTERRUPTED`; un corte abrupto puede dejar `RUNNING`. Ambos quedan
excluidos de la comparación. La siguiente invocación comienza con UUID nuevo y
BETO original, sin reutilizar estados. No hay política nueva de resume.
Los JSON mutables del run se actualizan por reemplazo atómico para conservar
la versión anterior si se interrumpe una escritura.

Los datasets solo se preparan para train/validation, usando las particiones
congeladas de Fase 0. La interfaz común puede leer test para integridad; se descarta
antes de tokenizar. Los guards existentes protegen entrenamiento, evaluación,
predict y dataloaders por split, índices, etiquetas, hash y modelo/revisión.
El ejecutor no llama a predict, no recibe test ni guarda predicciones. El resumen
solo lee resultados de validation y hashes de train/validation; no carga un modelo.
`test_used=false` acompaña manifiestos, épocas, resultados y resumen.

## Artefactos futuros y esquema

Las siguientes son **rutas de formato**, no ejecuciones ni resultados generados:

```text
models/transformer/<T1|T2|T3>/<RUN_uuid>/
  run_manifest.json
  checkpoint-<step época 1>/...
  checkpoint-<step época 2>/...
  checkpoint-<step época 3>/...
results/transformer/<T1|T2|T3>/<RUN_uuid>/
  run_manifest.json
  validation_epochs.json
  validation_result.json
results/transformer/validation_summary.json
```

`run_manifest.json` se duplica en modelos y resultados. Registra:

- `status`, `official=true`, `purpose="OFFICIAL_EXPERIMENT"`, `experiment_id`,
  `run_id`, `created_at_utc`, `test_used=false` y `resume_from_checkpoint=false`.
  `test_policy` conserva la política textual de reserva del JSON.
- `git_commit` y `working_tree_clean=true` identifican el código oficial exacto.
  `git` conserva commit, indicador de cambios locales (false en ejecución oficial),
  SHA-256 del diff de HEAD y SHA-256 del script como evidencia complementaria.
- `configuration`: copia íntegra del JSON, con seed, máximos de épocas, weight
  decay, max_length, clases y política; `configuration_sha256`: hash de sus bytes.
- `initial_checkpoint`: model_id y revision originales; `learning_rate`, batch
  físico, acumulación, batch efectivo y precision efectivos en campos separados.
- `training_arguments`: argumentos efectivos completos de HF; `runtime`: GPU(s),
  CUDA, PyTorch, Transformers, Python y soporte de precisión; `accelerate`.
- `data.train` y `data.validation`: records, sha256 del CSV e indice_original
  para trazabilidad. No contiene test, textos ni predicciones.
- Al terminar: `runtime_seconds`. Ante error: `failure_type` y estado de fallo.

`validation_epochs.json` se actualiza después de cada checkpoint completo. Contiene
`official`, `status`, `test_used`, `evaluation_partition="validation"`,
`experiment_id`, `run_id`, y `epochs`. Cada entrada de epochs contiene exactamente:
`experiment_id`, `run_id`, `learning_rate`, `epoch`, `global_step`, `checkpoint`
(ruta relativa a la raíz), y `metrics`. Metrics contiene `f1_macro`, `accuracy`,
`precision_macro`, `recall_macro` y `precision_class_0/1`, `recall_class_0/1`,
`f1_class_0/1`, `support_class_0/1`.

`validation_result.json` solo se escribe si existen las tres épocas completas.
Contiene todos los campos del manifiesto final, `evaluation_partition`, `epochs`,
`selected` (la entrada ganadora completa), `selection_rule`, `training_metrics`
y `checkpoint_loading`. El runtime de la ejecución mide train, evaluación,
guardado y selección; `training_metrics.train_runtime` es además el tiempo que
informa HF. Preparar/descargar/tokenizar queda fuera de este cronómetro.

`validation_summary.json` contiene `official`, `test_used`, `evaluation_partition`,
`selection_rule`, tres entradas de `experiments` y `selected_experiment_id`,
`selected_run_id`, `selected_checkpoint`. Cada entrada conserva la época ganadora
completa, `runtime_seconds`, `configuration_sha256`, `git`, `git_commit`,
`working_tree_clean=true`, `result_path` y un booleano `selected`.
Antes de seleccionar ganador global se requieren exactamente tres resultados
validados: T1/T2/T3 sin duplicados, run IDs distintos, mismo git_commit, hash del
JSON, hashes de train/validation y revisión de BETO. Cada ID conserva su LR fijo.
También se exige el mismo hash del script. Toda incompatibilidad aborta sin escribir
el resumen; se rechazan smoke, runs incompletos y código local sin commit.

Los tres resultados a comparar se indican explícitamente. Si hay varios reruns,
no se elige automáticamente el más reciente ni el de mayor F1. Un resumen existente
no se sobrescribe: `--summary-output` permite otra ruta nueva bajo results/transformer.
Los directorios models/ y results/transformer/ siguen ignorados por Git.

## Comandos futuros en Colab — no ejecutados en Fase 3A

Desde la raíz de la copia que incluya estos cambios y las particiones congeladas:

```bash
python -m scripts.transformer --experiment T1
python -m scripts.transformer --experiment T2
python -m scripts.transformer --experiment T3
```

No se proporcionan LR, batch, precisión ni dispositivo: el ID fija todo el
protocolo oficial y la CLI rechaza esos overrides. Cada proceso termina su propio
run antes de comenzar el siguiente. No hay comando que lance los tres automáticamente.

Después, sustituir las rutas por las tres rutas reales elegidas:

```bash
python -m scripts.transformer --summarize \
  results/transformer/T1/RUN_REAL_T1/validation_result.json \
  results/transformer/T2/RUN_REAL_T2/validation_result.json \
  results/transformer/T3/RUN_REAL_T3/validation_result.json
```

Antes de entrenar, disponer de la copia final del código/configuración en un commit
con working tree limpio, el entorno
CUDA/fp16 de una GPU, las dependencias de Fase 2B y los CSV/manifiesto originales en
`data/processed/` bajo esa raíz. No reconstruir splits ni volver a ejecutar Fase 0
para obtener otros datos. Conservar espacio para las tres épocas de cada run y
respaldar artefactos antes de que termine la sesión Colab. Si hay CUDA OOM, detener
el run; no cambiar automáticamente el protocolo oficial.

No hay una decisión metodológica pendiente. La validación integral del nuevo
ejecutor sobre una GPU solo ocurrirá al ejecutar la fase oficial autorizada;
los tests locales de 3A prueban contratos y persistencia con dobles. PyTorch y
Accelerate no están instalados localmente, por lo que la regresión real de AMP
existente se omite; el smoke real exitoso de Colab permanece como evidencia de 2B.
