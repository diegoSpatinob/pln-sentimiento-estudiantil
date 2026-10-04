# Fase 2B: Colab y smoke real de BETO — NO OFICIAL

Esta guía prepara una libreta **privada y nueva** en Google Colab. Copiar sus bloques
en celdas y ejecutarlos en orden. Delega todo el modelado en `scripts/transformer.py`:
no ejecuta el notebook de Diego, no verifica `baseline_C1.joblib`, no reconstruye
limpieza/split y no lanza T1/T2/T3. No usar «Ejecutar todo» antes de revisar el entorno.

El pipeline revisado y probado en la ejecución final de Fase 2B corresponde al commit
`c6ef8a4ad3b5d4b9c591297c33eef607b6d240af`.
Esta guía local no necesita estar publicada para copiar sus celdas. Una revisión
posterior debe fijarse y revisarse explícitamente; no se actualiza el código a un
HEAD móvil durante una ejecución.

La primera ejecución real utilizó **Tesla T4, CUDA del runtime PyTorch 13.0,
fp16, batch físico 16 y acumulación 1**. Las particiones de Fase 0 pasaron sus
verificaciones. No hubo CUDA OOM: se cargó BETO, se ejecutaron dos pasos de
optimización con backward y se evaluaron los 16 registros de validation con
cálculo de métricas. El smoke falló únicamente en la comprobación final de
save/reload por comparar forwards bajo condiciones AMP distintas; ese primer
intento no debe registrarse como un smoke completamente verificado. Se corrigió
la comprobación técnica sin cambiar hiperparámetros metodológicos. La ejecución
final posterior verificó correctamente la recarga, como se registra a continuación.

## Resultado final real de Fase 2B — NO OFICIAL

El smoke final pasó las comprobaciones técnicas de actualización de parámetros,
guardado y recarga, incluida la igualdad exacta del `state_dict`. Es **NO OFICIAL**:
sus métricas de validation no se utilizan para seleccionar modelos,
hiperparámetros, batch ni precisión, ni para comparar resultados con el baseline.
Este resultado verifica la infraestructura; no corresponde a T1/T2/T3 ni a
entrenamiento completo. Test no se utilizó predictivamente.

| Campo | Resultado final real |
| --- | --- |
| Commit probado | `c6ef8a4ad3b5d4b9c591297c33eef607b6d240af` |
| Run ID | `SMOKE_8b466ec8bf40` |
| Registro técnico | `TECH_e84c3c45f78c` |
| GPU | Tesla T4 |
| PyTorch | 2.11.0+cu130 |
| CUDA del runtime PyTorch | 13.0 |
| Transformers | 4.57.1 |
| Accelerate | 1.10.1 |
| `precision` | `fp16` |
| `physical_batch_size` | `16` |
| `gradient_accumulation_steps` | `1` |
| `learning_rate` | `2e-5` |
| `train_records` | `32` |
| `validation_records` | `16` |
| `optimizer_steps` | `2` |
| `parameter_update_verified` | `true` |
| `save_reload_verified` | `true` |
| `state_dict_verified` | `true` |
| `test_used` | `false` |
| CUDA OOM | No |
| Fallback físico 8 / acumulación 2 | No se ejecutó |

Las referencias a los artefactos de esta ejecución, según las rutas del pipeline,
son `results/transformer/smoke/SMOKE_8b466ec8bf40/smoke_test_NO_OFICIAL.json`,
`models/transformer/smoke/SMOKE_8b466ec8bf40/save_reload_diagnostics_NO_OFICIAL.json`
y `results/transformer/colab/TECH_e84c3c45f78c/technical_record_NO_OFICIAL.json`.
El batch 16 funcionó sin CUDA OOM; no se probó batch 8 ni se eligió batch mediante
métricas. No se incluyen aquí valores de métricas de clasificación como resultados
experimentales oficiales.

El stack observado en la primera ejecución fue exactamente el siguiente; es un registro
del entorno utilizado, no una instrucción para reemplazar el PyTorch funcional de
una sesión futura:

| Dependencia | Versión observada |
| --- | --- |
| torch | 2.11.0+cu130 |
| transformers | 4.57.1 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| accelerate | 1.10.1 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| joblib | 1.6.0 |

## 1. Activar GPU e inspeccionar antes de instalar

En Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU**.
La GPU asignada puede variar; no asumir T4, L4 o A100. Si no hay GPU, detenerse.
Colab no garantiza tipo de GPU ni continuidad de la sesión
([FAQ oficial](https://research.google.com/colaboratory/faq.html)).

```python
import json, subprocess, sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from packaging.version import Version
import torch

torch_dist = version("torch")
print("Python:", sys.version)
print("PyTorch preinstalado:", torch.__version__)
print("CUDA del runtime PyTorch:", torch.version.cuda)
if not Version("2.6") <= Version(torch_dist) < Version("3"):
    raise RuntimeError("DETENER: PyTorch no cumple >=2.6,<3. No reemplazarlo automáticamente.")
if not torch.cuda.is_available() or torch.version.cuda is None:
    raise RuntimeError("DETENER: GPU/CUDA no utilizable. Revisar el tipo de sesión antes de instalar.")
if torch.cuda.device_count() != 1:
    raise RuntimeError("DETENER: el protocolo exige una sola GPU visible para batch efectivo 16.")
probe = (torch.ones(1, device="cuda") * 2).item()  # Solo comprobación técnica; sin datos/modelo.
free_bytes, total_bytes = torch.cuda.mem_get_info()
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free",
                      "--format=csv,noheader"], capture_output=True, text=True, check=True)
before_versions = {}
for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "accelerate",
             "numpy", "pandas", "scikit-learn", "joblib"):
    try:
        before_versions[name] = version(name)
    except PackageNotFoundError:
        before_versions[name] = None
preflight = {"python": sys.version, "versions_before": before_versions,
             "torch_cuda_runtime": torch.version.cuda, "nvidia_smi": smi.stdout.strip(),
             "gpu_free_bytes_before": free_bytes, "gpu_total_bytes": total_bytes,
             "cuda_probe_ok": probe == 2}
Path("/content/beto_colab_preflight.json").write_text(json.dumps(preflight, indent=2))
print(smi.stdout)
```

Si importar PyTorch, ejecutar el probe o consultar el driver falla, **detenerse y
conservar el diagnóstico**. No instalar otra build ni cambiar CUDA en esta guía.
El requisito >=2.6 permite leer el `.bin` original de BETO con el cargador de
Transformers 4.57.1. CUDA de PyTorch y versión del driver son datos distintos.

## 2. Clonar o actualizar y fijar el código revisado

```python
import os, subprocess
from pathlib import Path

REPO = Path("/content/pln-sentimiento-estudiantil")
REVIEWED_COMMIT = "c6ef8a4ad3b5d4b9c591297c33eef607b6d240af"
REPO_URL = "https://github.com/diegoSpatinob/pln-sentimiento-estudiantil.git"
if not REPO.exists():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
else:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True)
    if dirty.strip():
        raise RuntimeError("DETENER: checkout con cambios; no se descartan ni sobrescriben.")
    subprocess.run(["git", "fetch", "origin"], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--detach", REVIEWED_COMMIT], cwd=REPO, check=True)
os.chdir(REPO)
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
```

Si ese commit no está disponible en GitHub, detenerse: debe publicarse la revisión
ya revisada. No sustituirlo silenciosamente por main u otra rama. `fetch` permite
actualizar los objetos del clon, pero la ejecución queda fijada al SHA indicado.

## 3. Plan de instalación sin reemplazar PyTorch

Mantener el PyTorch preinstalado compatible. No usar indiscriminadamente
`pip install -U torch`, `transformers[torch]` ni instalar todo el baseline.
Las dependencias de datos ya presentes se conservan: aquí solo se usa su interfaz
CSV, sin cargar el pickle histórico del baseline.

La siguiente celda crea restricciones para conservar torch y las librerías de datos
presentes. Solicita las cuatro versiones HF y añade librerías de datos **solo si
faltan**, con versiones ya registradas en `requirements.txt`. El dry-run no instala
paquetes. Revisar su salida antes de ejecutar la siguiente celda
([opciones oficiales de pip](https://pip.pypa.io/en/stable/cli/pip_install/)).

```python
import json, subprocess, sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

required_hf = ["transformers==4.57.1", "tokenizers==0.22.2",
               "huggingface-hub==0.36.2", "accelerate==1.10.1"]
data_defaults = {"numpy": "2.1.3", "pandas": "2.2.3",
                 "scikit-learn": "1.6.1", "joblib": "1.6.0"}
constraints, missing_data = [], []
for name in ("torch", *data_defaults):
    try:
        constraints.append(f"{name}=={version(name)}")
    except PackageNotFoundError:
        if name == "torch":
            raise RuntimeError("DETENER: PyTorch ausente; no se instala automáticamente.")
        missing_data.append(f"{name}=={data_defaults[name]}")
constraint_path = Path("/content/beto_colab_constraints.txt")
constraint_path.write_text("\n".join(constraints) + "\n")
install_command = [sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
                   "--constraint", str(constraint_path), *required_hf, *missing_data]
subprocess.run([*install_command, "--dry-run", "--report", "/content/beto_install_plan.json"], check=True)
plan = json.loads(Path("/content/beto_install_plan.json").read_text())
if any(item["metadata"]["name"].lower() == "torch" for item in plan["install"]):
    raise RuntimeError("DETENER: el plan propone cambiar PyTorch. Revisar sin aplicar.")
print("Comando de instalación revisable:", install_command)
```

Si el resolvedor falla o el plan revela una incompatibilidad, **no ejecutar la
instalación ni cambiar versiones para resolverla**: guardar el mensaje y revisarlo.
Los paquetes transitivos necesarios pueden instalarse; no se fuerza una build CUDA.

```python
import re
from importlib.metadata import distributions, distribution
from packaging.requirements import Requirement, InvalidRequirement
from packaging.utils import canonicalize_name

# Identificar lo que ya estaba instalado antes de aplicar el plan revisado.
preinstalled = {canonicalize_name(dist.metadata["Name"]) for dist in distributions()}
subprocess.run(install_command, check=True)
check = subprocess.run([sys.executable, "-m", "pip", "check"],
                       capture_output=True, text=True, check=False)
diagnostic_path = Path("/content/beto_pip_check.json")
diagnostic = {"stdout": check.stdout, "stderr": check.stderr, "returncode": check.returncode}
# Guardar e imprimir antes de clasificar, incluso si luego hay un error.
diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
print(json.dumps(diagnostic, indent=2))

expected_stack = {**dict(spec.split("==", 1) for spec in required_hf), **data_defaults,
                  "torch": preflight["versions_before"]["torch"]}
needed, visited, metadata_errors = set(expected_stack), set(), []
pending = [(name, frozenset()) for name in expected_stack]
while pending:
    name, extras = pending.pop()
    name = canonicalize_name(name)
    if (name, extras) in visited:
        continue
    visited.add((name, extras))
    needed.add(name)
    try:
        for text in distribution(name).requires or []:
            requirement = Requirement(text)
            if requirement.marker and not any(requirement.marker.evaluate({"extra": extra})
                                               for extra in set(extras) | {""}):
                continue
            pending.append((requirement.name, frozenset(requirement.extras)))
    except (PackageNotFoundError, InvalidRequirement) as error:
        metadata_errors.append(f"{name}: {error}")

pattern = re.compile(r"(?P<package>[A-Za-z0-9][A-Za-z0-9._-]*) (?P<version>\S+) "
                     r"(?:requires .+, which is not installed\.|"
                     r"has requirement .+, but you have .+\.)")
lines = [line.strip() for line in check.stdout.splitlines() if line.strip()]
conflicts, uncertain = [], list(metadata_errors)
if check.stderr.strip():
    uncertain.append("stderr no vacío: revisar el diagnóstico sin asumir que es inocuo.")
if check.returncode not in (0, 1) or not lines:
    uncertain.append("Código de salida o salida de pip check no reconocidos.")
if check.returncode == 0 and lines != ["No broken requirements found."]:
    uncertain.append("Salida inesperada para pip check exitoso.")
for line in lines:
    if check.returncode == 0 and line == "No broken requirements found.":
        continue
    match = pattern.fullmatch(line)
    if match is None:
        uncertain.append(line)
        continue
    package = canonicalize_name(match["package"])
    try:
        if version(package) != match["version"]:
            raise ValueError("La versión declarada no coincide con los metadatos instalados.")
    except (PackageNotFoundError, ValueError) as error:
        uncertain.append(f"{line}: {error}")
        continue
    classification = ("blocking" if package in needed else
                      "external_preinstalled" if package in preinstalled else "uncertain")
    conflicts.append({"line": line, "declaring_package": package,
                      "classification": classification})
    if classification == "uncertain":
        uncertain.append(f"{package}: no pertenece al inventario previo ni al stack verificado.")

actual_stack, version_errors = {}, []
for name, expected in expected_stack.items():
    try:
        actual_stack[name] = version(name)
    except PackageNotFoundError:
        actual_stack[name] = None
    if actual_stack[name] != expected:
        version_errors.append(f"{name}: esperado {expected}, encontrado {actual_stack[name]}")
diagnostic.update(conflicts=conflicts, unclassified=uncertain,
                  expected_stack=expected_stack, actual_stack=actual_stack,
                  version_errors=version_errors,
                  torch_unchanged=actual_stack["torch"] == preflight["versions_before"]["torch"])
diagnostic["can_continue"] = (not uncertain and not version_errors
                               and not any(item["classification"] == "blocking" for item in conflicts))
diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
print(json.dumps(diagnostic, indent=2))
print("Diagnóstico conservado en:", diagnostic_path)
if not diagnostic["can_continue"]:
    raise RuntimeError("DETENER: conflicto relevante/incierto o versiones incorrectas; revisar el JSON.")
print("Stack exacto verificado; PyTorch intacto. Conflictos externos registrados, si los hubo.")
```

`pip check` comprueba **todo el entorno**, incluidos paquetes preinstalados de Colab
que este experimento no utiliza. En nuestra ejecución notificó estos conflictos:

- `ipython 7.34.0` requiere `jedi`, que no estaba instalado.
- `diffusers 0.40.0` requiere `huggingface-hub >=1.23`, incompatible con el
  `huggingface-hub 0.36.2` fijado para BETO.
- `gradio 6.27.0` requiere `huggingface-hub >=1.16`, también incompatible con
  esa versión fijada.

No ignorar indiscriminadamente un código de salida distinto de cero. La celda
anterior captura `stdout`, `stderr` y `returncode` sin abortar automáticamente,
los imprime y los conserva en `/content/beto_pip_check.json`. Clasifica cada
conflicto por el **paquete que declara el requisito roto**, normalizando su nombre,
y comprueba que su versión coincide con los metadatos instalados:

- **Bloqueante para el experimento:** un paquete del stack BETO/datos, o una
  dependencia transitiva necesaria para su ejecución, tiene un requisito ausente
  o incompatible. También bloquean versiones HF distintas de las fijadas,
  cambios inesperados de PyTorch y fallos de importación o de GPU/CUDA. No avanzar
  al smoke ni ajustar versiones automáticamente; explicar la incompatibilidad.
- **Ajeno al experimento:** el requisito incumplido pertenece a un paquete
  presente antes de la instalación y fuera del stack y de su árbol de dependencias
  activas. No hay una lista de tres paquetes ignorados: la clasificación deriva
  del inventario previo y de los metadatos de dependencias, incluidos markers y
  extras requeridos. Solo continuar si todos los conflictos son externos, o no
  hay conflictos, y las versiones del stack coinciden exactamente. Deben pasar
  además las comprobaciones de importación y runtime tras reiniciar.
- **Origen o impacto incierto:** detenerse para investigarlo; no asumir que es
  inocuo por proceder del entorno preinstalado. Una línea con formato desconocido,
  `stderr` no vacío, un código de salida inesperado o metadatos no verificables
  también detienen la celda.

Las versiones exactas comprobadas son las cuatro HF fijadas y las cuatro de datos
ya indicadas en `data_defaults`; PyTorch debe conservar exactamente la versión de
la inspección inicial, sin imponer una build CUDA distinta. Un desajuste se registra
y bloquea la continuación, sin reinstalar ni cambiar versiones automáticamente.

Importa quién **exige** la dependencia: los conflictos de diffusers y gradio
mencionan `huggingface-hub`, que sí pertenece al stack, pero sus requisitos
corresponden a consumidores que no se importan ni se utilizan aquí. No justifican
actualizar el Hub fijado. El requisito de jedi pertenece al entorno interactivo de
IPython, no al proceso de entrenamiento; en esta ejecución no impidió ejecutar las
celdas ni el smoke. Si afectara al funcionamiento de la sesión, habría que detenerse
por ese problema técnico. Esta clasificación específica no declara compatible
todo Colab ni autoriza omitir conflictos nuevos.

Tras revisar los conflictos, **reiniciar la sesión de Python** para evitar módulos
HF antiguos en memoria, sin borrar la VM. Después continuar desde el bloque
siguiente; no ejecutar todavía el smoke. Conservar la salida de `pip check` y su
clasificación junto al registro técnico de la ejecución.

## 4. Verificar dependencias tras reiniciar

```python
import os, json, sys
from pathlib import Path
from importlib.metadata import version
from packaging.version import Version

REPO = Path("/content/pln-sentimiento-estudiantil")
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
preflight = json.loads(Path("/content/beto_colab_preflight.json").read_text())
expected_hf = {"transformers": "4.57.1", "tokenizers": "0.22.2",
               "huggingface-hub": "0.36.2", "accelerate": "1.10.1"}
if any(version(name) != expected for name, expected in expected_hf.items()):
    raise RuntimeError("DETENER: versiones HF diferentes de las acordadas.")
if version("torch") != preflight["versions_before"]["torch"]:
    raise RuntimeError("DETENER: PyTorch cambió respecto a la inspección inicial.")
from scripts.transformer import detect_runtime, load_config
runtime = detect_runtime("cuda")
config = load_config()
if runtime["device"] != "cuda" or len(runtime["gpu"]) != 1:
    raise RuntimeError("DETENER: se necesita una GPU CUDA utilizable.")
if config["max_length"] != 128:
    raise RuntimeError("DETENER: max_length distinto del fijado en Fase 1C.")
precision = ("bf16" if runtime["precision_support"]["bf16"] else
             "fp16" if runtime["precision_support"]["fp16"] else "fp32")
precision_reason = {"bf16": "soporte nativo bf16 comprobado por PyTorch",
                    "fp16": "fp16 compatible; no hay bf16 nativo",
                    "fp32": "no se habilita mixed precision con las capacidades detectadas"}[precision]
print(json.dumps(runtime, indent=2))
print("Accelerate:", version("accelerate"))
print("Precisión técnica:", precision, "—", precision_reason)
```

La preferencia fija es **bf16 nativo → fp16 compatible → fp32**. No se prueban
precisiones para comparar métricas. T4 normalmente conduce a fp16; el resultado
efectivo se deriva siempre de `detect_runtime()`, no del nombre supuesto de la GPU.
El dispositivo, batch y precisión no se escriben ahora en `configs/transformer.json`.

## 5. Transferir los cuatro archivos privados y verificar Fase 0

Subir desde el equipo local exactamente `train.csv`, `validation.csv`, `test.csv`
y `metadata.json`, procedentes de `data/processed/`. La transferencia se realiza
directamente al runtime privado de Colab. No se necesita Drive montado ni enlace
público, y no se suben los datos al repositorio GitHub. No convertir ni volver a
guardar los CSV con pandas: preservar sus bytes y hashes.

```python
import os, tempfile, hashlib
from google.colab import files
from scripts.fase0 import cargar_particiones

expected_files = {"train.csv", "validation.csv", "test.csv", "metadata.json"}
with tempfile.TemporaryDirectory(prefix="fase0_upload_", dir="/content") as staging:
    try:
        os.chdir(staging)  # files.upload no deja CSV privados en la raíz del repositorio.
        uploaded = files.upload()
    finally:
        os.chdir(REPO)
    if set(uploaded) != expected_files:
        raise RuntimeError("Seleccionar conjuntamente los cuatro artefactos exactos de Fase 0.")
    destination = REPO / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in uploaded.items():
        target = destination / name
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(f"DETENER: {name} ya existe con otros bytes; no se sobrescribe.")
    for name, content in uploaded.items():
        target = destination / name
        if not target.exists():
            target.write_bytes(content)

partitions = cargar_particiones()
expected_sizes = {"train": 16186, "validation": 3468, "test": 3469}
if {name: len(frame) for name, frame in partitions.items()} != expected_sizes:
    raise RuntimeError("Tamaños de particiones diferentes.")
print("Integridad Fase 0 y tamaños verificados:", expected_sizes)
del partitions, uploaded  # No explorar ejemplos ni utilizar test para decisiones.
data_hashes = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
               for name in sorted(expected_files)}
config_hash = hashlib.sha256((REPO / "configs/transformer.json").read_bytes()).hexdigest()
```

`cargar_particiones()` verifica hashes, esquema, etiquetas y separación de los
splits. La lectura de test es únicamente para integridad de esta interfaz. No se
llama a la verificación del modelo baseline. Los cuatro archivos deben volver a
transferirse si Colab elimina la VM. Compartir una libreta comparte sus salidas,
pero no los archivos privados de la VM: no imprimir ejemplos del corpus.

## 6. Ejecutar solo el smoke; fallback exclusivamente ante CUDA OOM

El comando exacto, desde la raíz, es el siguiente (sustituir `PRECISION` por la
decisión técnica bf16/fp16/fp32 del bloque 4):

```bash
python -m scripts.transformer --smoke-test --learning-rate 2e-5 --physical-batch-size 16 --precision PRECISION --device cuda
```

`2e-5` es un valor técnico fijo para el smoke, ya incluido en el protocolo; no es
un learning rate seleccionado por resultados y no modifica la lista futura.
La ejecución siguiente utiliza el mismo comando con el intérprete real de Colab.
Cada intento es un **proceso nuevo**, lo que libera sus objetos CUDA al finalizar.
La precisión y learning rate no cambian al reintentar. No cargar modelos en otras
celdas antes del smoke.

```python
import subprocess, sys, json, re, hashlib, uuid
from pathlib import Path
from importlib.metadata import version

record_dir = REPO / "results/transformer/colab" / f"TECH_{uuid.uuid4().hex[:12]}"
record_dir.mkdir(parents=True, exist_ok=False)
attempts = []
existing_results = set((REPO / "results/transformer/smoke").glob("SMOKE_*/smoke_test_NO_OFICIAL.json"))
for physical in (16, 8):
    command = [sys.executable, "-m", "scripts.transformer", "--smoke-test",
               "--learning-rate", "2e-5", "--physical-batch-size", str(physical),
               "--precision", precision, "--device", "cuda"]
    print("Ejecutando únicamente smoke:", command)
    completed = subprocess.run(command, cwd=REPO, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    log_path = record_dir / f"smoke_batch_{physical}.log"
    log_path.write_text(completed.stdout)
    oom = bool(re.search(r"(?m)^(?:(?:torch(?:\.cuda)?\.)?OutOfMemoryError|RuntimeError): CUDA out of memory",
                         completed.stdout))
    attempts.append({"physical_batch_size": physical, "gradient_accumulation_steps": 16 // physical,
                     "command": command, "returncode": completed.returncode,
                     "cuda_oom": oom, "log": log_path.name})
    (record_dir / "attempts.json").write_text(json.dumps(attempts, indent=2))
    if completed.returncode == 0:
        break
    if physical == 16 and oom:
        print("CUDA OOM confirmado: reintento técnico con físico 8 / acumulación 2.")
        continue
    raise RuntimeError(f"DETENER: smoke falló. Revisar {log_path}; no cambiar hiperparámetros.")

new_results = set((REPO / "results/transformer/smoke").glob("SMOKE_*/smoke_test_NO_OFICIAL.json")) - existing_results
if len(new_results) != 1:
    raise RuntimeError("DETENER: no hay exactamente un resultado técnico exitoso nuevo.")
result_path = new_results.pop()
result = json.loads(result_path.read_text())
manifest_path = REPO / "models/transformer/smoke" / result["run_id"] / "run_manifest.json"
manifest = json.loads(manifest_path.read_text())
arguments = manifest["training_arguments"]
if not (result["official"] is False and result["test_used"] is False
        and result["train_records"] == 32 and result["validation_records"] == 16
        and result["optimizer_steps"] == 2 and result["parameter_update_verified"] is True
        and result["save_reload_verified"] is True and manifest["official"] is False
        and set(manifest["data"]) == {"train", "validation"}
        and manifest["data"]["train"]["records"] == 32
        and manifest["data"]["validation"]["records"] == 16
        and manifest["configuration"] == config
        and manifest["initial_checkpoint"] == {"model_id": config["model_id"], "revision": config["model_revision"]}
        and arguments["per_device_train_batch_size"] == attempts[-1]["physical_batch_size"]
        and arguments["gradient_accumulation_steps"] == attempts[-1]["gradient_accumulation_steps"]
        and arguments["max_steps"] == 2 and arguments["learning_rate"] == 2e-5):
    raise RuntimeError("DETENER: el resultado no cumple el contrato del smoke.")
if any(hashlib.sha256((REPO / "data/processed" / name).read_bytes()).hexdigest() != digest
       for name, digest in data_hashes.items()):
    raise RuntimeError("DETENER: cambió un artefacto de datos.")
if hashlib.sha256((REPO / "configs/transformer.json").read_bytes()).hexdigest() != config_hash:
    raise RuntimeError("DETENER: cambió la configuración metodológica.")
record = {"purpose": "COLAB_SMOKE_NO_OFICIAL", "official": False,
          "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
          "preflight": preflight, "runtime_after": runtime,
          "versions_after": {name: version(name) for name in preflight["versions_before"]},
          "precision": precision, "precision_reason": precision_reason,
          "batch_rule": "16/1; solo ante CUDA OOM, 8/2; nunca por F1",
          "attempts": attempts, "configuration_sha256": config_hash, "data_sha256": data_hashes,
          "smoke_result": str(result_path.relative_to(REPO)), "test_predictively_used": False}
(record_dir / "technical_record_NO_OFICIAL.json").write_text(json.dumps(record, indent=2))
(record_dir / "smoke_test_NO_OFICIAL.json").write_bytes(result_path.read_bytes())
(record_dir / "run_manifest.json").write_bytes(manifest_path.read_bytes())
print("Smoke técnico correcto. Registro:", record_dir)
```

Si batch 16 funciona, **no probar batch 8**. Si batch 8 también produce OOM,
detenerse: no reducir max_length, no alterar batch efectivo, no cambiar LR ni
comparar F1 para decidir batch. Los directorios parciales del intento fallido se
conservan separados; el reintento usa un nuevo ID y BETO original, no su checkpoint.
Un error de descarga, dependencias, recarga o métrica no autoriza el fallback.

### Comprobación técnica de save/reload y corrección del wrapper AMP

En la primera ejecución real, la comparación anterior con `torch.allclose(..., rtol=1e-4,
atol=1e-4)` falló con `La recarga técnica no conserva resultados de validation`.
Ambos modelos estaban en `eval()` y recibían el mismo batch en el mismo dispositivo.
La asimetría estaba en el forward: el modelo de Trainer conservaba el wrapper AMP
de Accelerate, que calculaba bajo autocast fp16 y convertía las salidas a fp32,
mientras que el modelo recargado calculaba directamente en fp32. `eval()` y
`no_grad()` no retiran ese wrapper; el dtype visible de los logits tampoco revela
por sí solo la precisión interna del cálculo.

La comprobación corregida de `scripts/transformer.py`:

- Compara exactamente claves, formas, tipos y valores del `state_dict`, incluidos
  buffers; no aplica tolerancia a los pesos.
- Mide `max_abs_diff` y `mean_abs_diff` de logits antes de retirar el wrapper.
- Para calcular diferencias y `allclose`, normaliza copias de ambos logits a
  float32, conservando los dtypes originales en el diagnóstico y sin modificar
  los outputs. Así también puede diagnosticar BF16 frente a Float sin abortar
  por una incompatibilidad de dtype.
- Utiliza `accelerator.unwrap_model(..., keep_fp32_wrapper=False,
  keep_torch_compile=False)` y repite ambos forwards finales en fp32, `eval()`,
  sobre los mismos tensores y dispositivo, con autocast desactivado.
- Mantiene `rtol=1e-4` y `atol=1e-4`. No amplía la tolerancia para conseguir éxito.
- Solo establece `save_reload_verified=true` si el estado coincide exactamente,
  los estados flotantes y ambos outputs finales originales son fp32, ambas
  pérdidas y logits son finitos y la comparación bajo condiciones equivalentes pasa.

La precisión fp32 corresponde **solo a la verificación final de serialización**;
entrenamiento y evaluación mantienen la precisión técnica elegida, fp16 en esta
ejecución. El wrapper se retira al terminar y ese Trainer no se reutiliza. No se
modifican modelo, revisión, datos, max_length, batch efectivo ni hiperparámetros.

El diagnóstico se imprime en el log y se guarda en
`models/transformer/smoke/<run_id>/save_reload_diagnostics_NO_OFICIAL.json`.
Incluye las diferencias agregadas antes y después de retirar el wrapper, tipos,
comprobación del estado, tolerancias y flag de verificación. Se conserva también
si la comprobación falla; no contiene logits crudos ni predicciones. En caso de
éxito también queda incorporado al resultado `smoke_test_NO_OFICIAL.json`.
No disponemos de los logits de la primera ejecución fallida para cuantificar su
diferencia retrospectivamente. La ejecución final conserva el diagnóstico agregado;
esta guía no inventa magnitudes numéricas que no se hayan aportado.

El fallo anterior deja el manifiesto, el checkpoint y `smoke_saved_model` en su
directorio de modelos, pero no crea el resultado final de éxito. Conservar los
artefactos parciales para trazabilidad; una nueva ejecución usa un nuevo
`SMOKE_<uuid>` y BETO original, sin reanudar ese checkpoint. No probar batch 8:
el intento real con batch 16 no tuvo OOM.

La ejecución final `SMOKE_8b466ec8bf40` ya verificó esta corrección en Colab con el
commit probado indicado al comienzo. Para reproducirla, utilizar ese mismo commit,
que incluye las correcciones de AMP y de comparación entre dtypes, y ejecutar
únicamente el smoke con el protocolo fijado. Una reproducción tendrá IDs nuevos;
los identificadores de esta guía corresponden al registro histórico final.

## 7. Descargar el registro técnico pequeño

```python
import zipfile
from google.colab import files

archive = Path("/content") / f"{record_dir.name}_NO_OFICIAL.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(record_dir.iterdir()):
        if path.is_file():
            bundle.write(path, arcname=path.name)
files.download(str(archive))
```

El ZIP contiene solo JSON técnicos, manifiesto y logs. No incluye CSV, textos,
predicciones, pesos ni checkpoints. Guardarlo de forma privada: el manifiesto
incluye índices originales para trazabilidad. Los resultados/modelos permanecen
ignorados por Git. Si el smoke falla, el directorio técnico conserva logs e
`attempts.json`: descargar esos archivos antes de perder la VM.
Si falla save/reload, conservar además
`models/transformer/smoke/<run_id>/save_reload_diagnostics_NO_OFICIAL.json`.
Ese archivo separado no se añade automáticamente al ZIP de la celda anterior;
sus contenidos sí se incluyen en el log y, si hay éxito, en el JSON de resultado.

## Evidencia esperada y límites

- Resultado `SMOKE_TEST_NO_OFICIAL`, `official=false`, 32 train / 16 validation,
  dos pasos, `parameter_update_verified=true`, `save_reload_verified=true`.
- Carga inicial desde BETO original y revisión fijada en el manifiesto; tokens a
  128 con padding dinámico; forward/backward y evaluación a través del smoke de 2A.
- Checkpoint, guardado y recarga de ese mismo smoke. Sus métricas de validation
  son técnicas; no existe un umbral de F1 para declarar el smoke correcto.
- Recarga respaldada por igualdad exacta de `state_dict` y comparación de logits
  bajo condiciones numéricas equivalentes; conservar el diagnóstico de diferencias.
- `load_training_frames()` descarta test antes de convertir datos. La clase Trainer
  protege `train`, `evaluate`, `predict` y los dataloaders. El manifiesto enumera
  únicamente train/validation; los hashes de los cuatro artefactos no cambian.
- El flag `test_used=false` es una declaración complementaria: la evidencia se
  apoya también en el código fijado, los controles de procedencia y los tests
  `tests/test_transformer.py`. No inferir ausencia de uso solo por el flag.
- Persistir commit, configuración/hashes, revisión BETO, versiones, driver/CUDA,
  GPU/VRAM inicial, precisión y su razón, batch/acumulación e intento OOM si lo hubo.

Quedan riesgos reales de disponibilidad/terminación de Colab, incompatibilidades
del entorno preinstalado, red/caché del Hub y OOM. El resolvedor de pip no valida
todos los programas preinstalados: `pip check` y las comprobaciones tras reiniciar
deben revisarse antes del smoke. El stack necesario y el runtime deben pasar sus
controles; un `pip check` global no nulo solo permite continuar tras documentar que
todos los conflictos son ajenos al experimento según el criterio de la sección 3.
Si hay un conflicto relevante o incierto, conservar el diagnóstico y detenerse;
no sustituir versiones ni modificar el protocolo automáticamente.

Esta guía registra el primer intento fallido y el **smoke final real verificado
en Colab**, con actualización de parámetros, save/reload y `state_dict` comprobados.
Fase 2B queda validada técnicamente; sus métricas siguen siendo NO OFICIALES y no
habilitan selección de hiperparámetros ni ejecución automática de T1/T2/T3.
No ejecutar experimentos oficiales ni realizar commits desde Colab en esta fase.
