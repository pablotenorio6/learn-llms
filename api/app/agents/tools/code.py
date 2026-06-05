"""Tool python_exec: ejecuta Python en un subproceso aislado.

Aislamiento (en Linux, que es donde corre el contenedor api):
  - Subproceso `python -I` (modo aislado: ignora env vars de Python, no mete
    cwd en sys.path, no usa el site del usuario).
  - rlimits vía preexec_fn: memoria (RLIMIT_DATA), CPU (RLIMIT_CPU), tamaño de
    ficheros (RLIMIT_FSIZE) y nº de descriptores (RLIMIT_NOFILE).
  - Timeout wall-clock: si se pasa, se mata todo el grupo de procesos (SIGKILL).
  - Env mínimo: NO se heredan las variables del contenedor, así el código
    ejecutado no puede leer secretos (OPENAI_API_KEY, LITELLM_MASTER_KEY, ...).
  - cwd = directorio temporal efímero, borrado al terminar.

Lo que esto NO es: un sandbox de seguridad contra código adversarial. El
subproceso corre como el mismo usuario del contenedor y puede leer/escribir el
filesystem del contenedor; el "blast radius" es el propio contenedor api
(efímero y reconstruible). Contención fuerte exigiría network/mount namespaces
o un contenedor descartable por ejecución (la opción que descartamos por peso).
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import signal
import sys
import tempfile

from app.agents.registry import tool
from app.config import get_settings

log = logging.getLogger(__name__)

try:
    import resource  # POSIX-only
except ImportError:  # pragma: no cover - Windows (los tests reales corren en el contenedor Linux)
    resource = None  # type: ignore[assignment]


def _make_preexec(timeout: float, max_memory_mb: int):
    if resource is None:
        return None

    def _limits() -> None:
        mem = max_memory_mb * 1024 * 1024
        # RLIMIT_DATA (heap/brk real), NO RLIMIT_AS (address space virtual):
        # numpy/BLAS reservan arenas de VIRT enormes al importarse y RLIMIT_AS
        # las mata aunque la RAM usada sea mínima. RLIMIT_DATA limita lo que de
        # verdad consume y deja importar numpy/pandas.
        resource.setrlimit(resource.RLIMIT_DATA, (mem, mem))
        cpu = int(math.ceil(timeout)) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        fsize = 16 * 1024 * 1024  # 16 MB por fichero
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return _limits


@tool
async def python_exec(code: str) -> dict:
    """Ejecuta un fragmento de código PYTHON y devuelve su stdout/stderr. ÚSALA para cálculos complejos con varios pasos, manipulación de datos/strings, simulaciones, generar o transformar texto programáticamente, o cualquier tarea que se resuelva mejor con código que razonando a mano. Imprime tus resultados con print() — solo se captura lo que escribas a stdout/stderr. Tienes la librería estándar de Python más pandas y numpy (impórtalos como `pd`/`np`) para dataframes y cálculo numérico. NO hay acceso a internet ni otros paquetes externos. El proceso tiene límite de tiempo y memoria y se mata si los excede. Para mera aritmética usa `calculator`, que es más directa.

    Args:
        code: Código Python a ejecutar. Usa print() para devolver resultados.
    """
    settings = get_settings()
    src = code or ""
    if not src.strip():
        return {"error": "código vacío"}

    workdir = tempfile.mkdtemp(prefix="pyexec_")
    script = os.path.join(workdir, "snippet.py")
    try:
        with open(script, "w", encoding="utf-8") as f:
            f.write(src)

        # Env mínimo: sin secretos del contenedor. PATH básico para que arranque.
        # Capamos los hilos de BLAS/OpenMP a 1: con RLIMIT_DATA apretado, los
        # thread pools de numpy reservan un arena por hilo y pueden chocar con el
        # límite; además un snippet de un solo paso no gana nada paralelizando.
        clean_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HOME": workdir,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-I", script,
                cwd=workdir,
                env=clean_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,  # propio grupo de procesos → matamos el árbol
                preexec_fn=_make_preexec(settings.python_exec_timeout, settings.python_exec_max_memory_mb),
            )
        except Exception as e:  # pragma: no cover
            log.warning("python_exec.spawn_failed", extra={"err": str(e)})
            return {"error": f"no se pudo lanzar el subproceso: {e}"}

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=settings.python_exec_timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                proc.kill()
            stdout_b, stderr_b = await proc.communicate()

        def _decode(b: bytes) -> tuple[str, bool]:
            text = (b or b"").decode("utf-8", errors="replace")
            if len(text) > settings.python_exec_max_output:
                return text[: settings.python_exec_max_output], True
            return text, False

        stdout, out_trunc = _decode(stdout_b)
        stderr, err_trunc = _decode(stderr_b)

        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": None if timed_out else proc.returncode,
            "timed_out": timed_out,
            "truncated": out_trunc or err_trunc,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
