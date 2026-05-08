"""Benchmark de latencia y throughput para los modelos del .env (BENCH_MODELS).

Mide:
- Time-to-first-token (TTFT)
- Tokens/segundo en generación
- Tokens/segundo en prompt processing
- Latencia total
- Tamaño y digest del modelo

Lanza el script desde el contenedor de la API:
    docker compose exec api python -m bench.benchmark

Resultados:
- bench/results/results.jsonl  (una línea por run)
- bench/results/results.md     (tabla resumen)
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.config import get_settings

PROMPT_SHORT = "Escribe un haiku sobre el otoño."
PROMPT_LONG = (
    "Explica en 6-8 frases qué es la cuantización de modelos LLM, "
    "incluyendo Q4_K_M frente a Q8_0, y su efecto en VRAM y calidad."
)
WARMUP_PROMPT = "Hola"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RunResult:
    model: str
    prompt_label: str
    ttft_ms: float
    total_ms: float
    prompt_eval_count: int
    prompt_eval_ms: float
    prompt_tokens_per_s: float
    eval_count: int
    eval_ms: float
    gen_tokens_per_s: float
    output_chars: int


async def _stream_run(client: httpx.AsyncClient, model: str, prompt: str, label: str) -> RunResult:
    settings = get_settings()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 256},
        "keep_alive": settings.ollama_keep_alive,
    }

    start = time.perf_counter()
    ttft_ms: float | None = None
    output_parts: list[str] = []
    prompt_eval_count = eval_count = 0
    prompt_eval_ms = eval_ms = 0.0

    async with client.stream(
        "POST", f"{settings.ollama_host}/api/chat", json=payload, timeout=600
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line:
                continue
            obj = json.loads(line)
            piece = (obj.get("message") or {}).get("content")
            if piece:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000.0
                output_parts.append(piece)
            if obj.get("done"):
                prompt_eval_count = int(obj.get("prompt_eval_count") or 0)
                eval_count = int(obj.get("eval_count") or 0)
                # Ollama da nanosegundos
                prompt_eval_ms = float(obj.get("prompt_eval_duration") or 0) / 1e6
                eval_ms = float(obj.get("eval_duration") or 0) / 1e6

    total_ms = (time.perf_counter() - start) * 1000.0
    output = "".join(output_parts)
    return RunResult(
        model=model,
        prompt_label=label,
        ttft_ms=round(ttft_ms or total_ms, 2),
        total_ms=round(total_ms, 2),
        prompt_eval_count=prompt_eval_count,
        prompt_eval_ms=round(prompt_eval_ms, 2),
        prompt_tokens_per_s=round(
            (prompt_eval_count / (prompt_eval_ms / 1000.0)) if prompt_eval_ms > 0 else 0.0, 2
        ),
        eval_count=eval_count,
        eval_ms=round(eval_ms, 2),
        gen_tokens_per_s=round(
            (eval_count / (eval_ms / 1000.0)) if eval_ms > 0 else 0.0, 2
        ),
        output_chars=len(output),
    )


async def bench_model(client: httpx.AsyncClient, model: str, runs_per_prompt: int = 3) -> list[RunResult]:
    print(f"\n=== {model} ===")
    print("  warmup ...", flush=True)
    try:
        await _stream_run(client, model, WARMUP_PROMPT, "warmup")
    except httpx.HTTPStatusError as e:
        print(f"  ERROR warmup: {e}; ¿modelo descargado? Prueba `make pull-models`")
        return []

    out: list[RunResult] = []
    for label, prompt in (("short", PROMPT_SHORT), ("long", PROMPT_LONG)):
        print(f"  prompt={label} (x{runs_per_prompt}) ...", flush=True)
        for i in range(runs_per_prompt):
            r = await _stream_run(client, model, prompt, label)
            out.append(r)
            print(
                f"    run={i+1} ttft={r.ttft_ms:>7.1f}ms  "
                f"gen={r.gen_tokens_per_s:>6.1f}t/s  "
                f"prompt={r.prompt_tokens_per_s:>6.1f}t/s  "
                f"out={r.eval_count}t"
            )
    return out


def _summarize(results: list[RunResult]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[RunResult]] = {}
    for r in results:
        by_model.setdefault(r.model, []).append(r)

    summary: dict[str, dict[str, float]] = {}
    for model, rs in by_model.items():
        gen = [r.gen_tokens_per_s for r in rs if r.gen_tokens_per_s > 0]
        ttft = [r.ttft_ms for r in rs]
        prompt = [r.prompt_tokens_per_s for r in rs if r.prompt_tokens_per_s > 0]
        summary[model] = {
            "runs": len(rs),
            "gen_tps_mean": round(statistics.mean(gen), 2) if gen else 0,
            "gen_tps_p50": round(statistics.median(gen), 2) if gen else 0,
            "ttft_ms_mean": round(statistics.mean(ttft), 2) if ttft else 0,
            "prompt_tps_mean": round(statistics.mean(prompt), 2) if prompt else 0,
        }
    return summary


def _write_markdown(summary: dict[str, dict[str, float]]) -> Path:
    path = RESULTS_DIR / "results.md"
    lines = [
        f"# Benchmark — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo | Runs | Gen t/s (avg) | Gen t/s (p50) | TTFT ms (avg) | Prompt t/s (avg) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, s in sorted(summary.items()):
        lines.append(
            f"| `{model}` | {int(s['runs'])} | {s['gen_tps_mean']} | "
            f"{s['gen_tps_p50']} | {s['ttft_ms_mean']} | {s['prompt_tps_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main():
    settings = get_settings()
    models = settings.bench_models_list
    print(f"Models: {models}")
    print(f"Ollama host: {settings.ollama_host}")

    all_results: list[RunResult] = []
    async with httpx.AsyncClient(timeout=600) as client:
        for m in models:
            all_results.extend(await bench_model(client, m))

    # Persistir runs crudas
    jsonl_path = RESULTS_DIR / "results.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in all_results:
            f.write(json.dumps({"ts": ts, **asdict(r)}) + "\n")

    summary = _summarize(all_results)
    md_path = _write_markdown(summary)

    print("\n=== Summary ===")
    for model, s in sorted(summary.items()):
        print(
            f"  {model:<45} gen={s['gen_tps_mean']:>6.1f}t/s  "
            f"ttft={s['ttft_ms_mean']:>7.1f}ms  prompt={s['prompt_tps_mean']:>6.1f}t/s"
        )
    print(f"\nWrote {md_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())
