"""Benchmark de latencia y throughput contra LiteLLM proxy.

Mide, vía streaming OpenAI-compat:
- Time-to-first-token (TTFT)
- Tokens/segundo en generación
- Latencia total

A través del proxy, los aliases son los declarados en litellm-config.yaml
(p.ej. `llama-local`, `qwen-local`, `gpt-4o-mini`). Esto permite comparar
local vs cloud con la misma herramienta.

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

from openai import APIError, AsyncOpenAI

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
    prompt_tokens: int
    completion_tokens: int
    gen_tokens_per_s: float
    output_chars: int


async def _stream_run(client: AsyncOpenAI, model: str, prompt: str, label: str) -> RunResult:
    start = time.perf_counter()
    ttft_ms: float | None = None
    output_parts: list[str] = []
    prompt_tokens = completion_tokens = 0

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
        stream=True,
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if not chunk.choices and getattr(chunk, "usage", None):
            usage = chunk.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            continue
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = (delta.content or "") if delta else ""
        if piece:
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            output_parts.append(piece)

    total_ms = (time.perf_counter() - start) * 1000.0
    output = "".join(output_parts)
    gen_per_s = 0.0
    if completion_tokens > 0 and total_ms > 0:
        gen_per_s = round(completion_tokens / (total_ms / 1000.0), 2)
    return RunResult(
        model=model,
        prompt_label=label,
        ttft_ms=round(ttft_ms or total_ms, 2),
        total_ms=round(total_ms, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        gen_tokens_per_s=gen_per_s,
        output_chars=len(output),
    )


async def bench_model(client: AsyncOpenAI, model: str, runs_per_prompt: int = 3) -> list[RunResult]:
    print(f"\n=== {model} ===")
    print("  warmup ...", flush=True)
    try:
        await _stream_run(client, model, WARMUP_PROMPT, "warmup")
    except APIError as e:
        print(f"  ERROR warmup: {e}; ¿alias declarado en litellm-config.yaml? ¿API key configurada?")
        return []

    out: list[RunResult] = []
    for label, prompt in (("short", PROMPT_SHORT), ("long", PROMPT_LONG)):
        print(f"  prompt={label} (x{runs_per_prompt}) ...", flush=True)
        for i in range(runs_per_prompt):
            try:
                r = await _stream_run(client, model, prompt, label)
            except APIError as e:
                print(f"    run={i+1} ERROR: {e}")
                continue
            out.append(r)
            print(
                f"    run={i+1} ttft={r.ttft_ms:>7.1f}ms  "
                f"gen={r.gen_tokens_per_s:>6.1f}t/s  "
                f"out={r.completion_tokens}t"
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
        summary[model] = {
            "runs": len(rs),
            "gen_tps_mean": round(statistics.mean(gen), 2) if gen else 0,
            "gen_tps_p50": round(statistics.median(gen), 2) if gen else 0,
            "ttft_ms_mean": round(statistics.mean(ttft), 2) if ttft else 0,
        }
    return summary


def _write_markdown(summary: dict[str, dict[str, float]]) -> Path:
    path = RESULTS_DIR / "results.md"
    lines = [
        f"# Benchmark — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "| Modelo | Runs | Gen t/s (avg) | Gen t/s (p50) | TTFT ms (avg) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, s in sorted(summary.items()):
        lines.append(
            f"| `{model}` | {int(s['runs'])} | {s['gen_tps_mean']} | "
            f"{s['gen_tps_p50']} | {s['ttft_ms_mean']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def main():
    settings = get_settings()
    models = settings.bench_models_list
    print(f"Models: {models}")
    print(f"LiteLLM proxy: {settings.litellm_base_url}")

    client = AsyncOpenAI(
        base_url=settings.litellm_base_url.rstrip("/") + "/v1",
        api_key=settings.litellm_master_key,
        timeout=600,
        max_retries=0,
    )
    all_results: list[RunResult] = []
    try:
        for m in models:
            all_results.extend(await bench_model(client, m))
    finally:
        await client.close()

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
            f"ttft={s['ttft_ms_mean']:>7.1f}ms"
        )
    print(f"\nWrote {md_path}")
    print(f"Wrote {jsonl_path}")


if __name__ == "__main__":
    asyncio.run(main())
