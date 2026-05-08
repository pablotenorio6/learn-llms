# Bench

Benchmark de los modelos definidos en `BENCH_MODELS` (.env).

```bash
make bench
```

Salidas:
- `results/results.jsonl` — una línea por cada run (acumulativo entre ejecuciones).
- `results/results.md` — tabla resumen de la última ejecución.

Métricas:
- **TTFT (ms)** — tiempo al primer token.
- **Gen t/s** — tokens generados por segundo.
- **Prompt t/s** — velocidad de prompt processing.
- **Eval count / Prompt count** — tokens reales contados por Ollama.

Para medir VRAM/CPU en paralelo (otra terminal):

```bash
watch -n 0.5 nvidia-smi
docker stats llmops-ollama
```

Apunta los picos a mano en una nota junto al resultado.
