# guardrail-eval

Unified evaluation harness for multimodal guardrail models.

Currently wired for:

| Model | Local path | Modality |
| --- | --- | --- |
| `nemotron_cs` | `models/Nemotron-3-Content-Safety` | text + image |
| `llama_guard_4` | `models/Llama-Guard-4-12B` | text + image |
| `gemma_3_4b_it` | `models/gemma-3-4b-it` | text + image |

on benchmarks:

| Benchmark | Local path | Size | All samples |
| --- | --- | --- | --- |
| `siuo` | `datasets/SIUO` | 168 | expected `unsafe` |
| `vlsbench` | `datasets/vlsbench` | 2,241 | expected `unsafe` |
| `holisafe` | `datasets/holisafe-bench` | 4,031 | mixed `safe` / `unsafe` from `type` |

None of these datasets is loaded through the standard parquet-with-Image path
in this repo. They are loaded from local dataset directories in
dataset-specific loaders:

- **SIUO** — read `datasets/SIUO/siuo_gen.json` and open images from `datasets/SIUO/images/`.
- **VLSBench** — read `datasets/vlsbench/data.json` + `datasets/vlsbench/imgs.tar`, then extract the tar once into `datasets/vlsbench/imgs/`.
- **HoliSafe** — read `datasets/holisafe-bench/holisafe_bench.json` and open images from `datasets/holisafe-bench/images/`; expected labels are derived from the final character of HoliSafe's `type` code (`S` => `safe`, `U` => `unsafe`).

## Setup

```bash
conda env create -f environment.yml
conda activate guardrail-eval
pip install -e .
```

Manually place the model directories here before running:

```text
models/Nemotron-3-Content-Safety
models/Llama-Guard-4-12B
models/gemma-3-4b-it
```

The bundled model YAMLs use `model_path:` and resolve relative paths from the
repo root, so no Hugging Face model download is needed at runtime.

Manually place the benchmark assets here before running:

```text
datasets/SIUO/siuo_gen.json
datasets/SIUO/images/...

datasets/vlsbench/data.json
datasets/vlsbench/imgs.tar

datasets/holisafe-bench/holisafe_bench.json
datasets/holisafe-bench/images/...
```

## Run

```bash
# Single pair
python scripts/run_eval.py --model nemotron_cs --benchmark siuo

# Quick smoke test
python scripts/run_eval.py --model nemotron_cs --benchmark siuo --limit 20

# Larger runs: flush less often to reduce results I/O overhead
python scripts/run_eval.py --model nemotron_cs --benchmark holisafe --batch-size 32 --flush-every-batches 32

# Pure multimodal base-model comparison with a frozen binary safety prompt
python scripts/run_eval.py --model gemma_3_4b_it --benchmark holisafe --limit 20

# Full grid (all configs under configs/models × configs/benchmarks)
python scripts/run_eval.py --model all --benchmark all
```

Notes:

- The `transformers` backend now batches chat-template preprocessing and generation across each evaluator batch.
- `backend_kwargs.use_cache` defaults to `true` for the `transformers` backend and can be disabled in model YAML if needed.
- Image encoding caches default to 1024 entries. Override with `GUARDRAIL_EVAL_IMAGE_CACHE_MAXSIZE=<n>` when tuning RAM usage.

Per-run output lands at `results/<model>/<benchmark>/`:

- `results.jsonl` — one JSON record per line with prediction, raw model output, latency
- `results_summary.json` — accuracy, safe/unsafe precision-recall-F1, balanced accuracy, confusion counts, legacy unsafe-only fields, per-category breakdown, and HoliSafe per-`type` breakdown
- `config.json` — frozen copy of the model + benchmark YAMLs used

## Adding a new model

1. Subclass `GuardrailModel` in `src/guardrail_eval/models/<name>.py`. Implement
   `classify_batch(samples) -> list[Verdict]`. Decorate with `@register_model("<name>")`.
2. Drop a YAML at `configs/models/<name>.yaml` pointing to the class and a local
   `model_path:`.
3. Import the module in `src/guardrail_eval/models/__init__.py`.

That's the whole contract — the evaluator, CLI, and metrics pick it up automatically.

## Adding a new benchmark

Mirror of the model path: subclass `Benchmark` (or `LocalFileBenchmark` from
`_hf_common.py` if the repo stores JSON metadata + image files), decorate with
`@register_benchmark`, add a YAML, import in `benchmarks/__init__.py`. An
`LocalFileBenchmark` subclass is typically two methods: `_prepare()` (resolve local
files + return `(records, image_root)`) and `_record_to_sample()` (map one record
to a `Sample`). Everything else — tqdm totals, streaming, error handling —
is handled by the base class.

## Layout

```
guardrail-eval/
├── configs/{models,benchmarks}/*.yaml     # one file per model / benchmark
├── src/guardrail_eval/
│   ├── types.py                           # Sample, Verdict
│   ├── backends/vllm_backend.py           # vLLM engine wrapper (multimodal chat)
│   ├── models/{base,gemma_3_it,nemotron,llama_guard,registry}.py
│   ├── benchmarks/{base,holisafe,siuo,vlsbench,_hf_common,registry}.py
│   ├── evaluator.py                       # model × benchmark runner
│   ├── metrics.py                         # accuracy / recall / confusion summaries
│   ├── io.py                              # JSONL / image-to-data-uri helpers
│   └── cli.py                             # `guardrail-eval ...`
└── scripts/run_eval.py
```

## Notes on the models

- **Nemotron-3-Content-Safety** (4B, Gemma-3 base). This repo uses a
  `transformers` backend by default and passes `request_categories="/categories"`
  through the chat template so `Safety Categories: ...` is emitted.
- **Llama-Guard-4-12B** (Llama-4 early-fusion). This repo uses a
  `transformers` backend by default because the model card documents that path
  directly and it is more reliable than the native vLLM model path for this
  setup.
- **Gemma-3-4B-IT** is treated here as a prompted binary classifier for clean
  `safe` / `unsafe` comparison against dedicated guardrail models.
