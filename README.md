# guardrail-eval

Unified evaluation harness for multimodal guardrail models.

Currently wired for:

| Model | Local path | Modality |
| --- | --- | --- |
| `nemotron_cs` | `models/Nemotron-3-Content-Safety` | text + image |
| `llama_guard_4` | `models/Llama-Guard-4-12B` | text + image |

on benchmarks:

| Benchmark | Local path | Size | All samples |
| --- | --- | --- | --- |
| `siuo` | `datasets/SIUO` | 168 | expected `unsafe` |
| `vlsbench` | `datasets/vlsbench` | 2,241 | expected `unsafe` |

Neither dataset is a standard parquet-with-Image-feature. Both are loaded from
local dataset directories in dataset-specific loaders:

- **SIUO** — read `datasets/SIUO/siuo_gen.json` and open images from `datasets/SIUO/images/`.
- **VLSBench** — read `datasets/vlsbench/data.json` + `datasets/vlsbench/imgs.tar`, then extract the tar once into `datasets/vlsbench/imgs/`.

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
```

The bundled model YAMLs use `model_path:` and resolve relative paths from the
repo root, so no Hugging Face model download is needed at runtime.

Manually place the benchmark assets here before running:

```text
datasets/SIUO/siuo_gen.json
datasets/SIUO/images/...

datasets/vlsbench/data.json
datasets/vlsbench/imgs.tar
```

## Run

```bash
# Single pair
python scripts/run_eval.py --model nemotron_cs --benchmark siuo

# Quick smoke test
python scripts/run_eval.py --model nemotron_cs --benchmark siuo --limit 20

# Full grid (all configs under configs/models × configs/benchmarks)
python scripts/run_eval.py --model all --benchmark all
```

Per-run output lands at `results/<model>/<benchmark>/`:

- `results.json` — one JSON record per line with prediction, raw model output, latency
- `results_summary.json` — `unsafe_recall`, `error_rate`, per-category breakdown
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
│   ├── models/{base,nemotron,llama_guard,registry}.py
│   ├── benchmarks/{base,siuo,vlsbench,_hf_common,registry}.py
│   ├── evaluator.py                       # model × benchmark runner
│   ├── metrics.py                         # unsafe-recall, category breakdown
│   ├── io.py                              # JSONL / image-to-data-uri helpers
│   └── cli.py                             # `guardrail-eval ...`
├── scripts/run_eval.py
└── tests/                                 # parser + metrics unit tests (no vLLM needed)
```

## Notes on the models

- **Nemotron-3-Content-Safety** (4B, Gemma-3 base). This repo uses a
  `transformers` backend by default and passes `request_categories="/categories"`
  through the chat template so `Safety Categories: ...` is emitted.
- **Llama-Guard-4-12B** (Llama-4 early-fusion). This repo uses a
  `transformers` backend by default because the model card documents that path
  directly and it is more reliable than the native vLLM model path for this
  setup.

## Tests

```bash
pytest
```

Parser and metric tests do not import vLLM or download any weights.
