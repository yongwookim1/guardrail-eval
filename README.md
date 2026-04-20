# guardrail-eval

Unified evaluation harness for multimodal guardrail models.

Currently wired for:

| Model | HF id | Modality |
| --- | --- | --- |
| `nemotron_cs` | `nvidia/Nemotron-3-Content-Safety` | text + image |
| `llama_guard_4` | `meta-llama/Llama-Guard-4-12B` | text + image |

on benchmarks:

| Benchmark | HF id | All samples |
| --- | --- | --- |
| `siuo` | `MMInstruction/SIUO` | expected `unsafe` |
| `vlsbench` | `Foreverlasting1202/VLSBench` | expected `unsafe` |

## Setup

```bash
conda env create -f environment.yml
conda activate guardrail-eval
pip install -e .
huggingface-cli login   # for gated datasets / models
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

- `raw.jsonl` — one record per sample with prediction, raw model output, latency
- `summary.json` — `unsafe_recall`, `error_rate`, per-category breakdown
- `config.json` — frozen copy of the model + benchmark YAMLs used

## Adding a new model

1. Subclass `GuardrailModel` in `src/guardrail_eval/models/<name>.py`. Implement
   `classify_batch(samples) -> list[Verdict]`. Decorate with `@register_model("<name>")`.
2. Drop a YAML at `configs/models/<name>.yaml` pointing to the class and its HF id.
3. Import the module in `src/guardrail_eval/models/__init__.py`.

That's the whole contract — the evaluator, CLI, and metrics pick it up automatically.

## Adding a new benchmark

Mirror of the model path: subclass `Benchmark`, `@register_benchmark`, add a YAML,
import in `benchmarks/__init__.py`. For simple HF-dataset benchmarks you usually
only need a YAML + a 10-line loader delegating to `_hf_common.iter_hf_samples`.

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

- **Nemotron-3-Content-Safety** (4B, Gemma-3 base). Requires `request_categories`
  chat-template arg; we pass `"/categories"` so `Safety Categories: ...` is emitted.
- **Llama-Guard-4-12B** (Llama-4 early-fusion). HF card documents only the
  `transformers` path, but the architecture is vLLM-supported. If vLLM fails to
  load this model, implement a `transformers` backend and switch `backend: vllm`
  in `configs/models/llama_guard_4.yaml`.

## Tests

```bash
pytest
```

Parser and metric tests do not import vLLM or download any weights.
