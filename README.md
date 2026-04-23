# guardrail-eval

Unified evaluation harness for multimodal guardrail models.

Currently wired for:

| Model | Local path | Modality |
| --- | --- | --- |
| `nemotron_cs` | `models/Nemotron-3-Content-Safety` | text + image |
| `llama_guard_4` | `models/Llama-Guard-4-12B` | text + image |
| `gemma_3_4b_it` | `models/gemma-3-4b-it` | text + image |
| `qwen2_5_vl_3b_instruct` | `models/Qwen2.5-VL-3B-Instruct` | text + image |
| `qwen2_5_omni_3b` | `models/Qwen2.5-Omni-3B` | text + image |
| `omniguard_3b` | `models/OmniGuard-3B` | text + image |
| `guardreasoner_vl_3b` | `models/GuardReasoner-VL-3B` | text + image |

on benchmarks:

| Benchmark | Local path | Size | All samples |
| --- | --- | --- | --- |
| `siuo` | `datasets/SIUO` | 168 | expected `unsafe` |
| `vlsbench` | `datasets/vlsbench` | 2,241 | expected `unsafe` |
| `holisafe` | `datasets/holisafe-bench` | 4,031 | mixed `safe` / `unsafe` from `type` |
| `mmbench` | `datasets/MMBench` | 4.3k (`validation`) | multiple-choice accuracy with circular option shifts |
| `mmmu_pro` | `datasets/MMMU_Pro` | 1,730 | multiple-choice accuracy (`standard (10 options)`) |

None of these datasets is loaded through the standard parquet-with-Image path
in this repo. They are loaded from local dataset directories in
dataset-specific loaders:

- **SIUO** — read `datasets/SIUO/siuo_gen.json` and open images from `datasets/SIUO/images/`.
- **VLSBench** — read `datasets/vlsbench/data.json` + `datasets/vlsbench/imgs.tar`, then extract the tar once into `datasets/vlsbench/imgs/`.
- **HoliSafe** — read `datasets/holisafe-bench/holisafe_bench.json` and open images from `datasets/holisafe-bench/images/`; expected labels are derived from the final character of HoliSafe's `type` code (`S` => `safe`, `U` => `unsafe`).
- **MMBench** — read local parquet files from `datasets/MMBench/`, materialize images into `datasets/MMBench/.cache/mmbench_images/`, and emit one evaluation row per circular option shift.
- **MMMU-Pro** — read local parquet files from `datasets/MMMU_Pro/standard (10 options)/` and materialize embedded images into `datasets/MMMU_Pro/.cache/mmmu_pro_images/` on first use.

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
models/Qwen2.5-VL-3B-Instruct
models/Qwen2.5-Omni-3B
models/OmniGuard-3B
models/GuardReasoner-VL-3B
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

datasets/MMBench/*.parquet

datasets/MMMU_Pro/standard (10 options)/test-*.parquet
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

# MMMU-Pro direct-answer scoring over option letters with circular option shifts
python scripts/run_eval.py --model gemma_3_4b_it --benchmark mmmu_pro --limit 20

# MMBench multiple-choice evaluation with circular option shifts
python scripts/run_eval.py --model gemma_3_4b_it --benchmark mmbench --limit 20

# Check whether semantic predictions stay stable across permutations
python scripts/check_permutation_bias.py --model gemma_3_4b_it --benchmark mmmu_pro
python scripts/check_permutation_bias.py --model gemma_3_4b_it --benchmark mmbench

# Full grid (all configs under configs/models × configs/benchmarks)
python scripts/run_eval.py --model all --benchmark all
```

Notes:

- The `transformers` backend now batches chat-template preprocessing and generation across each evaluator batch.
- The MMBench path expands each base question into one scored sample per circular option shift, so `results.jsonl` and summary counts reflect passes rather than unique questions.
- The MMMU-Pro path now also expands each base question into one scored sample per circular option shift, and can be checked with `scripts/check_permutation_bias.py` for semantic consistency across passes.
- Both MMBench and MMMU-Pro still score option letters after the full question/options context, while batching all sample-option rows in each evaluator batch.
- `backend_kwargs.use_cache` defaults to `true` for the `transformers` backend and can be disabled in model YAML if needed.
- Set `backend_kwargs.max_choice_rows` in a model YAML if you need to cap the flattened choice batch size for memory safety on long prompts.
- Qwen2.5-Omni choice configs disable audio output and leave the system prompt empty to avoid spurious audio/talker warnings during text-only benchmark scoring.
- Image encoding caches default to 1024 entries. Override with `GUARDRAIL_EVAL_IMAGE_CACHE_MAXSIZE=<n>` when tuning RAM usage.
- `--model all --benchmark all` now auto-routes each model to the right inference path for the benchmark task type instead of exposing separate user-facing model names.

Per-run output lands at `results/<model>/<benchmark>/`:

- `results.jsonl` — one JSON record per line with prediction, raw model output, latency
- `results_summary.json` — classification runs emit safe/unsafe precision-recall-F1 plus breakdowns; multiple-choice runs emit pass-level accuracy plus by-subject accuracy, and repeated-permutation benchmarks also include automatic `question_level` and `permutation_bias` analyses
- `config.json` — frozen copy of the model + benchmark YAMLs used

## Adding a new model

1. Subclass `GuardrailModel` in `src/guardrail_eval/models/<name>.py`. Implement
   `classify_batch(samples) -> list[Verdict]` for safety benchmarks or
   `score_choice_batch(samples) -> list[ChoiceVerdict]` for multiple-choice benchmarks. Set
   `task_types: [multiple_choice]` in YAML for choice-only models. Decorate with
   `@register_model("<name>")`.
2. Drop a YAML at `configs/models/<name>.yaml` pointing to the class and a local
   `model_path:`.
3. Import the module in `src/guardrail_eval/models/__init__.py`.

That's the whole contract — the evaluator, CLI, and metrics pick it up automatically.

## Adding a new benchmark

Mirror of the model path: subclass `Benchmark`, `MultipleChoiceBenchmark`, or
`LocalFileBenchmark` from `_hf_common.py`, decorate with `@register_benchmark`,
add a YAML, import in `benchmarks/__init__.py`.

## Layout

```
guardrail-eval/
├── configs/{models,benchmarks}/*.yaml     # one file per model / benchmark
├── src/guardrail_eval/
│   ├── types.py                           # Sample / Verdict / ChoiceSample / ChoiceVerdict
│   ├── backends/vllm_backend.py           # vLLM engine wrapper (multimodal chat)
│   ├── backends/{transformers_common,transformers_choice_backends}.py
│   ├── models/{base,choice,gemma_3_it,nemotron,llama_guard,registry}.py
│   ├── benchmarks/{base,holisafe,mmmu_pro,siuo,vlsbench,_hf_common,registry}.py
│   ├── evaluator.py                       # classification + multiple-choice runners
│   ├── metrics.py                         # safety metrics
│   ├── metrics_choice.py                  # multiple-choice accuracy summaries
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
- **MMMU-Pro** is wired here as `standard (10 options)` with offline local parquet
  loading and direct option-letter scoring for benchmark-aligned
  multimodal understanding comparison.
