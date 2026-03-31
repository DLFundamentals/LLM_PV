# LLM-PV (OpenAI Responses API)

**Goal:** Prompt a reasoning-capable model to output **executable Python** that implements a hidden mapping. We compile & evaluate the returned function, track validation/test accuracy, and **early-stop** when validation hits **1.0**. Datasets are **deterministic and persisted** per task.

---

## Key ideas
- **Deterministic datasets** per (fn, L, sizes, seed): stored under `datasets/<target>/L<length>/seed<derived>/{train.txt,val.txt,test.txt,meta.json}`.
- **Strict output contract:** model must return one JSON object: `{"code": "<python function>"}`.
- **Exec:** code is `exec`’d in a restricted namespace; **not a security sandbox**
- **Early stop:** perfect validation ⇒ compute test and stop further attempts for that grid point.

---

## Setup
```bash
conda activate llm_pv
# Assuming requirements are installed
export OPENAI_API_KEY=sk-...   # required
```

## Minimal run
```bash
python program_synthesis/runner.py   --functions fn_a   --lengths 50   --attempts 5   --enable-code-interpreter   --concurrency 1   --timeout 1200
```

> **Note:** For tabular tasks, you do not need to provide `--lengths` parameter

## Replicating Paper (Uses Default Config) (Note: Consumes $$)
```bash
python program_synthesis/runner.py --enable-code-interpreter
```

### Common flags
- Grid: `--functions fn_a fn_b ...` • `--lengths 100 50 30 25 20` • `--attempts 5`
- OpenAI: `--model gpt-5` • `--max-output-tokens 20000`  
  Reasoning/text: `--reasoning-effort high` • `--verbosity low`  
  Tools: `--enable-code-interpreter` • `--tool-choice auto|none`
- Data: `--train-size 100 --val-size 100 --test-size 10000 --seed 42 --dataset-dir datasets`
- Infra: `--concurrency 5 --timeout 1200`
- Artifacts: `--out-jsonl results_attempts.jsonl --out-csv results_attempts.csv`
- Dry run (no API call, print prompt): `--dry-run`

> Function IDs map to targets in `src/target_functions.py` via `EXPERIMENT_FUNCTION_MAPPING` (e.g., `fn_a → parity_all`). Decimal tasks (`prime_decimal*`) receive a decimal problem statement but the line format is the same.

---

## Artifacts
- **`results_attempts.jsonl`** — one record per attempt (prompt, raw text, usage tokens, timings, val/test accuracy, errors).
- **`results_attempts.csv`** — flat table (prompt/completion/reasoning tokens, `val_acc`, `test_acc`, `stopped_early`, etc.).
- **`datasets/`** — reused across runs for reproducibility.
- **`runner.log`** — JSON logs for each step (dataset reuse/generation, attempts, errors, early-stop, artifacts).

---

## Safety note
`exec()` is not a sandbox.
