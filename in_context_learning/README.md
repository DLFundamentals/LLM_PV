# In-Context Learning (ICL) with vLLM

**Goal:** Evaluate ICL by feeding **hundreds of few-shot examples** plus a single query to a **locally served** model (via vLLM), then parse and score predictions at scale.

---

## What it does
- Builds a grid of tasks: **(function ID, sequence length)**.
- For each task:
  - Generates a deterministic training block (few-shot examples) and a set of test prompts.
  - Runs **batched** inference with vLLM.
  - Parses `{"label": "0|1"}` from the output (robust JSON regex + fallback).
  - Logs **per-sample** results and a **per-task** accuracy summary.

---

## Setup
```bash
conda activate llm_pv
# Assuming requirements are installed
# Ensure CUDA & drivers match your vLLM build.
```

## Quick run
From repo root:
```bash
python in_context_learning/vllm_incontext.py   --model "Qwen/Qwen3-30B-A3B-Instruct-2507"   --functions fn_a   --lengths 50   --train-size 200   --test-size 100   --tensor-parallel-size 1   --out-csv qwen3_summary.csv   --out-jsonl qwen3_details.jsonl
```

## Replicate Paper
Here we used three different models, Qwen3-30B-A3B-Instruct-2507, Qwen3-Coder-30B-A3B-Instruct, and Deepseek-Coder-33B-Instruct. Rest all config are set default in the code.

```bash
python in_context_learning/vllm_incontext.py --model <huggingface-model-id>
```


### Notable arguments (subset)
- Grid: `--functions` (e.g., `fn_a fn_b ...`), `--lengths` (e.g., `100 50 30 25 20`)  
  > Special-case: `fn_h` (Dyck-2) uses lengths `[100, 80, 60, 40, 20]` if provided in metadata.  
  > Tabular datasets (`fn_m`, `fn_n`, `fn_o`, `fn_p`, `fn_q`): `--lengths` is auto-detected from metadata (fixed per dataset: `fn_m`=14, `fn_n`=20, `fn_o`=21, `fn_p`=8, `fn_q`=35).  
  > `fn_aa` (graph_has_cycle): `--lengths` must be multiples of 4 (e.g., `100 200 300`).
- Data: `--train-size` (few-shot examples per prompt), `--test-size` (prompts per task), `--seed`
- Model/vLLM: `--model`, `--tensor-parallel-size`, `--max-model-len`, `--temperature`, `--max-new-tokens`
- Artifacts: `--out-jsonl`, `--out-csv`

Defaults are defined in the `Config` dataclass inside `vllm_incontext.py`.

---

## Prompt shape (per test)
```
**Problem Statement:** ... (binary|decimal, length L)
**Data Examples:**
<seq> -> <label>
...
**Test Input:**
<seq>

You must output ONLY: {"label": "0" | "1"}
```

## Output files
- **JSONL (details):** each test prompt with prompt text, true label, raw model output, parsed prediction, correctness.
- **CSV (summary):** task-level accuracies (`fn_L<length> → accuracy`).
