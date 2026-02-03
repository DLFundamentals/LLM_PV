# Fine-Tuning / From-Scratch Training

Train (or finetune) LMs to **classify sequences** (binary/decimal) using the *same* task family as other tracks. Includes a YAML sweep runner.

---

## What’s here
- **`main.py`** — single run trainer.
- **`models.py`** — small registry of:
  - *From-scratch* configs: `llama3`, `deepseek`, `qwen0.6B`, `qwen1.7B`
  - *Finetune wrappers*: `llama3_finetune`, `qwen3_finetune`, `deepseek_finetune` (top-N layers unfreezed; binary head)
- **`dataloaders.py`** — `CodeDataset` that delegates to shared generators in `src/`, adds BOS, and caches tensors.
- **`collators.py`** — tokenizer-aware collator for pretrained models.
- **`sweep.py` & `sweep_config.yaml`** — YAML grid launcher (non-interactive) with per-run folders & logs.

---

## Setup
```bash
conda activate llm_pv
# Assuming requirements are installed
```

## Quick single run
```bash
# Example: Qwen 1.7B, parity_all at length 50
python finetuning/main.py   --results-path ./results_new/1/0   --model qwen1.7B   --target_func fn_a   --sequence_length 50   --train_set_size 200   --test_set_size 10000   --batch_size 20   --n_epochs 200   --lr 1e-5 --eta_min 1e-6   --precision bf16   --device cuda
```

**Artifacts (per run):**
```
results_new/<exp_id>/<run_id>/
  ├── config.json     # full CLI config
  ├── logs.log        # training/eval logs
  └── metrics.py      # pythonized arrays (loss/accuracy per checkpoint)
```

---

## YAML sweeps (recommended)
From repo root:
```bash
python finetuning/sweep.py finetuning/sweep_config.yaml
```

Example `finetuning/sweep_config.yaml` (included):
```yaml
experiment_name: "qwen1.7B_binary_tasks_paper_replication"
results_root: "./results_new"

base_args:
  model: "qwen1.7B"
  train_set_size: 200
  test_set_size: 10000
  batch_size: 20
  lr: 1e-5
  eta_min: 1e-6
  n_epochs: 200
  weight_decay: 1e-5
  precision: "bf16"
  device: "cuda"
  seed: 42

grid:
  sequence_length: [100, 50, 30, 25, 20]
  target_func: ["fn_g", "fn_a", "fn_d", "fn_c"]
```

> The sweep script writes each job’s **stdout** to `stdout.log` and dumps a **config.json** in the job folder.

---

## Data pipeline (unified with other tracks)
- Uses `src/data_handler.get_data_generator(...)` for **balanced 50/50** datasets.
- Builds one pool of size `train+test`, then uses `create_stratified_splits(...)` (validation size is `0` here).
- **Caching:** tensors saved under `sgd_datasets_cache/seed_<derived>/<hash>/` keyed by a **derived seed** of (func, length, sizes, global seed). Re-runs reuse identical data.

---

## Task-specific requirements

- **Tabular datasets** (`fn_m`, `fn_n`, `fn_o`, `fn_p`, `fn_q`): `--sequence_length` is auto-detected from metadata. You don't need to specify it, but if provided, it must match the fixed length:
  - `fn_m` (adult_income): `14`
  - `fn_n` (mushroom): `20`
  - `fn_o` (cdc_diabetes): `21`
  - `fn_p` (htru2): `8`
  - `fn_q` (chess): `35`

- **`fn_aa` (graph_has_cycle)**: `--sequence_length` must be a multiple of 4 (e.g., 100, 200, 300).

---

## Choosing models
- **From scratch**: `--model qwen1.7B|llama3|deepseek` (tiny configs, fast iterations).
- **Finetune HF weights**: `--model qwen3_finetune|llama3_finetune|deepseek_finetune`  
  The wrapper:
  - Freezes base weights
  - Unfreezes top `--num_layers_to_finetune` transformer blocks
  - Replaces lm_head with a **binary** head
---

## CLI (main.py) — frequently used
- Paths/seed: `--results-path`, `--seed`, `--device`
- Task: `--target_func fn_a|...`, `--sequence_length`
- Sizes: `--train_set_size`, `--test_set_size`
- Model: `--model`, `--num_layers_to_finetune`, `--vocab_size`, `--context_length`
- Train: `--batch_size`, `--n_epochs`, `--lr`, `--eta_min`, `--weight_decay`, `--precision bf16|fp16|fp32`

---
