# vllm_incontext.py
"""
This script orchestrates an end-to-end experiment loop for in-context learning
using a local model served by vLLM.

The process is as follows:
  1. For each task (a combination of a target function and sequence length),
     it generates a deterministic set of prompts. Each prompt contains a number
     of few-shot examples (the "in-context" part) and one test query.
  2. It uses vLLM to run batched inference on these prompts.
  3. It parses the model's JSON response (e.g., {"label": "1"}) to extract the
     predicted label.
  4. It compares the prediction to the ground truth to calculate accuracy.
  5. It logs results and saves a detailed JSONL file with per-sample outcomes
     and a final CSV summary with accuracy per task.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import random, hashlib
import numpy as np, torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

from openai import OpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the data generators
from src.data_handler import (
    BaseDataGenerator,
    get_data_generator,
    create_stratified_splits,
)

from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Configuration
# =========================
@dataclass
class Config:
    """Manages all configuration for the experiment."""
    # Experiment Grid
    functions: List[str] = field(default_factory=lambda: ["fn_a", "fn_b", "fn_c", "fn_d", "fn_e", "fn_f", "fn_g", "fn_h", "fn_i", "fn_j", "fn_k", "fn_l"])
    lengths: List[int] = field(default_factory=lambda: [20, 25, 30, 50, 100])

    # Data Generation
    train_size: int = 200  # Number of in-context examples per prompt
    test_size: int = 100   # Number of test prompts per task (fn, L)
    seed: int = 42

    # Backend & Model
    # Keep existing vLLM defaults, and allow OpenAI as an optional addition.
    provider: str = "vllm"  # choices: openai, vllm
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # model: str = "deepseek-ai/deepseek-coder-33b-instruct"
    # model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    # model: str = "gpt-5.1"  # use with --provider openai
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    reasoning_effort: str = "high"
    verbosity: str = "low"
    max_output_tokens: int = 20000
    openai_max_concurrency: int = 8

    # vLLM-only settings
    tensor_parallel_size: int = 1
    max_model_len: int = 25000
    trust_remote_code: bool = True

    # Sampling Parameters
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 1024
    # Test-time inference mode: default to unbatched for wall-clock comparability.
    batch_test_inference: bool = False

    # Artifacts
    output_jsonl: str = "in_context_learning/vllm_results_details.jsonl"
    output_csv: str = "in_context_learning/vllm_results_summary.csv"
    stream_jsonl: bool = False
    append_summary_csv: bool = True


# =========================
# Constants (from target functions for consistency)
# =========================
FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING

DECIMAL_FNS = {"prime_decimal", "prime_decimal_tf_check"}
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}


# =========================
# Prompt Generation
# =========================
def build_user_prompt(
    data_examples: List[str],
    test_input: str,
    seq_len: int,
    decimal: bool = False,
    tabular: bool = False,
) -> str:
    """Creates a structured few-shot prompt for an in-context classification task."""
    if tabular:
        problem_statement = (
            f"**Problem Statement:**\n"
            f"You are given tabular input data (comma-separated feature:value pairs) and their "
            f"corresponding binary outputs ('0' or '1'). Your task is to analyze the provided "
            f"examples to understand the underlying pattern or function. Then, for the final "
            f"test input, predict its correct label."
        )
    else:
        problem_type = "decimal" if decimal else "binary"
        problem_statement = (
            f"**Problem Statement:**\n"
            f"You are given input vectors (type: {problem_type}, length: {seq_len}) and their "
            f"corresponding binary outputs ('0' or '1'). Your task is to analyze the provided "
            f"examples to understand the underlying pattern or function. Then, for the final "
            f"test input, predict its correct label."
        )
    prompt = f"{problem_statement}\n\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += "**Test Input:**\n```\n" + test_input + "\n```\n\n"
    prompt += 'Based on the examples, what is the label for the test input? You must output ONLY a single JSON object in the format: {"label": "<your predicted label>"}. Do not write any code, explanations, solutions, or additional text. Output only the JSON object with no other content.'
    return prompt


class PromptGenerator:
    """Generates deterministic and reproducible prompts for each task."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _stable_derived_seed(self, fn: str, L: int) -> int:
        """
        Match program_synthesis/runner.py:
          key = f"{fn}|L={L}|train={train+val}|test={test}|base_seed={seed}"
          derived = sha256(key)[:8] -> int -> mask 31 bits
        """
        key = (
            f"{fn}|L={L}"
            f"|train={self.cfg.train_size + 0}"   # val=0 in ICL
            f"|test={self.cfg.test_size}"
            f"|base_seed={self.cfg.seed}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return (int.from_bytes(digest[:8], "big") & 0x7FFFFFFF)

    def _get_data_generator(self, target_name: str, L: int, size: int) -> BaseDataGenerator:
        """Selects and instantiates the correct data generator for a given task."""
        return get_data_generator(target_name, L, size)

    def _generate_lines(self, generator: BaseDataGenerator, target_name: str) -> List[str]:
        """Runs the data generator and formats the output into lines."""
        data = generator.generate_data()
        return [f"{''.join(sample['Input'])} -> {sample['Output']}" for sample in data]

    def generate_prompts_for_task(self, fn: str, L: int) -> List[Dict[str, Any]]:
        """
        Generates a list of fully-formed prompts for a given (fn, L) task.
        Each prompt includes training examples and a single unique test query.
        """
        if fn not in FUNCTION_NAME_MAPPING:
            logger.warning(f"Function key '{fn}' not in FUNCTION_NAME_MAPPING. Skipping.")
            return []

        target_name = FUNCTION_NAME_MAPPING[fn]
        is_decimal = target_name in DECIMAL_FNS
        is_tabular = target_name in TABULAR_FNS

        # Use the SAME derived seed scheme as program_synthesis/runner.py
        derived_seed = self._stable_derived_seed(fn, L)

        # Seed ALL relevant RNGs for reproducibility across generators that use torch / numpy
        random.seed(derived_seed)
        np.random.seed(derived_seed)
        try:
            torch.manual_seed(derived_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(derived_seed)
        except Exception:
            pass
        
        logger.info(f"Generating pooled data for task (fn={fn}, L={L}) with seed {derived_seed}...")
        try:
            # 1) Build ONE pooled dataset of size (train + val=0 + test)
            total = self.cfg.train_size + self.cfg.test_size
            pooled_gen = self._get_data_generator(target_name, L, total)
            pooled = pooled_gen.generate_data()  # List[dict: {'Input': np.array, 'Output': '0'|'1'}]

            # 2) Stratified split (val_size=0) using the shared helper
            train_split, val_split, test_split = create_stratified_splits(
                all_samples=pooled,
                train_size=self.cfg.train_size,
                val_size=0,
                test_size=self.cfg.test_size,
                device='cpu'
            )

            # 3) Convert to the same "<seq> -> <label>" line format
            def _to_lines(samples):
                return [f"{''.join(s['Input'])} -> {s['Output']}" for s in samples]
            train_lines = _to_lines(train_split)
            test_lines  = _to_lines(test_split)

            # 4) Shuffle train lines (runner shuffles train & val before persisting)
            random.shuffle(train_lines)
        except Exception as e:
            logger.error(f"Failed to generate data for (fn={fn}, L={L}): {e}", exc_info=True)
            return []

        # Create a prompt for each test line
        prompts = []
        for test_line in test_lines:
            test_input, true_label = [part.strip() for part in test_line.split("->")]
            prompt_text = build_user_prompt(train_lines, test_input, L, is_decimal, is_tabular)
            prompts.append({
                "prompt": prompt_text,
                "true_label": true_label,
                "fn": fn,
                "length": L
            })
        return prompts


# =========================
# VLLM Runner
# =========================
class VLLMRunner:
    """Wraps either OpenAI or vLLM for in-context classification."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt_generator = PromptGenerator(cfg)
        self.llm = None
        self.sampling_params = None
        self.openai_client: Optional[OpenAI] = None

        if cfg.provider == "openai":
            if not cfg.api_key:
                raise SystemExit("OPENAI_API_KEY is required for provider=openai.")
            self.openai_client = OpenAI(api_key=cfg.api_key)
            logger.info(f"OpenAI client initialized for model: {cfg.model}")
        elif cfg.provider == "vllm":
            logger.info(f"Initializing vLLM engine for model: {cfg.model}")
            try:
                from vllm import LLM, SamplingParams
            except ImportError as e:
                raise SystemExit("vLLM is not installed. Please install it with: pip install vllm") from e
            try:
                self.llm = LLM(
                    model=cfg.model,
                    tensor_parallel_size=cfg.tensor_parallel_size,
                    trust_remote_code=cfg.trust_remote_code,
                    max_model_len=cfg.max_model_len
                )
            except Exception as e:
                logger.error(f"Failed to initialize vLLM LLM engine: {e}", exc_info=True)
                raise
            self.sampling_params = SamplingParams(
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_new_tokens
            )
            logger.info("vLLM engine initialized successfully.")
        else:
            raise SystemExit(f"Unsupported provider: {cfg.provider}")

    def _run_vllm_for_task(self, task_prompts: List[Dict[str, Any]]) -> List[str]:
        if self.llm is None or self.sampling_params is None:
            raise RuntimeError("vLLM is not initialized.")
        prompts_to_run = [p['prompt'] for p in task_prompts]
        if self.cfg.batch_test_inference:
            request_outputs = self.llm.generate(prompts_to_run, self.sampling_params, use_tqdm=True)
            return [output.outputs[0].text.strip() for output in request_outputs]

        # Unbatched mode: run one prompt at a time to match non-batched test-time timing.
        outputs: List[str] = []
        for prompt in prompts_to_run:
            request_outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            if not request_outputs or not request_outputs[0].outputs:
                outputs.append("")
            else:
                outputs.append(request_outputs[0].outputs[0].text.strip())
        return outputs

    def _run_openai_for_task(self, task_prompts: List[Dict[str, Any]], fn: str, L: int) -> List[str]:
        if self.openai_client is None:
            raise RuntimeError("OpenAI client is not initialized.")

        total = len(task_prompts)
        if total == 0:
            return []

        # Strict schema: model must return {"label": "0"} or {"label": "1"}.
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "label_prediction",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["0", "1"],
                            "description": "The predicted binary label for the test input.",
                        }
                    },
                    "required": ["label"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        def _extract_chat_content(resp: Any) -> str:
            content = resp.choices[0].message.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        parts.append(str(item.get("text", "")))
                    else:
                        parts.append(str(getattr(item, "text", "")))
                return "".join(parts).strip()
            return str(content).strip()

        def process_single_prompt(idx: int, prompt_text: str) -> Tuple[int, str]:
            base_kwargs: Dict[str, Any] = {
                "model": self.cfg.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "response_format": response_schema,
            }

            try:
                try:
                    response = self.openai_client.chat.completions.create(
                        **base_kwargs,
                        reasoning_effort=self.cfg.reasoning_effort,
                    )
                except Exception:
                    response = self.openai_client.chat.completions.create(**base_kwargs)
                return idx, _extract_chat_content(response)
            except Exception as e:
                logger.warning(
                    f"OpenAI API call failed for task (fn={fn}, L={L}, idx={idx}): {e}"
                )
                return idx, ""

        outputs: List[str] = [""] * total
        max_workers = max(1, min(self.cfg.openai_max_concurrency, total))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_prompt, i, p["prompt"]): i
                for i, p in enumerate(task_prompts)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                outputs[idx] = result

        return outputs

    def _append_summary_row(
        self,
        fn: str,
        L: int,
        accuracy: float,
        adaptation_duration_ms: int,
        test_duration_ms: int,
        total_wall_clock_duration_ms: int,
    ) -> None:
        """Append one task-level summary row immediately after task completion."""
        out_dir = os.path.dirname(self.cfg.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        file_exists = os.path.exists(self.cfg.output_csv)
        needs_header = (not file_exists) or os.path.getsize(self.cfg.output_csv) == 0
        with open(self.cfg.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "function",
                    "length",
                    "accuracy",
                    "duration_ms",
                    "adaptation_duration_ms",
                    "test_duration_ms",
                    "total_wall_clock_duration_ms",
                ],
            )
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "function": fn,
                    "length": int(L),
                    "accuracy": float(accuracy),
                    "duration_ms": int(total_wall_clock_duration_ms),
                    "adaptation_duration_ms": int(adaptation_duration_ms),
                    "test_duration_ms": int(test_duration_ms),
                    "total_wall_clock_duration_ms": int(total_wall_clock_duration_ms),
                }
            )
            f.flush()

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Executes the full experiment across all specified tasks."""
        all_results = []
        jsonl_fp = None
        try:
            if self.cfg.stream_jsonl:
                out_dir = os.path.dirname(self.cfg.output_jsonl)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                jsonl_fp = open(self.cfg.output_jsonl, "w", encoding="utf-8")
                logger.info(f"Streaming per-prompt results to {self.cfg.output_jsonl}")

            for fn in self.cfg.functions:
                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = task_meta.get("lengths", self.cfg.lengths)
                for L in current_lengths:
                    task_total_t0 = time.perf_counter()
                    adaptation_t0 = time.perf_counter()
                    task_prompts = self.prompt_generator.generate_prompts_for_task(fn, L)
                    adaptation_duration_ms = int((time.perf_counter() - adaptation_t0) * 1000)
                    if not task_prompts:
                        continue

                    logger.info(f"Running inference for task (fn={fn}, L={L}) with {len(task_prompts)} prompts...")
                    logger.info(
                        "Test inference mode: %s",
                        "batched" if self.cfg.batch_test_inference else "unbatched",
                    )
                    test_t0 = time.perf_counter()

                    if self.cfg.provider == "vllm":
                        model_outputs = self._run_vllm_for_task(task_prompts)
                    else:
                        model_outputs = self._run_openai_for_task(task_prompts, fn, L)

                    test_duration_s = time.perf_counter() - test_t0
                    test_duration_ms = int(test_duration_s * 1000)
                    throughput = len(task_prompts) / test_duration_s if test_duration_s > 0 else 0.0
                    total_wall_clock_duration_ms = int((time.perf_counter() - task_total_t0) * 1000)
                    logger.info(
                        f"Task (fn={fn}, L={L}) completed: "
                        f"adapt_ms={adaptation_duration_ms}, test_ms={test_duration_ms}, "
                        f"total_ms={total_wall_clock_duration_ms} ({throughput:.2f} prompts/s)."
                    )

                    # Combine prompts with outputs for evaluation
                    task_correct = 0
                    task_total = 0
                    for prompt_data, model_output_text in zip(task_prompts, model_outputs):
                        row = {
                            **prompt_data,
                            "model_output": model_output_text,
                            "duration_ms": total_wall_clock_duration_ms,
                            "adaptation_duration_ms": adaptation_duration_ms,
                            "test_duration_ms": test_duration_ms,
                            "total_wall_clock_duration_ms": total_wall_clock_duration_ms,
                        }
                        all_results.append(row)
                        pred_label = _extract_pred_label(model_output_text)
                        if pred_label is not None and pred_label == prompt_data["true_label"]:
                            task_correct += 1
                        task_total += 1
                        if jsonl_fp is not None:
                            jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                            jsonl_fp.flush()

                    if self.cfg.append_summary_csv and task_total > 0:
                        task_acc = task_correct / task_total
                        self._append_summary_row(
                            fn=fn,
                            L=L,
                            accuracy=task_acc,
                            adaptation_duration_ms=adaptation_duration_ms,
                            test_duration_ms=test_duration_ms,
                            total_wall_clock_duration_ms=total_wall_clock_duration_ms,
                        )
                        logger.info(
                            f"Appended summary row for task (fn={fn}, L={L}) with accuracy={task_acc:.2%} "
                            f"to {self.cfg.output_csv}"
                        )
        finally:
            if jsonl_fp is not None:
                jsonl_fp.close()
        return all_results


# =========================
# Evaluation & Artifacts
# =========================
def _extract_pred_label(model_output: str) -> Optional[str]:
    json_match = re.search(r'\{"label":\s*["\']?([01])["\']?\}', model_output)
    if json_match:
        return json_match.group(1)

    json_objects = re.findall(r'\{[^{}]*"label"[^{}]*\}', model_output, re.DOTALL)
    for json_str in reversed(json_objects):
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and 'label' in data:
                label_val = str(data['label']).strip()
                if label_val in ['0', '1']:
                    return label_val
        except Exception:
            continue
    return None


def parse_and_evaluate(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    task_stats = {}
    for res in results:
        task_key = (res['fn'], res['length'])
        if task_key not in task_stats:
            task_stats[task_key] = {'correct': 0, 'total': 0, 'failed_parses': 0}

        model_output = res['model_output']
        pred_label = _extract_pred_label(model_output)

        res['predicted_label'] = pred_label
        res['is_correct'] = (pred_label is not None and pred_label == res['true_label'])

        task_stats[task_key]['total'] += 1
        if res['is_correct']:
            task_stats[task_key]['correct'] += 1
        if pred_label is None:
            task_stats[task_key]['failed_parses'] += 1

    accuracies = {}
    for (fn, L), stats in task_stats.items():
        total_valid = stats['total']
        if total_valid > 0:
            accuracies[f"{fn}_L{L}"] = stats['correct'] / total_valid
        else:
            accuracies[f"{fn}_L{L}"] = 0.0
        
        if stats['failed_parses'] > 0:
            logger.warning(
                f"Task (fn={fn}, L={L}): Failed to parse "
                f"{stats['failed_parses']}/{stats['total']} outputs."
            )

    return results, accuracies

def save_artifacts(cfg: Config, results: List[Dict[str, Any]], accuracies: Dict[str, float]):
    """Saves detailed JSONL results and a CSV summary."""
    # Save detailed JSONL
    if cfg.stream_jsonl:
        logger.info(f"Detailed JSONL already streamed during run to {cfg.output_jsonl}.")
    else:
        logger.info(f"Saving {len(results)} detailed results to {cfg.output_jsonl}...")
        with open(cfg.output_jsonl, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Save summary CSV
    if cfg.append_summary_csv:
        logger.info(f"Summary CSV rows already appended per task to {cfg.output_csv}.")
    else:
        logger.info(f"Saving summary accuracies to {cfg.output_csv}...")
        summary_rows = []
        for task_key, acc in accuracies.items():
            fn, L_str = task_key.split('_L')
            rows_for_task = [
                r for r in results
                if r.get("fn") == fn and int(r.get("length")) == int(L_str)
            ]
            first_row = rows_for_task[0] if rows_for_task else {}
            summary_rows.append({
                'function': fn,
                'length': int(L_str),
                'accuracy': acc,
                'duration_ms': first_row.get('duration_ms'),
                'adaptation_duration_ms': first_row.get('adaptation_duration_ms'),
                'test_duration_ms': first_row.get('test_duration_ms'),
                'total_wall_clock_duration_ms': first_row.get('total_wall_clock_duration_ms'),
            })

        with open(cfg.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'function',
                    'length',
                    'accuracy',
                    'duration_ms',
                    'adaptation_duration_ms',
                    'test_duration_ms',
                    'total_wall_clock_duration_ms',
                ],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
    logger.info("Artifacts saved successfully.")


# =========================
# CLI & Main Execution
# =========================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run in-context learning experiments with OpenAI or vLLM.")
    cfg = Config()

    # Grid
    p.add_argument("--functions", nargs="*", help=f"Function IDs to test (default: {cfg.functions})")
    p.add_argument("--lengths", nargs="*", type=int, help=f"Sequence lengths to test (default: {cfg.lengths})")
    
    # Data
    p.add_argument("--train-size", type=int, help=f"In-context examples per prompt (default: {cfg.train_size})")
    p.add_argument("--test-size", type=int, help=f"Test prompts per task (default: {cfg.test_size})")
    p.add_argument("--seed", type=int, help=f"Global random seed (default: {cfg.seed})")

    # Backend & Model
    p.add_argument("--provider", choices=["openai", "vllm"], help=f"Inference backend (default: {cfg.provider})")
    p.add_argument("--model", type=str, help=f"Model ID (default: {cfg.model}; for OpenAI e.g. gpt-5.1)")
    p.add_argument("--reasoning-effort", choices=["minimal", "medium", "high"], help=f"OpenAI reasoning effort (default: {cfg.reasoning_effort})")
    p.add_argument("--verbosity", choices=["low", "medium", "high"], help=f"OpenAI text verbosity (default: {cfg.verbosity})")
    p.add_argument("--max-output-tokens", type=int, help=f"OpenAI max output tokens (default: {cfg.max_output_tokens})")
    p.add_argument("--openai-max-concurrency", type=int, help=f"OpenAI max parallel requests (default: {cfg.openai_max_concurrency})")

    # vLLM-only
    p.add_argument("--tensor-parallel-size", type=int, help=f"GPU tensor parallelism (default: {cfg.tensor_parallel_size})")
    p.add_argument("--max-model-len", type=int, help=f"Max model sequence length (default: {cfg.max_model_len})")
    p.add_argument("--batch-test-inference", action="store_true", help="Enable batched test-time inference (default: off).")

    # Sampling
    p.add_argument("--temperature", type=float, help=f"Sampling temperature (default: {cfg.temperature})")
    p.add_argument("--top-p", type=float, help=f"top_p (default: {cfg.top_p})")
    p.add_argument("--max-new-tokens", type=int, help=f"Max generated tokens (default: {cfg.max_new_tokens})")

    # Output
    p.add_argument("--out-jsonl", help="Output JSONL path (default: results_attempts.jsonl)")
    p.add_argument("--out-csv", help="Output CSV path (default: results_attempts.csv)")
    p.add_argument("--no-stream-jsonl", action="store_true", help="Disable live per-prompt JSONL streaming.")
    p.add_argument("--no-append-summary-csv", action="store_true", help="Disable per-task append mode for summary CSV.")

    args = p.parse_args()

    # Apply overrides from CLI
    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.train_size: cfg.train_size = args.train_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed: cfg.seed = args.seed
    if args.provider: cfg.provider = args.provider
    if args.model: cfg.model = args.model
    if args.reasoning_effort: cfg.reasoning_effort = args.reasoning_effort
    if args.verbosity: cfg.verbosity = args.verbosity
    if args.max_output_tokens: cfg.max_output_tokens = args.max_output_tokens
    if args.openai_max_concurrency is not None: cfg.openai_max_concurrency = max(1, args.openai_max_concurrency)
    if args.tensor_parallel_size: cfg.tensor_parallel_size = args.tensor_parallel_size
    if args.max_model_len: cfg.max_model_len = args.max_model_len
    if args.batch_test_inference: cfg.batch_test_inference = True
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.top_p is not None: cfg.top_p = args.top_p
    if args.out_jsonl: cfg.output_jsonl = args.out_jsonl
    if args.out_csv: cfg.output_csv = args.out_csv
    if args.no_stream_jsonl: cfg.stream_jsonl = False
    if args.no_append_summary_csv: cfg.append_summary_csv = False

    return cfg


def main():
    """Main entry point for the script."""
    config = parse_args()
    logger.info(f"Starting experiment with configuration: {config}")

    runner = VLLMRunner(config)
    results = runner.run_experiment()
    
    if not results:
        logger.warning("No results were generated. Exiting.")
        return

    evaluated_results, accuracies = parse_and_evaluate(results)
    
    logger.info("--- Experiment Summary ---")
    for task, acc in accuracies.items():
        logger.info(f"Task: {task:<20} Accuracy: {acc:.2%}")
    logger.info("--------------------------")
    
    save_artifacts(config, evaluated_results, accuracies)


if __name__ == "__main__":
    main()