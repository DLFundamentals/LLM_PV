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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# Ensure vLLM is available
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("vLLM is not installed. Please install it with: pip install vllm")
    exit(1)

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

    # vLLM & Model
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # model: str = "deepseek-ai/deepseek-coder-33b-instruct"
    # model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

    tensor_parallel_size: int = 1
    max_model_len: int = 25000
    trust_remote_code: bool = True

    # Sampling Parameters
    temperature: float = 0.2
    top_p: float = 0.95
    max_new_tokens: int = 1024

    # Artifacts
    output_jsonl: str = "in_context_learning/vllm_results_details.jsonl"
    output_csv: str = "in_context_learning/vllm_results_summary.csv"


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
    """Wraps the vLLM engine and orchestrates the inference process."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prompt_generator = PromptGenerator(cfg)
        logger.info(f"Initializing vLLM engine for model: {cfg.model}")
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

    def run_experiment(self) -> List[Dict[str, Any]]:
        """Executes the full experiment across all specified tasks."""
        all_results = []
        for fn in self.cfg.functions:
            task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
            current_lengths = task_meta.get("lengths", self.cfg.lengths)
            for L in current_lengths:
                task_prompts = self.prompt_generator.generate_prompts_for_task(fn, L)
                if not task_prompts:
                    continue

                logger.info(f"Running inference for task (fn={fn}, L={L}) with {len(task_prompts)} prompts...")
                start_time = time.perf_counter()

                prompts_to_run = [p['prompt'] for p in task_prompts]
                request_outputs = self.llm.generate(prompts_to_run, self.sampling_params, use_tqdm=True)

                duration = time.perf_counter() - start_time
                throughput = len(prompts_to_run) / duration
                logger.info(f"Task (fn={fn}, L={L}) completed in {duration:.2f}s ({throughput:.2f} prompts/s).")

                # Combine prompts with outputs for evaluation
                for prompt_data, output in zip(task_prompts, request_outputs):
                    model_output_text = output.outputs[0].text.strip()
                    all_results.append({
                        **prompt_data,
                        "model_output": model_output_text
                    })
        return all_results


# =========================
# Evaluation & Artifacts
# =========================
def parse_and_evaluate(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    task_stats = {}
    for res in results:
        task_key = (res['fn'], res['length'])
        if task_key not in task_stats:
            task_stats[task_key] = {'correct': 0, 'total': 0, 'failed_parses': 0}

        pred_label = None
        model_output = res['model_output']

        json_match = re.search(r'\{"label":\s*["\']?([01])["\']?\}', model_output)
        if json_match:
            pred_label = json_match.group(1)
        else:
            json_objects = re.findall(r'\{[^{}]*"label"[^{}]*\}', model_output, re.DOTALL)
            for json_str in reversed(json_objects):
                try:
                    data = json.loads(json_str)
                    if isinstance(data, dict) and 'label' in data:
                        label_val = str(data['label']).strip()
                        if label_val in ['0', '1']:
                            pred_label = label_val
                            break
                except:
                    continue

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
    logger.info(f"Saving {len(results)} detailed results to {cfg.output_jsonl}...")
    with open(cfg.output_jsonl, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Save summary CSV
    logger.info(f"Saving summary accuracies to {cfg.output_csv}...")
    summary_rows = []
    for task_key, acc in accuracies.items():
        fn, L_str = task_key.split('_L')
        summary_rows.append({'function': fn, 'length': int(L_str), 'accuracy': acc})

    with open(cfg.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['function', 'length', 'accuracy'])
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Artifacts saved successfully.")


# =========================
# CLI & Main Execution
# =========================
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run in-context learning experiments with vLLM.")
    cfg = Config()

    # Grid
    p.add_argument("--functions", nargs="*", help=f"Function IDs to test (default: {cfg.functions})")
    p.add_argument("--lengths", nargs="*", type=int, help=f"Sequence lengths to test (default: {cfg.lengths})")
    
    # Data
    p.add_argument("--train-size", type=int, help=f"In-context examples per prompt (default: {cfg.train_size})")
    p.add_argument("--test-size", type=int, help=f"Test prompts per task (default: {cfg.test_size})")
    p.add_argument("--seed", type=int, help=f"Global random seed (default: {cfg.seed})")

    # Model & VLLM
    p.add_argument("--model", type=str, help=f"Hugging Face model ID (default: {cfg.model})")
    p.add_argument("--tensor-parallel-size", type=int, help=f"GPU tensor parallelism (default: {cfg.tensor_parallel_size})")
    p.add_argument("--max-model-len", type=int, help=f"Max model sequence length (default: {cfg.max_model_len})")

    # Sampling
    p.add_argument("--temperature", type=float, help=f"Sampling temperature (default: {cfg.temperature})")
    p.add_argument("--top-p", type=float, help=f"top_p (default: {cfg.top_p})")
    p.add_argument("--max-new-tokens", type=int, help=f"Max generated tokens (default: {cfg.max_new_tokens})")

    # Output
    p.add_argument("--out-jsonl", help="Output JSONL path (default: results_attempts.jsonl)")
    p.add_argument("--out-csv", help="Output CSV path (default: results_attempts.csv)")

    args = p.parse_args()

    # Apply overrides from CLI
    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.train_size: cfg.train_size = args.train_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed: cfg.seed = args.seed
    if args.model: cfg.model = args.model
    if args.tensor_parallel_size: cfg.tensor_parallel_size = args.tensor_parallel_size
    if args.max_model_len: cfg.max_model_len = args.max_model_len
    if args.temperature is not None: cfg.temperature = args.temperature
    if args.top_p is not None: cfg.top_p = args.top_p
    if args.out_jsonl: cfg.output_jsonl = args.out_jsonl
    if args.out_csv: cfg.output_csv = args.out_csv

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