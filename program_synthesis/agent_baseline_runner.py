"""Agent baseline runner for program_synthesis tasks.

Two agent baselines:
  1. iterative_agent — calls OpenAI directly with runner.py-style prompts,
     tests the code on validation data, and iterates with feedback.
  2. mlagentbench   — uses the MLAgentBench autonomous research agent.

Each trial logs every generated code attempt to <run_dir>/attempts/.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.data_handler import create_stratified_splits, get_data_generator
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING

PROJECT_ROOT = os.path.abspath(parent_dir)
FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("agent_baseline_runner")


@dataclass
class Config:
    agents: List[str] = field(default_factory=lambda: ["iterative_agent", "mlagentbench"])
    functions: List[str] = field(
        default_factory=lambda: [
            "fn_a", "fn_b", "fn_c", "fn_d", "fn_e", "fn_f",
            "fn_g", "fn_h", "fn_i", "fn_j", "fn_k", "fn_l", "fn_v",
            "fn_aa",
        ]
    )
    lengths: List[int] = field(default_factory=lambda: [100])
    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int = int(os.getenv("VAL_SIZE", "100"))
    test_size: int = int(os.getenv("TEST_SIZE", "10000"))
    seed: int = int(os.getenv("GLOBAL_SEED", "42"))
    num_trials: int = int(os.getenv("NUM_TRIALS", "5"))
    timeout_s: int = int(os.getenv("AGENT_TIMEOUT_S", "1800"))
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-4o")
    agent_steps: int = int(os.getenv("AGENT_STEPS", "10"))
    mlab_max_steps: int = int(os.getenv("MLAB_MAX_STEPS", "30"))
    mlab_llm: str = os.getenv("MLAB_LLM", "gpt-4")
    dataset_dir: str = os.path.join(PROJECT_ROOT, os.getenv("DATASET_DIR", "program_synthesis/datasets"))
    runs_dir: str = os.path.join(PROJECT_ROOT, os.getenv("AGENT_RUNS_DIR", "program_synthesis/agent_runs"))
    out_jsonl: str = os.path.join(PROJECT_ROOT, os.getenv("OUT_JSONL", "program_synthesis/agent_baseline_results.jsonl"))
    out_csv: str = os.path.join(PROJECT_ROOT, os.getenv("OUT_CSV", "program_synthesis/agent_baseline_results.csv"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _stable_derived_seed(cfg: Config, fn: str, L: int) -> int:
    key = f"{fn}|L={L}|train={cfg.train_size + cfg.val_size}|test={cfg.test_size}|base_seed={cfg.seed}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _format_input(input_val: Any) -> str:
    if isinstance(input_val, np.ndarray):
        return input_val.item() if input_val.size == 1 else "".join(input_val.tolist())
    if isinstance(input_val, (list, tuple)):
        return "".join(input_val)
    return str(input_val)


def get_or_create_splits(cfg: Config, fn: str, L: int) -> Tuple[List[str], List[str], List[str], str]:
    target_name = FUNCTION_NAME_MAPPING[fn]
    derived_seed = _stable_derived_seed(cfg, fn, L)
    base = os.path.join(cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
    paths = {k: os.path.join(base, f"{k}.txt") for k in ("train", "val", "test")}

    if all(os.path.exists(p) for p in paths.values()):
        return _read_lines(paths["train"]), _read_lines(paths["val"]), _read_lines(paths["test"]), target_name

    _set_seed(derived_seed)
    total_samples = cfg.train_size + cfg.val_size + cfg.test_size
    generator = get_data_generator(target_name, L, total_samples)
    all_samples = generator.generate_data()
    train_split, val_split, test_split = create_stratified_splits(
        all_samples, cfg.train_size, cfg.val_size, cfg.test_size, device="cpu"
    )
    os.makedirs(base, exist_ok=True)
    for split_name, split_data in (("train", train_split), ("val", val_split), ("test", test_split)):
        with open(paths[split_name], "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(f"{_format_input(s['Input'])} -> {s['Output']}\n")
    return _read_lines(paths["train"]), _read_lines(paths["val"]), _read_lines(paths["test"]), target_name


def _normalize_pred_to01(pred: Any) -> int:
    try:
        if hasattr(pred, "item"):
            pred = pred.item()
    except Exception:
        pass
    if isinstance(pred, bool):
        return 1 if pred else 0
    if isinstance(pred, int):
        return 1 if pred != 0 else 0
    if isinstance(pred, str):
        s = pred.strip().strip("\"'")
        if s in ("0", "1"):
            return int(s)
        sl = s.lower()
        if sl in ("true", "false"):
            return 1 if sl == "true" else 0
        try:
            return 1 if int(float(s)) != 0 else 0
        except Exception:
            return 1 if len(s) > 0 else 0
    return 1 if pred else 0


def _parse_tabular_input(x_str: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in x_str.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        try:
            result[k.strip()] = float(v.strip())
        except ValueError:
            result[k.strip()] = v.strip()
    return result


def evaluate_accuracy(fn_callable: Callable[[Any], Any], lines: List[str], is_tabular: bool) -> float:
    if not lines:
        return 0.0
    correct = 0
    for line in lines:
        x_raw, y_raw = line.split("->")
        y_true = int(y_raw.strip())
        x_in = _parse_tabular_input(x_raw.strip()) if is_tabular else x_raw.strip()
        y_pred = _normalize_pred_to01(fn_callable(x_in))
        correct += int(y_pred == y_true)
    return correct / len(lines)


def compile_callable_from_code(code: str, label: str = "<agent>") -> Callable[[Any], Any]:
    code = textwrap.dedent(code.strip())
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code, flags=re.IGNORECASE | re.DOTALL)
    tree = ast.parse(code)

    safe_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)):
            safe_nodes.append(node)

    safe_tree = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(safe_tree)

    fn_names = [n.name for n in safe_nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    ns: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        exec(compile(safe_tree, filename=label, mode="exec"), ns)
    except Exception:
        ns = {"__builtins__": __builtins__}
        exec(compile(tree, filename=label, mode="exec"), ns)

    fn = ns.get("f")
    if not callable(fn):
        for name in fn_names:
            if callable(ns.get(name)):
                fn = ns[name]
                break
    if not callable(fn):
        raise ValueError(f"No callable function found in agent output ({label})")
    return fn


def _make_run_dir(cfg: Config, agent: str, fn: str, L: int, trial: int) -> str:
    anon_key = (
        f"{agent}|{fn}|L={L}|trial={trial}|seed={cfg.seed}|"
        f"train={cfg.train_size}|val={cfg.val_size}|test={cfg.test_size}"
    )
    run_uid = hashlib.sha256(anon_key.encode("utf-8")).hexdigest()[:12]
    run_dir = os.path.join(cfg.runs_dir, agent, f"job_{run_uid}", f"trial{trial}")
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _materialize_task(
    run_dir: str, L: int, is_tabular: bool,
    train_lines: List[str], val_lines: List[str],
) -> Dict[str, str]:
    """Write anonymized task files into run_dir. Returns absolute paths."""
    paths = {}
    for name, lines in (("train.txt", train_lines), ("val.txt", val_lines)):
        p = os.path.join(run_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        paths[name] = p

    task_desc = _build_task_description(L, is_tabular, train_lines)
    task_path = os.path.join(run_dir, "task.md")
    with open(task_path, "w", encoding="utf-8") as f:
        f.write(task_desc)
    paths["task.md"] = task_path
    paths["solution.py"] = os.path.join(run_dir, "solution.py")
    return paths


def _build_task_description(L: int, is_tabular: bool, train_lines: List[str]) -> str:
    input_fmt = "x is a dict[str, float|str] of named features" if is_tabular else "x is a string"
    return (
        f"Sequence length: {L}\n\n"
        "Goal:\n"
        "Given binary classification examples, write a Python function `f(x)` in `solution.py`.\n"
        "The function should return 0 or 1.\n\n"
        f"Input format: {input_fmt}\n\n"
        "Training examples:\n"
        + "\n".join(train_lines)
        + "\n"
    )


def _save_attempt(run_dir: str, step: int, code: str, val_acc: float,
                  error: Optional[str]) -> None:
    """Save each agent attempt for inspection."""
    attempts_dir = os.path.join(run_dir, "attempts")
    os.makedirs(attempts_dir, exist_ok=True)
    meta = {"step": step, "val_acc": val_acc, "error": error}
    with open(os.path.join(attempts_dir, f"step_{step:02d}.py"), "w") as f:
        f.write(f"# val_acc={val_acc:.4f}  error={error}\n")
        f.write(code)
    with open(os.path.join(attempts_dir, f"step_{step:02d}.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Iterative OpenAI Agent — uses runner.py-style prompts + feedback loop
# ---------------------------------------------------------------------------

def _build_initial_prompt(L: int, is_tabular: bool, train_lines: List[str]) -> str:
    """Same prompt style as runner.py's build_user_prompt."""
    if is_tabular:
        desc = (
            f"Given tabular input data (comma-separated feature:value pairs) and "
            f"their corresponding scalar binary outputs ('0' or '1'), find a concise "
            f"Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or "
            f"mathematical representation of the target function."
        )
    else:
        desc = (
            f"Given a sequence of input vectors (binary, length {L}) and their "
            f"corresponding scalar binary outputs ('0' or '1'), find a concise "
            f"Python function `f(x)` that accurately approximates the underlying relationship. "
            f"The function should not be a trainable model, but a direct logical or "
            f"mathematical representation of the target function."
        )
    prompt = f"**Problem Statement:**\n{desc}\n\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(train_lines) + "\n```\n\n"
    prompt += (
        'You must output ONLY a single JSON object: {"code": "<python function>"}.\n'
        "The code must define a function `def f(x):` that takes a string x and returns 0 or 1.\n"
    )
    return prompt


def _build_feedback_prompt(prev_code: str, val_acc: float, error: Optional[str],
                           train_lines: List[str]) -> str:
    """Send back accuracy/error so the model can iterate."""
    prompt = "Your previous solution:\n```python\n" + prev_code + "\n```\n\n"
    if error:
        prompt += f"**Error:** {error}\n\n"
    else:
        prompt += f"**Validation accuracy:** {val_acc:.4f} ({int(val_acc*100)}% correct)\n\n"
    prompt += (
        "Please improve the function. Analyze the patterns in the data more carefully.\n\n"
        "**Data Examples (reminder):**\n```\n" + "\n".join(train_lines[:30]) + "\n```\n\n"
        'Output ONLY: {"code": "<improved python function>"}\n'
    )
    return prompt


def _extract_code_from_response(text: str) -> str:
    """Pull code out of the JSON or markdown response."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "code" in obj:
            return obj["code"]
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{[^{}]*"code"\s*:\s*"(.*?)"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            return obj["code"]
        except (json.JSONDecodeError, KeyError):
            pass

    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    if "def f(" in text:
        lines = text.split("\n")
        code_lines = []
        capture = False
        for line in lines:
            if line.strip().startswith("def f("):
                capture = True
            if capture:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

    return text


def run_iterative_agent(
    run_dir: str,
    train_lines: List[str],
    val_lines: List[str],
    L: int,
    is_tabular: bool,
    model: str,
    steps: int,
) -> Tuple[str, float, List[Dict]]:
    """Iterative agent: generate → test → feedback → improve.
    Returns (best_code, best_val_acc, attempt_log)."""
    import openai
    client = openai.OpenAI()

    best_code: Optional[str] = None
    best_val_acc = 0.0
    prev_code: Optional[str] = None
    prev_acc = 0.0
    prev_error: Optional[str] = None
    attempt_log: List[Dict] = []

    for step in range(steps):
        if step == 0:
            user_msg = _build_initial_prompt(L, is_tabular, train_lines)
        else:
            user_msg = _build_feedback_prompt(prev_code, prev_acc, prev_error, train_lines)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert at finding patterns in binary sequences. "
                        "Output only valid JSON with a 'code' field containing a Python function def f(x)."
                    )},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
                max_tokens=4096,
            )
            raw_text = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("  Step %d: API error: %s", step, e)
            attempt_log.append({"step": step, "error": str(e), "code": None, "val_acc": 0.0})
            continue

        code_str = _extract_code_from_response(raw_text)
        val_acc = 0.0
        error_msg = None

        try:
            fn_callable = compile_callable_from_code(code_str, label=f"step{step}")
            val_acc = evaluate_accuracy(fn_callable, val_lines, is_tabular)
        except Exception as e:
            error_msg = str(e)

        _save_attempt(run_dir, step, code_str, val_acc, error_msg)
        logger.info("    step %d: val_acc=%.4f%s", step, val_acc,
                     f" error={error_msg}" if error_msg else "")

        attempt_log.append({
            "step": step, "val_acc": val_acc,
            "error": error_msg, "code": code_str,
        })

        if val_acc > best_val_acc or best_code is None:
            best_val_acc = val_acc
            best_code = code_str

        prev_code = code_str
        prev_acc = val_acc
        prev_error = error_msg

        if val_acc >= 1.0:
            logger.info("    Perfect accuracy at step %d, stopping early", step)
            break

    if best_code is None:
        raise RuntimeError("Iterative agent produced no code across all steps")
    return best_code, best_val_acc, attempt_log


# ---------------------------------------------------------------------------
# MLAgentBench agent
# ---------------------------------------------------------------------------

def _setup_mlab_benchmark(run_dir: str, task_desc: str) -> str:
    mlab_root = os.path.join(PROJECT_ROOT, "external", "MLAgentBench")
    benchmarks_dir = os.path.join(mlab_root, "MLAgentBench", "benchmarks")

    bench_name = "custom_" + hashlib.sha256(run_dir.encode()).hexdigest()[:10]
    bench_dir = os.path.join(benchmarks_dir, bench_name)

    if os.path.exists(bench_dir):
        shutil.rmtree(bench_dir)

    env_dir = os.path.join(bench_dir, "env")
    scripts_dir = os.path.join(bench_dir, "scripts")
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    for fname in ("train.txt", "val.txt"):
        src = os.path.join(run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(env_dir, fname))

    research_problem = (
        "Binary classification task. Files in the current directory:\n"
        "  - train.txt: training examples, each line is '<binary_string> -> <0 or 1>'\n"
        "  - val.txt: validation examples, same format\n"
        "  - train.py: starter script that creates a baseline solution.py\n\n"
        "REQUIRED STEPS:\n"
        "1. First, execute train.py to create the baseline solution.py\n"
        "2. Read train.txt and analyze the data — try counting 1s, checking specific "
        "   bit positions, looking at substrings, parity, etc.\n"
        "3. Edit solution.py to implement the best pattern you find as `def f(x):`\n"
        "   The function takes a binary string x and must return 0 or 1.\n"
        "4. Execute solution.py to verify it runs without errors\n"
        "5. IMPORTANT: You MUST produce a solution.py with `def f(x):` even if "
        "   the pattern is unclear — try your best guess. A heuristic is better than nothing.\n"
    )
    with open(os.path.join(scripts_dir, "research_problem.txt"), "w") as f:
        f.write(research_problem)

    starter = textwrap.dedent("""\
        # Analyze training data and create baseline solution
        with open("train.txt") as fh:
            lines = [l.strip() for l in fh if "->" in l]

        print(f"Loaded {len(lines)} training examples")
        for line in lines[:5]:
            print(f"  {line}")

        xs = [l.split("->")[0].strip() for l in lines]
        ys = [int(l.split("->")[1].strip()) for l in lines]

        # Quick analysis: count of 1s vs output
        ones_when_0 = [x.count("1") for x, y in zip(xs, ys) if y == 0]
        ones_when_1 = [x.count("1") for x, y in zip(xs, ys) if y == 1]
        print(f"Avg 1s when output=0: {sum(ones_when_0)/max(len(ones_when_0),1):.1f}")
        print(f"Avg 1s when output=1: {sum(ones_when_1)/max(len(ones_when_1),1):.1f}")

        # Check if majority of 1 has higher count of 1s
        avg0 = sum(ones_when_0)/max(len(ones_when_0),1)
        avg1 = sum(ones_when_1)/max(len(ones_when_1),1)
        threshold = (avg0 + avg1) / 2

        # Baseline: threshold on count of 1s
        with open("solution.py", "w") as out:
            out.write(f"def f(x):\\n    return 1 if x.count('1') > {threshold:.1f} else 0\\n")
        print(f"Baseline solution.py written (threshold={threshold:.1f})")

        # Test baseline on training data
        correct = sum(1 for x, y in zip(xs, ys)
                      if (1 if x.count("1") > threshold else 0) == y)
        print(f"Baseline train accuracy: {correct}/{len(ys)} = {correct/len(ys):.3f}")
    """)
    with open(os.path.join(env_dir, "train.py"), "w") as f:
        f.write(starter)

    return bench_name


def _cleanup_mlab_benchmark(bench_name: str) -> None:
    mlab_root = os.path.join(PROJECT_ROOT, "external", "MLAgentBench")
    bench_dir = os.path.join(mlab_root, "MLAgentBench", "benchmarks", bench_name)
    if os.path.exists(bench_dir):
        shutil.rmtree(bench_dir)


def run_mlagentbench(run_dir: str, task_desc: str, max_steps: int, llm_name: str, timeout_s: int) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var is required for MLAgentBench")

    bench_name = _setup_mlab_benchmark(run_dir, task_desc)

    log_dir = os.path.join(run_dir, "mlab_logs")
    work_dir = os.path.join(run_dir, "mlab_workspace")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    mlab_root = os.path.join(PROJECT_ROOT, "external", "MLAgentBench")

    cmd = [
        sys.executable, "-u", "-m", "MLAgentBench.runner",
        "--python", sys.executable,
        "--task", bench_name,
        "--device", "0",
        "--log-dir", log_dir,
        "--work-dir", work_dir,
        "--llm-name", llm_name,
        "--fast-llm-name", llm_name,
        "--edit-script-llm-name", llm_name,
        "--agent-type", "ResearchAgent",
        "--agent-max-steps", str(max_steps),
        "--max-steps", str(max_steps),
    ]

    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = mlab_root + os.pathsep + env_vars.get("PYTHONPATH", "")

    stdout_path = os.path.join(run_dir, "mlab.stdout.log")
    stderr_path = os.path.join(run_dir, "mlab.stderr.log")

    try:
        with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
            subprocess.run(
                cmd, cwd=run_dir, env=env_vars,
                stdout=out_f, stderr=err_f,
                check=False, timeout=timeout_s,
            )
    finally:
        _cleanup_mlab_benchmark(bench_name)

    search_roots = [work_dir, run_dir]
    for search_root in search_roots:
        for dirpath, _dirs, files in os.walk(search_root):
            if "solution.py" in files:
                code = open(os.path.join(dirpath, "solution.py")).read()
                if "def f(" in code:
                    return code

    stderr_tail = ""
    if os.path.exists(stderr_path):
        with open(stderr_path) as ef:
            stderr_tail = ef.read()[-1500:]
    raise RuntimeError(f"MLAgentBench did not produce a solution with def f(x). stderr: {stderr_tail}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class AgentBaselineRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _run_single_trial(
        self, agent: str, fn: str, L: int, trial: int,
        train_lines: List[str], val_lines: List[str], test_lines: List[str],
        is_tabular: bool,
    ) -> Dict[str, Any]:
        run_dir = _make_run_dir(self.cfg, agent, fn, L, trial)
        paths = _materialize_task(run_dir, L, is_tabular, train_lines, val_lines)
        task_desc = open(paths["task.md"], "r").read()

        t0 = time.perf_counter()
        code = None
        error = None

        try:
            if agent == "iterative_agent":
                code, _best_val, _log = run_iterative_agent(
                    run_dir, train_lines, val_lines, L, is_tabular,
                    model=self.cfg.agent_model,
                    steps=self.cfg.agent_steps,
                )
            elif agent == "mlagentbench":
                code = run_mlagentbench(
                    run_dir, task_desc,
                    max_steps=self.cfg.mlab_max_steps,
                    llm_name=self.cfg.mlab_llm,
                    timeout_s=self.cfg.timeout_s,
                )
            else:
                raise ValueError(f"Unknown agent: {agent}")
        except Exception as e:
            error = str(e)
            logger.error("Agent %s failed on %s L=%s trial=%s: %s", agent, fn, L, trial, error)

        duration_ms = int((time.perf_counter() - t0) * 1000)

        if code is not None:
            with open(paths["solution.py"], "w", encoding="utf-8") as f:
                f.write(code)
            logger.info("  Generated code saved to %s", paths["solution.py"])
            logger.info("  --- Code ---\n%s\n  --- End ---", code)

        val_acc = 0.0
        test_acc = 0.0
        status = "ok"

        if error:
            status = "error"
        elif code is None:
            status = "no_code"
        else:
            try:
                fn_callable = compile_callable_from_code(code, label=f"{agent}/{fn}/L{L}/t{trial}")
                val_acc = evaluate_accuracy(fn_callable, val_lines, is_tabular)
                test_acc = evaluate_accuracy(fn_callable, test_lines, is_tabular)
            except Exception as e:
                status = "compile_error"
                error = str(e)

        return {
            "val_acc": val_acc,
            "test_acc": test_acc,
            "duration_ms": duration_ms,
            "status": status,
            "error": error,
            "code": code,
        }

    def run(self) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(self.cfg.out_jsonl), exist_ok=True)
        os.makedirs(self.cfg.runs_dir, exist_ok=True)

        with open(self.cfg.out_jsonl, "w", encoding="utf-8") as jsonl_file:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    logger.warning("Unknown fn=%s, skipping", fn)
                    continue
                for L in self.cfg.lengths:
                    train_lines, val_lines, test_lines, target_name = get_or_create_splits(self.cfg, fn, L)
                    is_tabular = target_name in TABULAR_FNS

                    for agent in self.cfg.agents:
                        logger.info("=== %s | %s | L=%s ===", agent, fn, L)
                        trial_results = []

                        for trial in range(self.cfg.num_trials):
                            logger.info("  Trial %s/%s", trial + 1, self.cfg.num_trials)
                            result = self._run_single_trial(
                                agent, fn, L, trial,
                                train_lines, val_lines, test_lines, is_tabular,
                            )
                            trial_results.append(result)
                            logger.info(
                                "  -> status=%s val=%.4f test=%.4f dur=%dms",
                                result["status"], result["val_acc"],
                                result["test_acc"], result["duration_ms"],
                            )

                        val_accs = [r["val_acc"] for r in trial_results]
                        test_accs = [r["test_acc"] for r in trial_results]
                        durations = [r["duration_ms"] for r in trial_results]
                        errors = [r["error"] for r in trial_results if r["error"]]
                        statuses = [r["status"] for r in trial_results]

                        row = {
                            "fn": fn,
                            "length": L,
                            "model": f"agent_{agent}",
                            "duration_ms": int(np.sum(durations)),
                            "adaptation_duration_ms": int(np.sum(durations)),
                            "test_duration_ms": 0,
                            "total_wall_clock_duration_ms": int(np.sum(durations)),
                            "val_acc": float(np.mean(val_accs)),
                            "val_acc_std": float(np.std(val_accs)),
                            "test_acc": float(np.mean(test_accs)),
                            "test_acc_std": float(np.std(test_accs)),
                            "best_params": json.dumps({
                                "statuses": statuses,
                                "errors": errors[:5],
                                "num_trials": self.cfg.num_trials,
                            }),
                            "best_cv_score": float(np.max(val_accs)),
                            "num_trials": self.cfg.num_trials,
                        }
                        all_rows.append(row)
                        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jsonl_file.flush()
        return all_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "fn", "length", "model",
        "duration_ms", "adaptation_duration_ms", "test_duration_ms", "total_wall_clock_duration_ms",
        "val_acc", "val_acc_std", "test_acc", "test_acc_std",
        "best_params", "best_cv_score", "num_trials",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Agent baseline runner (iterative OpenAI + MLAgentBench)")
    p.add_argument("--agents", nargs="*", help="Agents: iterative_agent, mlagentbench")
    p.add_argument("--functions", nargs="*", help="Function IDs (fn_a ... fn_aa)")
    p.add_argument("--lengths", nargs="*", type=int, help="Lengths (default: 100)")
    p.add_argument("--train-size", type=int)
    p.add_argument("--val-size", type=int)
    p.add_argument("--test-size", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--num-trials", type=int)
    p.add_argument("--timeout-s", type=int)
    p.add_argument("--agent-model", type=str, help="Model for iterative agent (default: gpt-4o)")
    p.add_argument("--agent-steps", type=int, help="Iterative agent improvement steps (default: 10)")
    p.add_argument("--mlab-max-steps", type=int, help="MLAgentBench max steps (default: 30)")
    p.add_argument("--mlab-llm", type=str, help="LLM for MLAgentBench (default: gpt-4)")
    p.add_argument("--dataset-dir")
    p.add_argument("--runs-dir")
    p.add_argument("--out-jsonl")
    p.add_argument("--out-csv")
    args = p.parse_args()

    cfg = Config()
    if args.agents: cfg.agents = args.agents
    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.train_size: cfg.train_size = args.train_size
    if args.val_size: cfg.val_size = args.val_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed is not None: cfg.seed = args.seed
    if args.num_trials is not None: cfg.num_trials = args.num_trials
    if args.timeout_s is not None: cfg.timeout_s = args.timeout_s
    if args.agent_model: cfg.agent_model = args.agent_model
    if args.agent_steps is not None: cfg.agent_steps = args.agent_steps
    if args.mlab_max_steps is not None: cfg.mlab_max_steps = args.mlab_max_steps
    if args.mlab_llm: cfg.mlab_llm = args.mlab_llm
    if args.dataset_dir: cfg.dataset_dir = os.path.abspath(args.dataset_dir)
    if args.runs_dir: cfg.runs_dir = os.path.abspath(args.runs_dir)
    if args.out_jsonl: cfg.out_jsonl = os.path.abspath(args.out_jsonl)
    if args.out_csv: cfg.out_csv = os.path.abspath(args.out_csv)
    return cfg


def main() -> None:
    cfg = parse_args()
    rows = AgentBaselineRunner(cfg).run()
    write_csv(cfg.out_csv, rows)
    logger.info("Results: %s and %s", cfg.out_jsonl, cfg.out_csv)


if __name__ == "__main__":
    main()
