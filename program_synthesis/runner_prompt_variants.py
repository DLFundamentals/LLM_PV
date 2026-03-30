from __future__ import annotations

import argparse
import asyncio
import ast
import csv
import hashlib
import json
import logging
import os
import random
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from openai import AsyncOpenAI

from src.data_handler import create_stratified_splits, get_data_generator
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

external_get_accuracy = None


def normalize_usage(usage_obj: Any) -> Dict[str, Any]:
    if not usage_obj:
        return {}
    if isinstance(usage_obj, Mapping):
        u = dict(usage_obj)
    else:
        u = {}
        for k in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
        ):
            v = getattr(usage_obj, k, None)
            if v is not None:
                u[k] = v

        def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
            if obj is None:
                return None
            if isinstance(obj, Mapping):
                return dict(obj)
            d: Dict[str, Any] = {}
            for kk in ("cached_tokens", "audio_tokens", "reasoning_tokens"):
                vv = getattr(obj, kk, None)
                if vv is not None:
                    d[kk] = vv
            return d or None

        ptd = getattr(usage_obj, "prompt_tokens_details", None)
        itd = getattr(usage_obj, "input_token_details", None)
        otd = getattr(usage_obj, "output_tokens_details", None)
        ctd = getattr(usage_obj, "completion_tokens_details", None)
        if (ptd := _to_dict(ptd)) is not None:
            u["prompt_tokens_details"] = ptd
        if (itd := _to_dict(itd)) is not None:
            u["input_token_details"] = itd
        if (otd := _to_dict(otd)) is not None:
            u["output_tokens_details"] = otd
        if (ctd := _to_dict(ctd)) is not None:
            u["completion_tokens_details"] = ctd

    if "prompt_tokens" not in u and "input_tokens" in u:
        u["prompt_tokens"] = u["input_tokens"]
    if "completion_tokens" not in u and "output_tokens" in u:
        u["completion_tokens"] = u["output_tokens"]

    details = u.get("prompt_tokens_details") or u.get("input_token_details") or {}
    if "cached_tokens" in details:
        u["cached_tokens"] = details["cached_tokens"]

    if "reasoning_tokens" not in u or u.get("reasoning_tokens") is None:
        rt = None
        for dkey in ("output_tokens_details", "completion_tokens_details", "input_token_details"):
            d = u.get(dkey) or {}
            if isinstance(d, Mapping):
                rt = d.get("reasoning_tokens")
            if rt is not None:
                break
        if rt is not None:
            u["reasoning_tokens"] = rt

    return u


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "ts": int(time.time() * 1000),
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        for k, v in getattr(record, "__dict__", {}).items():
            if k not in base and k not in ("msg", "args", "levelname", "name"):
                base[k] = v
        return json.dumps(base, ensure_ascii=False)


def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("runner_openai_prompt_variants")
    logger.setLevel(level.upper())

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())

    file_handler = logging.FileHandler("program_synthesis/runner_openai_prompt_variants.log", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    logger.handlers[:] = [stream_handler, file_handler]
    logger.propagate = False
    return logger


DEFAULT_PROMPT_VARIANTS: List[Dict[str, str]] = [
    {
        "id": "prompt_1",
        "problem_statement": (
            "Given a sequence of input vectors ({data_mode}, length {seq_len}) mapped to scalar binary outputs, "
            "extract the underlying relationship as a concise Python function `f(x)`. "
            "The solution must be a direct logical or mathematical expression, not a machine learning model."
        )
    },
    {
        "id": "prompt_2",
        "problem_statement": (
            "Analyze the provided input vectors ({data_mode}, length {seq_len}) and their corresponding binary outputs "
            "to determine the governing logic. Express this logic as a short, deterministic Python function `f(x)` "
            "using mathematical or logical operations, avoiding trainable parameters."
        )
    },
    {
        "id": "prompt_3",
        "problem_statement": (
            "Identify the mapping between the input vectors ({data_mode}, length {seq_len}) and binary scalar outputs. "
            "Represent this mapping through a concise, stateless Python function `f(x)` that relies purely "
            "on explicit mathematical or logical rules rather than learned weights."
        )
    },
    {
        "id": "prompt_4",
        "problem_statement": (
            "Discover the strict mathematical or logical rule that maps the input vectors ({data_mode}, length {seq_len}) "
            "to their binary outputs. Output a concise Python function `f(x)` that formally encodes this rule "
            "without relying on any trainable architecture."
        )
    },
]


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = os.getenv("OPENAI_MODEL", "gpt-5")
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "20000"))
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "high")
    verbosity: Optional[str] = os.getenv("TEXT_VERBOSITY", "low")
    tool_choice: str = os.getenv("TOOL_CHOICE", "auto")
    enable_code_interpreter: bool = os.getenv("ENABLE_CODE_INTERPRETER", "1") == "1"

    dry_run: bool = os.getenv("DRY_RUN", "0") == "1"
    concurrency: int = int(os.getenv("CONCURRENCY", "5"))
    per_call_timeout_s: float = float(os.getenv("PER_CALL_TIMEOUT_S", "1200"))

    functions: List[str] = field(
        default_factory=lambda: [
            "fn_a",
            "fn_b",
            "fn_c",
            "fn_d",
            "fn_e",
            "fn_f",
            "fn_g",
            "fn_h",
            "fn_i",
            "fn_j",
            "fn_k",
            "fn_l",
            "fn_v",
            "fn_t",
            "fn_m",
            "fn_n",
            "fn_o",
            "fn_p",
            "fn_q",
            "fn_aa",
        ]
    )
    lengths: List[int] = field(default_factory=lambda: [100, 50, 30, 25, 20])
    lengths_explicit: bool = False
    attempts: int = int(os.getenv("ATTEMPTS", "5"))
    num_trials: int = int(os.getenv("NUM_TRIALS", "1"))

    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int = int(os.getenv("VAL_SIZE", "100"))
    test_size: int = int(os.getenv("TEST_SIZE", "10000"))
    seed: int = int(os.getenv("GLOBAL_SEED", "42"))
    dataset_dir: str = os.getenv("DATASET_DIR", "program_synthesis/datasets")

    prompt_variants: List[Dict[str, str]] = field(default_factory=lambda: list(DEFAULT_PROMPT_VARIANTS))

    out_jsonl: str = os.getenv("OUT_JSONL", "program_synthesis/prompt_variants/results_prompt_variants.jsonl")
    out_csv: str = os.getenv("OUT_CSV", "program_synthesis/prompt_variants/results_prompt_variants.csv")


FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING
DECIMAL_FNS = {"prime_decimal", "prime_decimal_tf_check", "prime_plus_47", "collatz_steps_parity"}
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}


def build_user_prompt(
    data_examples: List[str],
    seq_len: int,
    prompt_problem_statement: str,
    decimal: bool = False,
    tabular: bool = False,
) -> str:
    template = prompt_problem_statement.strip()
    if tabular:
        # Match old prompt style for tabular tasks: no "(..., length {seq_len})" phrase.
        template = template.replace("({data_mode}, length {seq_len})", "{data_mode}")
        template = template.replace("({data_mode},length {seq_len})", "{data_mode}")
        data_mode_hint = "tabular input data (comma-separated feature:value pairs)"
    elif decimal:
        data_mode_hint = "decimal"
    else:
        data_mode_hint = "binary"

    # Safe placeholder replacement without str.format, so literal braces remain intact.
    variant_statement = template.replace("{seq_len}", str(seq_len)).replace("{data_mode}", data_mode_hint).strip()
    prompt = f"**Problem Statement:**\n{variant_statement}\n"
    prompt += "**Data Examples:**\n```\n" + "\n".join(data_examples) + "\n```\n\n"
    prompt += 'You must output ONLY a single JSON object: {"code": "<python function>"}.\n'
    return prompt


def _safe_write_text_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tmp:
        for ln in lines:
            tmp.write(f"{ln}\n")
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _safe_write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=os.path.dirname(path)) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


class DatasetStore:
    def __init__(self, cfg: Config, log: logging.Logger):
        self.cfg = cfg
        self.log = log

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except Exception:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def _paths(self, target_name: str, L: int, derived_seed: int) -> Dict[str, str]:
        base = os.path.join(self.cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
        return {
            "dir": base,
            "train": os.path.join(base, "train.txt"),
            "val": os.path.join(base, "val.txt"),
            "test": os.path.join(base, "test.txt"),
            "meta": os.path.join(base, "meta.json"),
        }

    def _stable_derived_seed(self, fn: str, L: int) -> int:
        key = (
            f"{fn}|L={L}|train={self.cfg.train_size + self.cfg.val_size}|"
            f"test={self.cfg.test_size}|base_seed={self.cfg.seed}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF

    def _ensure_splits(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool, bool]:
        target_name = FUNCTION_NAME_MAPPING[fn]
        is_decimal = target_name in DECIMAL_FNS
        is_tabular = target_name in TABULAR_FNS

        derived_seed = self._stable_derived_seed(fn, L)
        paths = self._paths(target_name, L, derived_seed)

        def exists_with_size(path: str, expect: int) -> bool:
            if not os.path.exists(path):
                return False
            try:
                return sum(1 for _ in open(path, "r", encoding="utf-8")) == expect
            except Exception:
                return False

        if (
            exists_with_size(paths["train"], self.cfg.train_size)
            and exists_with_size(paths["val"], self.cfg.val_size)
            and exists_with_size(paths["test"], self.cfg.test_size)
        ):
            self.log.info("dataset_reused", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
            return (
                _read_lines(paths["train"]),
                _read_lines(paths["val"]),
                _read_lines(paths["test"]),
                is_decimal,
                is_tabular,
            )

        self._set_seed(derived_seed)
        self.log.info("dataset_generating", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})

        total_samples = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        generator = get_data_generator(target_name, L, total_samples)
        all_samples_dicts = generator.generate_data()

        train_split_dicts, val_split_dicts, test_split_dicts = create_stratified_splits(
            all_samples=all_samples_dicts,
            train_size=self.cfg.train_size,
            val_size=self.cfg.val_size,
            test_size=self.cfg.test_size,
            device="cpu",
        )

        train_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in train_split_dicts]
        val_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in val_split_dicts]
        test_lines = [f"{''.join(s['Input'])} -> {s['Output']}" for s in test_split_dicts]

        random.shuffle(train_lines)
        random.shuffle(val_lines)

        _safe_write_text_lines(paths["train"], train_lines)
        _safe_write_text_lines(paths["val"], val_lines)
        _safe_write_text_lines(paths["test"], test_lines)
        _safe_write_json(
            paths["meta"],
            {
                "fn": fn,
                "target_name": target_name,
                "length": L,
                "decimal": is_decimal,
                "tabular": is_tabular,
                "derived_seed": derived_seed,
                "sizes": {
                    "train": self.cfg.train_size,
                    "val": self.cfg.val_size,
                    "test": self.cfg.test_size,
                },
                "created_ts": int(time.time()),
            },
        )

        self.log.info("dataset_written", extra={"fn": fn, "length": L, "seed": derived_seed, "dir": paths["dir"]})
        return train_lines, val_lines, test_lines, is_decimal, is_tabular

    def get(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str], bool, bool]:
        return self._ensure_splits(fn, L)


def extract_code_from_output(output_text: str) -> Optional[str]:
    if not output_text:
        return None
    try:
        obj = json.loads(output_text)
        if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
            return obj["code"]
    except Exception:
        pass
    m = re.search(r"\{.*\}", output_text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "code" in obj and isinstance(obj["code"], str):
                return obj["code"]
        except Exception:
            return None
    return None


def compile_callable_from_code(code_str: str) -> Callable[[str], int]:
    code_str = textwrap.dedent(code_str.strip())
    if code_str.startswith("```"):
        code_str = re.sub(r"^```(?:python)?\s*|\s*```$", "", code_str, flags=re.IGNORECASE | re.DOTALL)
    tree = ast.parse(code_str)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise ValueError("No function definition found in generated code.")
    prefer_name = "f" if "f" in fn_names else fn_names[0]
    local_ns: Dict[str, Any] = {}
    safe_globals = {"__builtins__": __builtins__}
    exec(compile(tree, filename="<generated>", mode="exec"), safe_globals, local_ns)
    fn = local_ns.get(prefer_name)
    if not callable(fn):
        raise ValueError(f"Function '{prefer_name}' not found after exec.")
    return fn


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
            v = int(float(s))
            return 1 if v != 0 else 0
        except Exception:
            return 1 if len(s) > 0 else 0

    return 1 if pred else 0


def _parse_tabular_input(x_str: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in x_str.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            k = k.strip()
            v = v.strip()
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
    return result


def _local_get_accuracy(
    fn_callable: Callable[[str], int],
    data_lines: List[str],
    logger: Optional[logging.Logger] = None,
    is_tabular: bool = False,
) -> float:
    if not data_lines:
        return 0.0
    correct = 0
    errors = 0
    for line in data_lines:
        try:
            x, y = line.split("->")
            x = x.strip()
            y_int = int(y.strip())
            if is_tabular:
                x = _parse_tabular_input(x)
            pred = fn_callable(x)
            pred_int = _normalize_pred_to01(pred)
            correct += int(pred_int == y_int)
        except Exception as e:
            if errors < 5 and logger is not None:
                logger.debug("evaluation_error", extra={"line": line, "error": str(e)})
            errors += 1
    if errors > 0 and logger is not None:
        logger.warning("evaluation_errors", extra={"count": errors})
    return correct / len(data_lines)


def evaluate_accuracy(
    fn_callable: Callable[[str], int],
    data_lines: List[str],
    logger: Optional[logging.Logger] = None,
    is_tabular: bool = False,
) -> float:
    if external_get_accuracy is not None:
        try:
            return float(external_get_accuracy(fn_callable, data_lines))
        except Exception:
            pass
    return _local_get_accuracy(fn_callable, data_lines, logger, is_tabular)


class Runner:
    def __init__(self, cfg: Config, logger: logging.Logger):
        if not cfg.api_key:
            raise SystemExit("OPENAI_API_KEY is required.")
        if not cfg.prompt_variants:
            raise SystemExit("At least one prompt variant is required.")
        self.cfg = cfg
        self.log = logger
        self.client = AsyncOpenAI(api_key=cfg.api_key)
        self.sem = asyncio.Semaphore(cfg.concurrency)
        self.tools: List[Dict[str, Any]] = []
        if cfg.enable_code_interpreter:
            self.tools.append({"type": "code_interpreter", "container": {"type": "auto"}})
        self.ds = DatasetStore(cfg, logger)

    async def aclose(self) -> None:
        await self.client.close()

    async def _call_once(
        self,
        fn: str,
        L: int,
        attempt_idx: int,
        prompt_variant_id: str,
        prompt_variant_text: str,
        data_examples: List[str],
        decimal: bool,
        tabular: bool = False,
    ) -> Dict[str, Any]:
        prompt_text = build_user_prompt(
            data_examples=data_examples,
            seq_len=L,
            prompt_problem_statement=prompt_variant_text,
            decimal=decimal,
            tabular=tabular,
        )
        body_preview_size = len(
            json.dumps({"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}]})
        )
        body: Dict[str, Any] = {
            "model": self.cfg.model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt_text}]}],
            "reasoning": {"effort": self.cfg.reasoning_effort},
            "max_output_tokens": self.cfg.max_output_tokens,
            "tool_choice": self.cfg.tool_choice,
        }
        if self.tools:
            body["tools"] = self.tools
        if self.cfg.verbosity:
            body["text"] = {"verbosity": self.cfg.verbosity}

        if self.cfg.dry_run:
            self.log.info(
                "dry_run_input",
                extra={
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "prompt_variant_id": prompt_variant_id,
                    "prompt_preview": prompt_text,
                },
            )
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt_variant_id": prompt_variant_id,
                "prompt_variant": prompt_variant_text,
                "prompt": prompt_text,
                "request_body": body,
                "text": None,
                "usage": {},
                "cached_tokens": 0,
                "duration_ms": 0,
            }

        async def _try_call(tag: str) -> Dict[str, Any]:
            t0 = time.perf_counter()
            async with self.sem:
                res = await asyncio.wait_for(self.client.responses.create(**body), timeout=self.cfg.per_call_timeout_s)
                tool_uses = 0
                tool_results_chars = 0
                for item in getattr(res, "output", []) or []:
                    t = getattr(item, "type", None)
                    if t == "tool_use":
                        tool_uses += 1
                    elif t == "tool_result":
                        content = getattr(item, "content", None)
                        if isinstance(content, list):
                            for part in content:
                                txt = (
                                    getattr(part, "text", None)
                                    if hasattr(part, "text")
                                    else part.get("text")
                                    if isinstance(part, dict)
                                    else None
                                )
                                if isinstance(txt, str):
                                    tool_results_chars += len(txt)
                        elif isinstance(content, str):
                            tool_results_chars += len(content)

            dt_ms = int((time.perf_counter() - t0) * 1000)
            usage = normalize_usage(getattr(res, "usage", {}))
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            out_text = (getattr(res, "output_text", "") or "").strip()
            self.log.info(
                tag,
                extra={
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "prompt_variant_id": prompt_variant_id,
                    "duration_ms": dt_ms,
                    "prompt_chars": len(prompt_text),
                    "request_body_bytes": len(json.dumps(body)),
                    "input_section_bytes": body_preview_size,
                    "tools_enabled": bool(self.tools),
                    "tool_count": len(self.tools),
                    "tool_uses": tool_uses,
                    "tool_results_chars": tool_results_chars,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
                    "cached_tokens": cached,
                    "completion_tokens": usage.get("completion_tokens"),
                },
            )
            return {
                "fn": fn,
                "length": L,
                "attempt": attempt_idx,
                "prompt_variant_id": prompt_variant_id,
                "prompt_variant": prompt_variant_text,
                "prompt": prompt_text,
                "text": out_text,
                "usage": usage,
                "cached_tokens": cached,
                "duration_ms": dt_ms,
                "request_body_bytes": len(json.dumps(body)),
                "prompt_chars": len(prompt_text),
                "tool_uses": tool_uses,
                "tool_results_chars": tool_results_chars,
            }

        try:
            return await _try_call("attempt_ok")
        except Exception as e1:
            self.log.warning(
                "attempt_retry_once",
                extra={"fn": fn, "length": L, "attempt": attempt_idx, "prompt_variant_id": prompt_variant_id, "error": str(e1)},
            )
            try:
                return await _try_call("attempt_ok_after_retry")
            except Exception as e2:
                self.log.error(
                    "attempt_failed",
                    extra={"fn": fn, "length": L, "attempt": attempt_idx, "prompt_variant_id": prompt_variant_id, "error": str(e2)},
                )
                return {
                    "fn": fn,
                    "length": L,
                    "attempt": attempt_idx,
                    "prompt_variant_id": prompt_variant_id,
                    "prompt_variant": prompt_variant_text,
                    "error": str(e2),
                }

    async def run(self) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []

        base_jsonl_path = self.cfg.out_jsonl
        base_name, ext = os.path.splitext(base_jsonl_path)
        os.makedirs(os.path.dirname(base_jsonl_path) if os.path.dirname(base_jsonl_path) else ".", exist_ok=True)
        final_jsonl_file = open(base_jsonl_path, "w", encoding="utf-8")

        try:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    self.log.error("unknown_function", extra={"fn": fn})
                    continue

                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = self.cfg.lengths if self.cfg.lengths_explicit else task_meta.get("lengths", self.cfg.lengths)
                for L in current_lengths:
                    train_lines, val_lines, test_lines, is_decimal, is_tabular = self.ds.get(fn, L)

                    for variant_idx, variant in enumerate(self.cfg.prompt_variants, start=1):
                        variant_id = str(variant.get("id") or f"prompt_{variant_idx}")
                        variant_text = str(variant.get("problem_statement") or variant.get("instructions") or "").strip()
                        if not variant_text:
                            self.log.warning("prompt_variant_skipped_empty", extra={"fn": fn, "length": L, "prompt_variant_id": variant_id})
                            continue

                        trial_test_accuracies: List[float] = []
                        trial_val_accuracies: List[float] = []
                        trial_best_rows: List[Dict[str, Any]] = []

                        for trial in range(self.cfg.num_trials):
                            trial_jsonl_path = f"{base_name}_{fn}_{variant_id}_L{L}_trial{trial + 1}{ext}"
                            trial_jsonl_file = open(trial_jsonl_path, "w", encoding="utf-8")
                            try:
                                best_test_acc = -1.0
                                best_val_acc: Optional[float] = None
                                best_row: Optional[Dict[str, Any]] = None
                                stopped_early = False

                                for k in range(1, self.cfg.attempts + 1):
                                    res = await self._call_once(
                                        fn=fn,
                                        L=L,
                                        attempt_idx=k,
                                        prompt_variant_id=variant_id,
                                        prompt_variant_text=variant_text,
                                        data_examples=train_lines,
                                        decimal=is_decimal,
                                        tabular=is_tabular,
                                    )
                                    out_text = res.get("text") or ""
                                    code_str = extract_code_from_output(out_text)

                                    val_acc = None
                                    test_acc = None
                                    compile_error = None

                                    if code_str:
                                        try:
                                            fn_callable = compile_callable_from_code(code_str)
                                            val_acc = evaluate_accuracy(fn_callable, val_lines, self.log, is_tabular)
                                            test_acc = evaluate_accuracy(fn_callable, test_lines, self.log, is_tabular)

                                            if test_acc > best_test_acc:
                                                best_test_acc = test_acc
                                                best_val_acc = val_acc
                                                best_row = {
                                                    **res,
                                                    "val_acc": val_acc,
                                                    "test_acc": test_acc,
                                                    "stopped_early": stopped_early,
                                                    "compile_error": None,
                                                    "trial": trial + 1,
                                                    "prompt_variant_index": variant_idx,
                                                }

                                            if val_acc == 1.0:
                                                stopped_early = True
                                                if best_row:
                                                    best_row["stopped_early"] = True
                                        except Exception as e:
                                            compile_error = str(e)
                                            self.log.warning(
                                                "compile_or_eval_error",
                                                extra={
                                                    "fn": fn,
                                                    "length": L,
                                                    "attempt": k,
                                                    "trial": trial + 1,
                                                    "prompt_variant_id": variant_id,
                                                    "error": compile_error,
                                                },
                                            )
                                    else:
                                        compile_error = "no_code_found"

                                    row = {
                                        **res,
                                        "val_acc": val_acc,
                                        "test_acc": test_acc,
                                        "stopped_early": stopped_early,
                                        "compile_error": compile_error,
                                        "trial": trial + 1,
                                        "prompt_variant_index": variant_idx,
                                    }
                                    all_rows.append(row)
                                    trial_jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                                    trial_jsonl_file.flush()

                                    if stopped_early:
                                        self.log.info(
                                            "early_stop",
                                            extra={
                                                "fn": fn,
                                                "length": L,
                                                "attempt": k,
                                                "trial": trial + 1,
                                                "prompt_variant_id": variant_id,
                                                "val_acc": val_acc,
                                                "test_acc": test_acc,
                                            },
                                        )
                                        break

                                if best_test_acc >= 0 and best_row:
                                    trial_test_accuracies.append(best_test_acc)
                                    trial_val_accuracies.append(best_val_acc if best_val_acc is not None else 0.0)
                                    trial_best_rows.append(best_row)
                            finally:
                                trial_jsonl_file.close()

                        for best_row in trial_best_rows:
                            final_jsonl_file.write(json.dumps(best_row, ensure_ascii=False) + "\n")
                            final_jsonl_file.flush()

                        if trial_test_accuracies:
                            import numpy as np

                            test_acc_mean = float(np.mean(trial_test_accuracies))
                            test_acc_std = float(np.std(trial_test_accuracies))
                            val_acc_mean = float(np.mean(trial_val_accuracies))
                            val_acc_std = float(np.std(trial_val_accuracies))

                            summary_row = {
                                "fn": fn,
                                "length": L,
                                "attempt": None,
                                "trial": None,
                                "prompt": None,
                                "text": None,
                                "duration_ms": None,
                                "cached_tokens": None,
                                "usage": {},
                                "tool_uses": None,
                                "tool_results_chars": None,
                                "prompt_variant_id": variant_id,
                                "prompt_variant": variant_text,
                                "prompt_variant_index": variant_idx,
                                "val_acc": val_acc_mean,
                                "val_acc_std": val_acc_std,
                                "test_acc": test_acc_mean,
                                "test_acc_std": test_acc_std,
                                "stopped_early": None,
                                "compile_error": None,
                                "num_trials": self.cfg.num_trials,
                                "is_summary": True,
                            }
                            all_rows.append(summary_row)
                            final_jsonl_file.write(json.dumps(summary_row, ensure_ascii=False) + "\n")
                            final_jsonl_file.flush()

                            self.log.info(
                                "prompt_variant_summary",
                                extra={
                                    "fn": fn,
                                    "length": L,
                                    "prompt_variant_id": variant_id,
                                    "test_acc_mean": test_acc_mean,
                                    "test_acc_std": test_acc_std,
                                    "val_acc_mean": val_acc_mean,
                                    "val_acc_std": val_acc_std,
                                    "num_trials": self.cfg.num_trials,
                                },
                            )

            self.log.info("dispatch_finished", extra={"total_results": len(all_rows)})
        finally:
            final_jsonl_file.close()

        return all_rows


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fieldnames = [
        "fn",
        "length",
        "prompt_variant_index",
        "prompt_variant_id",
        "prompt_variant",
        "attempt",
        "trial",
        "prompt",
        "text",
        "duration_ms",
        "cached_tokens",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "tool_uses",
        "tool_results_chars",
        "val_acc",
        "val_acc_std",
        "test_acc",
        "test_acc_std",
        "stopped_early",
        "compile_error",
        "num_trials",
        "is_summary",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            usage = r.get("usage") or {}
            w.writerow(
                {
                    "fn": r.get("fn"),
                    "length": r.get("length"),
                    "prompt_variant_index": r.get("prompt_variant_index"),
                    "prompt_variant_id": r.get("prompt_variant_id"),
                    "prompt_variant": r.get("prompt_variant"),
                    "attempt": r.get("attempt"),
                    "trial": r.get("trial"),
                    "prompt": r.get("prompt"),
                    "text": r.get("text"),
                    "duration_ms": r.get("duration_ms"),
                    "cached_tokens": r.get("cached_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "tool_uses": r.get("tool_uses"),
                    "tool_results_chars": r.get("tool_results_chars"),
                    "val_acc": r.get("val_acc"),
                    "val_acc_std": r.get("val_acc_std"),
                    "test_acc": r.get("test_acc"),
                    "test_acc_std": r.get("test_acc_std"),
                    "stopped_early": r.get("stopped_early"),
                    "compile_error": r.get("compile_error"),
                    "num_trials": r.get("num_trials"),
                    "is_summary": r.get("is_summary"),
                }
            )


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_best_per_pair(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for r in rows:
        if r.get("is_summary"):
            continue
        test_acc = r.get("test_acc")
        if not isinstance(test_acc, (int, float)):
            continue
        key = (r.get("fn"), r.get("length"), r.get("prompt_variant_id"))
        prev = best_by_key.get(key)
        if prev is None or float(test_acc) > float(prev.get("test_acc", float("-inf"))):
            best_by_key[key] = r
    return [best_by_key[k] for k in sorted(best_by_key.keys(), key=lambda x: (str(x[0]), int(x[1]), str(x[2])))]


def _derive_suffixed_path(path: str, suffix: str) -> str:
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".jsonl" if suffix.endswith(".jsonl") else ".csv"
    return f"{base}{suffix}"


def _load_prompt_variants_json(raw_json: str) -> List[Dict[str, str]]:
    obj = json.loads(raw_json)
    if not isinstance(obj, list):
        raise ValueError("Prompt variants must be a JSON list.")

    variants: List[Dict[str, str]] = []
    for idx, item in enumerate(obj, start=1):
        if isinstance(item, str):
            text = item.strip()
            if text:
                variants.append({"id": f"prompt_{idx}", "problem_statement": text})
            continue
        if isinstance(item, Mapping):
            pid = str(item.get("id") or f"prompt_{idx}").strip()
            text = str(item.get("problem_statement") or item.get("instructions") or "").strip()
            if text:
                variants.append({"id": pid, "problem_statement": text})
            continue
        raise ValueError("Each prompt variant must be either a string or an object.")

    if not variants:
        raise ValueError("No non-empty prompt variants provided.")
    return variants


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="OpenAI prompt-variant runner with persistent datasets")

    p.add_argument("--functions", nargs="*", help="Function IDs (e.g., fn_a fn_b ...)")
    p.add_argument("--lengths", nargs="*", type=int, help="Sequence lengths (e.g., 100 50 30 25 20)")
    p.add_argument("--attempts", type=int, help="Attempts per (fn, length, prompt)")
    p.add_argument("--num-trials", type=int, help="Number of trials per (fn, length, prompt)")
    p.add_argument("--concurrency", type=int, help="Max concurrent API calls")
    p.add_argument("--timeout", type=float, help="Per-call timeout seconds")

    p.add_argument("--model", help="Model name (default: gpt-5)")
    p.add_argument("--max-output-tokens", type=int, help="Max output tokens")
    p.add_argument("--enable-code-interpreter", action="store_true", help="Enable Code Interpreter tool")
    p.add_argument("--tool-choice", choices=["auto", "none"], help="Tool choice")
    p.add_argument("--verbosity", choices=["low", "medium", "high"], help="text.verbosity")
    p.add_argument("--reasoning-effort", choices=["minimal", "medium", "high"], help="reasoning.effort")

    p.add_argument("--prompt-variants-json", type=str, help="JSON list of prompt variants")
    p.add_argument("--train-size", type=int, help="Train size")
    p.add_argument("--val-size", type=int, help="Validation size")
    p.add_argument("--test-size", type=int, help="Test size")
    p.add_argument("--seed", type=int, help="Global seed")
    p.add_argument("--dataset-dir", type=str, help="Dataset directory")

    p.add_argument("--out-jsonl", help="Output JSONL path")
    p.add_argument("--out-csv", help="Output CSV path")
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level")
    p.add_argument("--dry-run", action="store_true", help="Dry run (print prompt and body in logs)")

    args = p.parse_args()
    cfg = Config()

    if args.functions:
        cfg.functions = args.functions
    if args.lengths:
        cfg.lengths = args.lengths
        cfg.lengths_explicit = True
    if args.attempts:
        cfg.attempts = args.attempts
    if args.num_trials:
        cfg.num_trials = args.num_trials
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.model:
        cfg.model = args.model
    if args.max_output_tokens:
        cfg.max_output_tokens = args.max_output_tokens
    if args.enable_code_interpreter:
        cfg.enable_code_interpreter = True
    if args.tool_choice:
        cfg.tool_choice = args.tool_choice
    if args.out_jsonl:
        cfg.out_jsonl = args.out_jsonl
    if args.out_csv:
        cfg.out_csv = args.out_csv
    if args.verbosity:
        cfg.verbosity = args.verbosity
    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort
    if args.timeout:
        cfg.per_call_timeout_s = args.timeout
    if args.train_size:
        cfg.train_size = args.train_size
    if args.val_size:
        cfg.val_size = args.val_size
    if args.test_size:
        cfg.test_size = args.test_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.dry_run:
        cfg.dry_run = True
    if args.prompt_variants_json:
        cfg.prompt_variants = _load_prompt_variants_json(args.prompt_variants_json)

    if cfg.enable_code_interpreter and cfg.tool_choice == "none":
        cfg.tool_choice = "auto"

    os.environ["LOG_LEVEL"] = args.log_level
    return cfg


async def _amain(cfg: Config) -> None:
    log = setup_logger(os.getenv("LOG_LEVEL", "INFO"))
    runner = Runner(cfg, log)
    try:
        rows = await runner.run()
    finally:
        await runner.aclose()
    write_csv(cfg.out_csv, rows)
    best_rows = select_best_per_pair(rows)
    best_csv_path = _derive_suffixed_path(cfg.out_csv, "_best_per_pair.csv")
    best_jsonl_path = _derive_suffixed_path(cfg.out_jsonl, "_best_per_pair.jsonl")
    write_csv(best_csv_path, best_rows)
    write_jsonl(best_jsonl_path, best_rows)
    log.info(
        "artifacts_written",
        extra={
            "jsonl": cfg.out_jsonl,
            "csv": cfg.out_csv,
            "best_per_pair_csv": best_csv_path,
            "best_per_pair_jsonl": best_jsonl_path,
            "best_rows": len(best_rows),
        },
    )


def main() -> None:
    cfg = parse_args()
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
