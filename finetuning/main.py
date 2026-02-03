# /finetuning/main.py

import argparse
import os
import sys
import logging
import gc
from typing import Dict, Any, Tuple, Callable
import json

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from itertools import islice

from . import utils, models
from .dataloaders import CodeDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.target_functions import TARGET_FUNCTIONS, EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

# =============================================================================
# Helper Functions
# =============================================================================

def setup_environment(seed):
    """Configures Torch backends and sets the random seed."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    utils.set_seed(seed)

def setup_logger(log_file_name="job_log.log", log_level=logging.INFO) -> logging.Logger:
    """Sets up a logger that writes to a file and the console."""
    os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file_name, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger

def log_model_architecture(model: nn.Module, logger: logging.Logger):
    """Logs the model's architecture, config, and trainable parameters."""
    try:
        logger.info("\n===== MODEL ARCHITECTURE (repr) =====\n%s", repr(model))
        base = getattr(model, "model", model)
        if hasattr(base, "config"):
            cfg_str = base.config.to_json_string(use_diff=False) if hasattr(base.config, "to_json_string") else str(base.config)
            logger.info("\n===== MODEL CONFIG =====\n%s", cfg_str)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("\n===== PARAMETER COUNT =====")
        logger.info(f"Trainable Parameters: {trainable_params:,}")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Percentage of Trainable Params: {100 * trainable_params / total_params:.4f}%")
    except Exception as e:
        logger.warning("Failed during model logging: %s", e)

class Metrics:
    def __init__(self):
        self.train_losses, self.test_losses = [], []
        self.train_accuracies, self.test_accuracies = [], []

    def update(self, train_loss, test_loss, train_acc, test_acc):
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)

# =============================================================================
# Core Training & Evaluation Logic
# =============================================================================
def evaluate_model(model, dataloader, device, model_name, amp_dtype):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    is_finetune = "finetune" in model_name
    is_binary = model_name in ['llama3', 'deepseek', 'qwen3BCoder', 'qwen7BCoder', 'qwen1.5BCoder', 'qwen1.7B', 'qwen1.5B', 'qwen0.6B', 'bloom', 'deberta', 'mlp'] or is_finetune
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids').to(device)
                attention_mask = batch.get('attention_mask')
                labels = batch.get('labels')
                if attention_mask is not None: attention_mask = attention_mask.to(device)
            else:
                input_ids = batch[0].to(device)
                labels = batch[1]
                attention_mask = None
            with autocast(device_type='cuda', dtype=amp_dtype):
                preds = model(input_ids, attention_mask=attention_mask)
                if is_binary:
                    labels = labels.to(device, dtype=torch.float)
                    loss = criterion(preds, labels.view(-1, 1))
                    probs = torch.sigmoid(preds)
                    preds_bin = (probs >= 0.5).float()
                    total_acc += (preds_bin == labels.view(-1, 1)).float().sum().item()
                else:
                    labels = labels.to(device, dtype=torch.long)
                    logits = preds[:, -1] if preds.dim() == 3 else preds
                    loss = criterion(logits, labels)
                    total_acc += (logits.argmax(1) == labels).float().sum().item()
            total_loss += loss.item() * input_ids.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_acc / len(dataloader.dataset)
    return avg_loss, avg_acc

def train_epoch(model, optimizer, scheduler, dataloader, device, train_iters, scaler, model_name, amp_dtype):
    model.train()
    is_finetune = "finetune" in model_name
    is_binary = model_name in ['llama3', 'deepseek', 'qwen3BCoder', 'qwen7BCoder', 'qwen1.5BCoder', 'qwen1.7B', 'qwen1.5B', 'qwen0.6B', 'bloom', 'deberta', 'mlp'] or is_finetune
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    for batch in islice(dataloader, train_iters):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids').to(device)
            attention_mask = batch.get('attention_mask')
            labels = batch.get('labels')
            if attention_mask is not None: attention_mask = attention_mask.to(device)
        else:
            input_ids = batch[0].to(device)
            labels = batch[1]
            attention_mask = None
        with autocast(device_type='cuda', dtype=amp_dtype):
            preds = model(input_ids, attention_mask=attention_mask)
            if is_binary:
                labels = labels.to(device, dtype=torch.float)
                loss = criterion(preds, labels.view(-1, 1))
            else:
                labels = labels.to(device, dtype=torch.long)
                logits = preds[:, -1] if preds.dim() == 3 else preds
                loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

# =============================================================================
# Configuration & Orchestration
# =============================================================================

def _get_task_config_from_args(args) -> Tuple[Dict[str, Any], Callable, str]:
    """Adapter function to translate from args to config dict for CodeDataset."""
    task_config = {}
    tokenizer_name = None
    target_func_id = args.target_func # e.g., 'fn_g'
    
    if target_func_id not in EXPERIMENT_FUNCTION_MAPPING:
        raise ValueError(f"Target function ID '{target_func_id}' is not supported. Use one of {list(EXPERIMENT_FUNCTION_MAPPING.keys())}.")
    
    # 1. Map from experiment ID to target function name (e.g., 'fn_g' -> 'palindrome')
    target_func_name = EXPERIMENT_FUNCTION_MAPPING[target_func_id]
    
    # 2. Map from target function name to the actual callable function object
    if target_func_name not in TARGET_FUNCTIONS:
         raise ValueError(f"Internal error: '{target_func_name}' not found in TARGET_FUNCTIONS registry.")
    python_code = TARGET_FUNCTIONS[target_func_name]

    if "finetune" in args.model:
        if "llama3" in args.model: tokenizer_name = "meta-llama/Llama-3.2-1B"
        elif "qwen" in args.model: tokenizer_name = "Qwen/Qwen3-1.7B"
        elif "deepseek" in args.model: tokenizer_name = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # This special logic now needs to use the target_func_name
    if target_func_name == 'patternmatch1': task_config['pattern'] = '10101010'
    elif target_func_name == 'patternmatch2': task_config['pattern'] = '00111111'
    elif target_func_name == 'palindrome': task_config['palindrome'] = True
    elif target_func_name == 'dyck2': task_config['dyck2'] = True
    elif target_func_name == 'prime_decimal': task_config['prime'] = True
    elif target_func_name == 'prime_decimal_tf_check': task_config['prime_odd'] = True
        
    return task_config, python_code, tokenizer_name

def load_data(args, logger):
    """Loads and prepares the dataset and dataloaders based on args."""
    logger.info("Setting up dataset...")
    task_config, python_code, tokenizer_name = _get_task_config_from_args(args)
    
    sequence_length = args.sequence_length
    task_meta = EXPERIMENT_FUNCTION_METADATA.get(args.target_func, {})
    if "lengths" in task_meta:
        required_lengths = task_meta["lengths"]
        if sequence_length not in required_lengths:
            logger.info(f"Auto-detected sequence_length {required_lengths[0]} for {args.target_func} (metadata specifies: {required_lengths})")
            sequence_length = required_lengths[0]
        else:
            logger.info(f"Using provided sequence_length {sequence_length} for {args.target_func}")
    elif args.target_func == "fn_aa":
        if sequence_length % 4 != 0:
            raise ValueError(f"fn_aa (graph_has_cycle) requires sequence_length to be a multiple of 4, got {sequence_length}")
    
    dataset = CodeDataset(
        python_code=python_code, sequence_length=sequence_length,
        train_set_size=args.train_set_size, test_set_size=args.test_set_size,
        batch_size=args.batch_size, bos_token=args.BOS_TOKEN,
        online=args.online, device=args.device, logger=logger,
        tokenizer_name=tokenizer_name, global_seed=args.seed,
        fn_id=args.target_func, val_set_size=0,
        test_sequence_length=getattr(args, 'test_sequence_length', None),
        **task_config
    )
    return dataset.create_dataloaders()

def load_model_and_optimizer(args, logger, train_loader):
    """Initializes the model, optimizer, and scheduler."""
    logger.info(f"Initializing model: {args.model}")
    # Pass the args object to the model factory
    model = models.get_model(args).to(args.device)
    
    if "bloom" in args.model:
        try:
            base = getattr(model, "model", model)
            base.set_attention_implementation("flash_attention_2")
            logger.info("Enabled Flash Attention 2 for the model.")
        except Exception:
            logger.warning("Could not set Flash Attention 2 implementation.")
        model = torch.compile(model, mode="reduce-overhead")

    log_model_architecture(model, logger)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
    total_steps = args.n_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=args.eta_min)
    return model, optimizer, scheduler

def run_training_loop(args, results_path, model, optimizer, scheduler, train_loader, test_loader, logger):
    """Executes the main training and evaluation loop."""
    metrics = Metrics()
    
    if args.precision in ['bf16', 'bfloat16']: amp_dtype = torch.bfloat16
    elif args.precision in ['f16', 'float16']: amp_dtype = torch.float16
    else: amp_dtype = torch.float32
    scaler = GradScaler(enabled=(amp_dtype == torch.float16))
    logger.info(f"Using AMP with dtype: {amp_dtype}")

    for epoch in tqdm(range(1, args.n_epochs + 1), desc="Epochs"):
        if not args.online:
            if (epoch % 100 == 1) or (epoch == args.n_epochs):
                train_loss, train_acc = evaluate_model(model, train_loader, args.device, args.model, amp_dtype)
                test_loss, test_acc = evaluate_model(model, test_loader, args.device, args.model, amp_dtype)
                metrics.update(train_loss, test_loss, train_acc, test_acc)
                logger.info(f"Epoch {epoch:04d} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
            
            train_epoch(model, optimizer, scheduler, train_loader, args.device, len(train_loader), scaler, args.model, amp_dtype)
            utils.save_data(results_path, metrics)

def cleanup(*args):
    """Clears memory by deleting objects and emptying CUDA cache."""
    for arg in args: del arg
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    """Main function to orchestrate the training process."""
    logger = setup_logger(log_file_name=os.path.join(args.results_path, "logs.log"))
    setup_environment(args.seed)
    
    # Save the run's configuration for reproducibility
    with open(os.path.join(args.results_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    logger.info(f"Loaded configuration from command-line arguments.")
    logger.info(f"Full config saved to: {os.path.join(args.results_path, 'config.json')}")
    logger.info(f"Results will be saved to: {args.results_path}")

    train_loader, test_loader = load_data(args, logger)
    model, optimizer, scheduler = load_model_and_optimizer(args, logger, train_loader)

    logger.info("Starting training loop...")
    run_training_loop(args, args.results_path, model, optimizer, scheduler, train_loader, test_loader, logger)
    logger.info("Training complete.")
    cleanup(model, optimizer, scheduler, train_loader, test_loader)
    logger.info("Cleanup complete. Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A reproducible training script for sequence classification tasks.")
    
    # --- Paths and Seed ---
    parser.add_argument('--results-path', type=str, required=True, help='Path to the directory where results will be stored.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda", "cpu").')
    
    # --- Dataset Parameters ---
    parser.add_argument('--target_func', type=str, default='func4', help='The target function to learn.')
    parser.add_argument('--sequence_length', type=int, default=50, help='Length of the input sequences for training.')
    parser.add_argument('--test_sequence_length', type=int, default=None, help='Length of the input sequences for evaluation. Defaults to 100 if not provided.')
    parser.add_argument('--train_set_size', type=int, default=100000, help='Number of samples in the training set.')
    parser.add_argument('--test_set_size', type=int, default=10000, help='Number of samples in the test set.')
    parser.add_argument('--BOS_TOKEN', type=int, default=2, help='Beginning of sequence token.')
    parser.add_argument('--online', action='store_true', help='Flag for online data generation (if supported).')

    # --- Model Parameters ---
    parser.add_argument('--model', type=str, default='qwen1.7B', help='Model architecture to use.')
    parser.add_argument('--vocab_size', type=int, default=3, help='Vocabulary size.')
    parser.add_argument('--context_length', type=int, default=None, help='Model context length.')
    parser.add_argument('--n_layer', type=int, default=4, help='Number of transformer layers.')
    parser.add_argument('--n_head', type=int, default=10, help='Number of attention heads.')
    parser.add_argument('--n_embd', type=int, default=500, help='Embedding dimension.')
    parser.add_argument('--num_layers_to_finetune', type=int, default=2, help='Number of top layers to unfreeze for fine-tuning.')
    parser.add_argument('--num_models', type=int, default=1, help='Number of models for ensemble (e.g., for Bloom).')
    
    # --- Training Parameters ---
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training and evaluation.')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for the optimizer.')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='Minimum learning rate for cosine annealing.')
    parser.add_argument('--precision', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'], help='Training precision.')

    args = parser.parse_args()
    
    # Set context_length automatically if not provided
    if args.context_length is None or args.context_length < args.sequence_length + 2:
        args.context_length = args.sequence_length + 2

    main(args)