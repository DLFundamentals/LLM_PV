# data_handler.py
"""
Data generation module for various machine learning tasks.

This module provides a collection of data generator classes built on a common
abstract base class, `BaseDataGenerator`. Each generator is responsible for
creating a balanced (50/50 split) dataset for a specific task, such as
primality testing, palindrome detection, or formal language recognition.

Available Generators:
- PrimeDecimalTailRestrictedDataGenerator: Prime vs. composite ending in {1,3,7,9}.
- PrimeDataGenerator: Slower, sympy-based prime number data generation.
- BinaryDataGenerator: Generic generator for various binary target functions.
- Dyck2DataGenerator: Generates sequences for the Dyck-2 language (e.g., '()[]').
- PatternBasedDataGenerator: Detects a specific binary pattern in sequences.
- PalindromeDataGenerator: Detects if a binary sequence is a palindrome.
"""

import torch
import math
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Set, Tuple, Union, Optional
from abc import ABC, abstractmethod

from .target_functions import TARGET_FUNCTIONS

# --- Module-level Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional dependency handling for performance
try:
    from sympy import nextprime, isprime
except ImportError:
    logger.error("Error: sympy library not found. Please run 'pip install sympy'")
    exit()

try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    logger.error("Error: ucimlrepo library not found. Please run 'pip install ucimlrepo'")
    exit()


# =============================================================================
# Abstract Base Class for All Data Generators
# =============================================================================

class BaseDataGenerator(ABC):
    """
    Abstract base class for data generators.

    This class provides a common structure for all data generators, handling
    initialization, and validation.
    Subclasses are required to implement the core logic for generating raw data
    and formatting individual samples.
    """

    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        if not isinstance(sequence_length, int) or sequence_length <= 0:
            msg = f"Sequence length must be a positive integer, but got {sequence_length}."
            logger.error(msg)
            raise ValueError(msg)

        if not isinstance(num_samples, int) or num_samples <= 0:
            msg = f"Number of samples must be a positive integer, but got {num_samples}."
            logger.error(msg)
            raise ValueError(msg)

        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.device = device

        if num_samples % 2 != 0:
            logger.warning(
                f"{self.__class__.__name__}: num_samples is odd ({num_samples}). "
                "For a 50/50 split, an even number is recommended."
            )
        self.num_positive_samples = self.num_samples // 2
        self.num_negative_samples = self.num_samples - self.num_positive_samples

    @abstractmethod
    def _generate_raw_data(self) -> Tuple[List[Any], List[Any]]:
        """
        Generates the raw data for positive and negative samples.
        Must be implemented by subclasses.

        Returns:
            A tuple of two lists: (positive_samples, negative_samples). The elements
            of the lists can be any type (e.g., int, torch.Tensor) that the
            _format_input method can handle.
        """
        pass

    @abstractmethod
    def _format_input(self, sample: Any) -> np.ndarray:
        """
        Formats a single raw sample into the required numpy array of strings.
        Must be implemented by subclasses.
        """
        pass

    def generate_data(self) -> List[Dict[str, Any]]:
        """
        Orchestrates the data generation, formatting, and shuffling process.
        This is the main public method to be called by users.
        """
        class_name = self.__class__.__name__
        logger.info(f"Starting data generation for {class_name}...")

        positive_samples, negative_samples = self._generate_raw_data()

        # Defensive checks to ensure subclass implementation is correct
        if len(positive_samples) != self.num_positive_samples:
            raise RuntimeError(f"{class_name} generated {len(positive_samples)} positive samples, expected {self.num_positive_samples}.")
        if len(negative_samples) != self.num_negative_samples:
            raise RuntimeError(f"{class_name} generated {len(negative_samples)} negative samples, expected {self.num_negative_samples}.")

        logger.info("Formatting and combining dataset...")
        dataset = []
        for p_sample in tqdm(positive_samples, desc="Formatting positive samples", leave=False, unit="sample"):
            dataset.append({'Input': self._format_input(p_sample), 'Output': '1'})

        for n_sample in tqdm(negative_samples, desc="Formatting negative samples", leave=False, unit="sample"):
            dataset.append({'Input': self._format_input(n_sample), 'Output': '0'})

        logger.info(f"Data generation complete for {class_name}. Total samples: {len(dataset)}.")
        return dataset


# =============================================================================
# Primality-Based Data Generators
# =============================================================================


class BaseDecimalGenerator(BaseDataGenerator):
    """Base class for decimal-based generators that support leading zeros."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_leading_zeros: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if self.num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a guaranteed 50/50 split.")

        self.allow_leading_zeros = allow_leading_zeros
        if allow_leading_zeros:
            self.start_range = 0
            self.end_range = 10 ** self.sequence_length
        else:
            self.start_range = 10 ** (self.sequence_length - 1) if self.sequence_length > 1 else 1
            self.end_range = 10 ** self.sequence_length

        max_possible = self.end_range - self.start_range
        if self.num_samples > max_possible:
            raise ValueError(f"Requested {self.num_samples} samples, but only {max_possible} unique sequences exist.")

    def _format_input(self, sample: int) -> np.ndarray:
        return np.array(list(str(sample).zfill(self.sequence_length)))
    

class PrimeDecimalTailRestrictedDataGenerator(BaseDecimalGenerator):
    """Generates primes vs. non-primes ending in a decimal from {1, 3, 7, 9}."""

    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_leading_zeros: bool = False,
                 allowed_nonprime_last_digits: Tuple[int, ...] = (1, 3, 7, 9)):
        super().__init__(sequence_length, num_samples, device, allow_leading_zeros)
        if not all(d in range(10) for d in allowed_nonprime_last_digits):
            raise ValueError("allowed_nonprime_last_digits must be decimal digits 0-9.")
        self.allowed_nonprime_last_digits = tuple(sorted(set(allowed_nonprime_last_digits)))
        logger.info(f"PrimeDecimalTailRestrictedDataGenerator initialized with allowed_nonprime_last_digits={self.allowed_nonprime_last_digits}")

    def _generate_raw_data(self) -> Tuple[List[int], List[int]]:
        primes_found: Set[int] = set()
        non_primes_found: Set[int] = set()

        # --- Generate Primes ---
        while len(primes_found) < self.num_positive_samples:
            rnd_start = random.randint(self.start_range, max(self.start_range, self.end_range - 2))
            candidate = nextprime(rnd_start)
            if self.start_range <= candidate < self.end_range and candidate not in primes_found:
                primes_found.add(candidate)

        # --- Generate Non-Primes (CONSTRUCTIVE method + last digit filter) ---
        a_len = self.sequence_length // 2
        b_len = self.sequence_length - a_len
        a_start, a_end = 10**(a_len - 1), 10**a_len
        b_start, b_end = 10**(b_len - 1), 10**b_len

        all_found = primes_found.copy()

        while len(non_primes_found) < self.num_negative_samples:
            f1 = random.randrange(a_start, a_end) if a_start < a_end else a_start
            f2 = random.randrange(b_start, b_end) if b_start < b_end else b_start
            candidate = f1 * f2
            
            # Keep only exact-length composites with the correct last digit
            if (self.start_range <= candidate < self.end_range and 
                candidate not in all_found and 
                (candidate % 10) in self.allowed_nonprime_last_digits):
                non_primes_found.add(candidate)
                all_found.add(candidate)
        
        logger.info(f"Generated {len(primes_found)} primes and {len(non_primes_found)} restricted non-primes.")
        return list(primes_found), list(non_primes_found)
    

class PrimeDataGenerator(BaseDataGenerator):
    """Generates decimal input vectors and corresponding prime/non-prime outputs using sympy."""
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        self.start_range = 10**(self.sequence_length - 1) if self.sequence_length > 1 else 1
        self.end_range = 10**self.sequence_length
        if self.num_samples > (self.end_range - self.start_range):
            raise ValueError(f"Requested {self.num_samples} samples, but only {self.end_range - self.start_range} unique numbers exist.")
        logger.info(f"PrimeDataGenerator (sympy) initialized for len={sequence_length}, samples={num_samples}")

    def _format_input(self, sample: int) -> np.ndarray:
        return np.array(list(str(sample)))

    def _generate_raw_data(self) -> Tuple[List[int], List[int]]:
        primes_found: Set[int] = set()
        non_primes_found: Set[int] = set()

        # --- Generate Primes ---
        while len(primes_found) < self.num_positive_samples:
            random_start = random.randint(self.start_range, self.end_range - 2) if self.end_range - 2 > self.start_range else self.start_range
            candidate = nextprime(random_start)
            if candidate < self.end_range and candidate not in primes_found:
                primes_found.add(candidate)
        
        # --- Generate Non-Primes (CONSTRUCTIVE method from SGD script) ---
        a_len = self.sequence_length // 2
        b_len = self.sequence_length - a_len
        a_start, a_end = 10**(a_len - 1), 10**a_len
        b_start, b_end = 10**(b_len - 1), 10**b_len

        all_found = primes_found.copy() # Avoid generating a number that is already in the prime set

        while len(non_primes_found) < self.num_negative_samples:
            f1 = random.randrange(a_start, a_end) if a_start < a_end else a_start
            f2 = random.randrange(b_start, b_end) if b_start < b_end else b_start
            candidate = f1 * f2

            # Keep only exact-length composites and ensure uniqueness
            if self.start_range <= candidate < self.end_range and candidate not in all_found:
                non_primes_found.add(candidate)
                all_found.add(candidate)

        logger.info(f"Successfully generated {len(primes_found)} primes and {len(non_primes_found)} non-primes.")
        return list(primes_found), list(non_primes_found)


class PrimePlus47DataGenerator(BaseDataGenerator):
    """Generates numbers n where n+47 is prime or non-prime."""
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu',
                 allowed_nonprime_last_digits: Tuple[int, ...] = (1, 3, 7, 9)):
        super().__init__(sequence_length, num_samples, device)
        if self.num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        
        if not all(d in range(10) for d in allowed_nonprime_last_digits):
            raise ValueError("allowed_nonprime_last_digits must be decimal digits 0-9.")
        self.allowed_nonprime_last_digits = tuple(sorted(set(allowed_nonprime_last_digits)))
        
        # Ensure n+47 >= 2, so n >= -45, but we want positive n
        # For n where n+47 is prime, we need n+47 >= 2, so n >= -45
        # But we want n to be in the sequence_length range
        self.start_range = max(1, 10**(self.sequence_length - 1)) if self.sequence_length > 1 else 1
        self.end_range = 10**self.sequence_length
        
        # Adjust start_range to ensure n+47 can be prime (n+47 >= 2)
        if self.start_range + 47 < 2:
            self.start_range = max(1, 2 - 47)
        
        if self.start_range >= self.end_range:
            raise ValueError(f"Invalid range: start={self.start_range}, end={self.end_range}")
        
        logger.info(f"PrimePlus47DataGenerator initialized: range=[{self.start_range}, {self.end_range}), allowed_nonprime_last_digits={self.allowed_nonprime_last_digits}")
    
    def _format_input(self, sample: int) -> np.ndarray:
        return np.array(list(str(sample).zfill(self.sequence_length)))
    
    def _generate_raw_data(self) -> Tuple[List[int], List[int]]:
        primes_plus_47: Set[int] = set()
        non_primes_plus_47: Set[int] = set()
        
        # Generate n where n-47 is prime (positive class)
        while len(primes_plus_47) < self.num_positive_samples:
            p_min = max(2, self.start_range - 47) if self.start_range > 47 else 2
            p_max = self.end_range - 47
            if p_min >= p_max:
                raise ValueError(f"Cannot generate enough primes: p_range=[{p_min}, {p_max})")
            random_start = random.randint(p_min, p_max - 2) if p_max - 2 > p_min else p_min
            p = nextprime(random_start)
            n = p + 47
            if self.start_range <= n < self.end_range and n not in primes_plus_47:
                primes_plus_47.add(n)
        
        # Generate n where n-47 is non-prime (negative class)
        a_len = self.sequence_length // 2
        b_len = self.sequence_length - a_len
        a_start, a_end = (10**(a_len - 1), 10**a_len) if a_len > 0 else (2, 3)
        b_start, b_end = (10**(b_len - 1), 10**b_len) if b_len > 0 else (2, 3)
        
        all_found = primes_plus_47.copy()
        
        while len(non_primes_plus_47) < self.num_negative_samples:
            f1 = random.randrange(a_start, a_end) if a_start < a_end else 2
            f2 = random.randrange(b_start, b_end) if b_start < b_end else 2
            composite = f1 * f2
            n = composite + 47
            
            if (self.start_range <= n < self.end_range and 
                n not in all_found and
                not isprime(composite) and
                (composite % 10) in self.allowed_nonprime_last_digits):
                non_primes_plus_47.add(n)
                all_found.add(n)
        
        logger.info(f"Generated {len(primes_plus_47)} numbers where n-47 is prime and {len(non_primes_plus_47)} where n-47 is non-prime.")
        return list(primes_plus_47), list(non_primes_plus_47)


# =============================================================================
# Tensor-Based Data Generators
# =============================================================================

class BaseTensorGenerator(BaseDataGenerator):
    """Base class for generators that internally use torch.Tensors."""
    def _format_input(self, sample: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        return np.array([str(item.item()) for item in sample])


class BinaryDataGenerator(BaseTensorGenerator):
    """Generates binary vectors from a target function with an exact 50/50 split."""
    _MAX_ATTEMPTS_NO_PROGRESS = 50

    def __init__(self, target_function_name: str, sequence_length: int, num_samples: int,
                 device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        if target_function_name not in TARGET_FUNCTIONS:
            raise ValueError(f"Unknown target function: {target_function_name}.")
        self.target_function = TARGET_FUNCTIONS[target_function_name]
        self.allow_duplicates = allow_duplicates
        if not allow_duplicates and num_samples > 2**sequence_length:
            raise ValueError(f"Cannot generate {num_samples} unique samples; only {2**sequence_length} are possible.")
        logger.info(f"BinaryDataGenerator initialized for target='{target_function_name}', allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        samples_1, samples_0 = self._generate_balanced_samples()
        return list(samples_1), list(samples_0)

    def _generate_balanced_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.allow_duplicates:
            return self._generate_with_duplicates()
        return self._generate_unique()

    def _generate_with_duplicates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_0, samples_1 = [], []
        attempts = 0
        while (len(samples_0) < self.num_positive_samples) or (len(samples_1) < self.num_negative_samples):
            need0 = self.num_positive_samples - len(samples_0)
            need1 = self.num_negative_samples - len(samples_1)
            batch_size = int(max(need0, need1, 1000) * 1.5)
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            outputs = self.target_function(candidates, self.device)

            idx0 = (outputs == 0).nonzero(as_tuple=True)[0]
            idx1 = (outputs == 1).nonzero(as_tuple=True)[0]
            progress = idx0.numel() > 0 or idx1.numel() > 0

            if need0 > 0 and idx0.numel() > 0: samples_0.extend(candidates[idx0])
            if need1 > 0 and idx1.numel() > 0: samples_1.extend(candidates[idx1])
            
            attempts = 0 if progress else attempts + 1
            if attempts > self._MAX_ATTEMPTS_NO_PROGRESS:
                raise RuntimeError("Failed to gather samples. Target function may be too skewed.")

        return torch.stack(samples_1[:self.num_positive_samples]), torch.stack(samples_0[:self.num_negative_samples])

    def _generate_unique(self) -> Tuple[torch.Tensor, torch.Tensor]:
        found_set: Set[Tuple[int, ...]] = set()
        samples_0, samples_1 = [], []
        attempts = 0
        while (len(samples_0) < self.num_positive_samples) or (len(samples_1) < self.num_negative_samples):
            needed = (self.num_positive_samples - len(samples_0)) + (self.num_negative_samples - len(samples_1))
            batch_size = int(max(needed, 1000) * 1.5)
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            outputs = self.target_function(candidates, self.device)

            progress = False
            for cand, out in zip(candidates, outputs):
                cand_tuple = tuple(cand.tolist())
                if cand_tuple in found_set: continue
                if out == 0 and len(samples_0) < self.num_negative_samples:
                    samples_0.append(cand)
                    found_set.add(cand_tuple)
                    progress = True
                elif out == 1 and len(samples_1) < self.num_positive_samples:
                    samples_1.append(cand)
                    found_set.add(cand_tuple)
                    progress = True
            
            attempts = 0 if progress else attempts + 1
            if attempts > self._MAX_ATTEMPTS_NO_PROGRESS:
                raise RuntimeError("Failed to gather unique samples. Target function may be too skewed.")

        return torch.stack(samples_1), torch.stack(samples_0)


class Dyck2DataGenerator(BaseTensorGenerator):
    """Generates sequences for the Dyck-2 language (e.g., '()[]') with a 50/50 split."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if sequence_length <= 0 or sequence_length % 4 != 0:
            raise ValueError("sequence_length must be a positive multiple of 4.")
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        self.paren_seq_length = sequence_length // 2
        self.paren_map = {"00": "(", "01": ")", "10": "[", "11": "]"}
        self.paren_to_bit_str = {v: k for k, v in self.paren_map.items()}
        self.open_to_close = {'(': ')', '[': ']'}
        self.close_to_open = {v: k for k, v in self.open_to_close.items()}
        self.allow_duplicates = allow_duplicates
        logger.info(f"Dyck2DataGenerator initialized, allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            valid_samples = self._generate_valid_samples_with_duplicates(self.num_positive_samples)
            invalid_samples = self._generate_invalid_samples_with_duplicates(self.num_negative_samples)
        else:
            valid_samples_set = self._generate_valid_samples_unique(self.num_positive_samples)
            invalid_samples_set = self._generate_invalid_samples_unique(self.num_negative_samples, valid_samples_set)
            valid_samples = torch.tensor(list(valid_samples_set), dtype=torch.long, device=self.device)
            invalid_samples = torch.tensor(list(invalid_samples_set), dtype=torch.long, device=self.device)

        return list(valid_samples), list(invalid_samples)

    def _is_valid_paren_seq(self, paren_seq: str) -> bool:
        stack = []
        for char in paren_seq:
            if char in self.open_to_close:
                stack.append(char)
            elif char in self.close_to_open:
                if not stack or stack.pop() != self.close_to_open[char]:
                    return False
        return not stack

    def _generate_one_valid_paren_seq(self) -> str:
        stack, seq = [], []
        while (len(stack) + len(seq)) < self.paren_seq_length:
            o, c = random.choice(list(self.open_to_close.items()))
            if not stack or random.random() < 0.5:
                seq.append(o)
                stack.append(c)
            else:
                seq.append(stack.pop())
        while stack:
            seq.append(stack.pop())
        return "".join(seq)

    def _generate_valid_samples_with_duplicates(self, count: int) -> torch.Tensor:
        samples = []
        for _ in range(count):
            paren_seq = self._generate_one_valid_paren_seq()
            bit_string = "".join(self.paren_to_bit_str[c] for c in paren_seq)
            samples.append(torch.tensor([int(b) for b in bit_string], dtype=torch.long))
        return torch.stack(samples, dim=0).to(self.device)

    def _generate_invalid_samples_with_duplicates(self, count: int) -> torch.Tensor:
        samples = []
        while len(samples) < count:
            valid_paren_seq = self._generate_one_valid_paren_seq()
            bits = list(''.join(self.paren_to_bit_str[c] for c in valid_paren_seq))
            # Corrupt 1 to 5 bits to create an invalid sequence
            num_flips = min(random.randint(1, 5), self.sequence_length)
            for pos in random.sample(range(self.sequence_length), k=num_flips):
                bits[pos] = '1' if bits[pos] == '0' else '0'
            corrupted_bit_str = "".join(bits)
            bit_pairs = [corrupted_bit_str[i:i+2] for i in range(0, self.sequence_length, 2)]

            if all(pair in self.paren_map for pair in bit_pairs):
                paren_seq = "".join([self.paren_map[p] for p in bit_pairs])
                if not self._is_valid_paren_seq(paren_seq):
                    samples.append(torch.tensor([int(b) for b in corrupted_bit_str], dtype=torch.long))
            else: # Guaranteed invalid if bit pairs are not well-formed
                samples.append(torch.tensor([int(b) for b in corrupted_bit_str], dtype=torch.long))
        return torch.stack(samples, dim=0).to(self.device)

    def _generate_valid_samples_unique(self, count: int) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            paren_seq = self._generate_one_valid_paren_seq()
            bit_string = "".join(self.paren_to_bit_str[c] for c in paren_seq)
            bit_tuple = tuple(int(b) for b in bit_string)
            unique_samples.add(bit_tuple)
        return unique_samples

    def _generate_invalid_samples_unique(self, count: int, existing_samples: Set[Tuple[int, ...]]) -> Set[Tuple[int, ...]]:
        unique_invalid: Set[Tuple[int, ...]] = set()
        while len(unique_invalid) < count:
            valid_paren_seq = self._generate_one_valid_paren_seq()
            bits = list(''.join(self.paren_to_bit_str[c] for c in valid_paren_seq))
            num_flips = min(random.randint(1, 5), self.sequence_length)
            for pos in random.sample(range(self.sequence_length), k=num_flips):
                bits[pos] = '1' if bits[pos] == '0' else '0'
            corrupted_bit_str = "".join(bits)
            
            is_valid_after_corruption = False
            bit_pairs = [corrupted_bit_str[i:i+2] for i in range(0, self.sequence_length, 2)]
            if all(pair in self.paren_map for pair in bit_pairs):
                paren_seq = "".join([self.paren_map[p] for p in bit_pairs])
                if self._is_valid_paren_seq(paren_seq):
                    is_valid_after_corruption = True

            if not is_valid_after_corruption:
                candidate_tuple = tuple(int(b) for b in corrupted_bit_str)
                if candidate_tuple not in existing_samples and candidate_tuple not in unique_invalid:
                    unique_invalid.add(candidate_tuple)
        return unique_invalid


class PatternBasedDataGenerator(BaseTensorGenerator):
    """Generates a 50/50 dataset of binary sequences with/without a given pattern."""
    def __init__(self, sequence_length: int, num_samples: int, pattern_string: str = '10101010',
                 device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        if not pattern_string or len(pattern_string) > sequence_length:
            raise ValueError("Invalid pattern_string length.")
        try:
            self.pattern_tensor = torch.tensor([int(b) for b in pattern_string], dtype=torch.long, device=device)
        except ValueError:
            raise ValueError("pattern_string must contain only '0's and '1's.")
        self.allow_duplicates = allow_duplicates
        logger.info(f"PatternBasedDataGenerator initialized for pattern='{pattern_string}', allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            with_p = self._generate_with_pattern_duplicates(self.num_positive_samples)
            without_p = self._generate_without_pattern_duplicates(self.num_negative_samples)
        else:
            with_p_set = self._generate_with_pattern_unique(self.num_positive_samples)
            without_p_set = self._generate_without_pattern_unique(self.num_negative_samples, with_p_set)
            with_p = torch.tensor(list(with_p_set), dtype=torch.long, device=self.device)
            without_p = torch.tensor(list(without_p_set), dtype=torch.long, device=self.device)
        return list(with_p), list(without_p)

    def _generate_with_pattern_duplicates(self, count: int) -> torch.Tensor:
        seqs = torch.randint(0, 2, (count, self.sequence_length), dtype=torch.long, device=self.device)
        insert_idx = torch.randint(0, self.sequence_length - len(self.pattern_tensor) + 1, (count,), device=self.device)
        for i in range(count):
            seqs[i, insert_idx[i] : insert_idx[i] + len(self.pattern_tensor)] = self.pattern_tensor
        return seqs

    def _generate_without_pattern_duplicates(self, count: int) -> torch.Tensor:
        seqs = torch.empty((count, self.sequence_length), dtype=torch.long, device=self.device)
        for i in range(count):
            while True:
                seq = torch.randint(0, 2, (self.sequence_length,), dtype=torch.long, device=self.device)
                if not self._contains_pattern(seq):
                    seqs[i] = seq
                    break
        return seqs

    def _generate_with_pattern_unique(self, count: int) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            batch_size = int((count - len(unique_samples)) * 1.5) + 10
            seqs = self._generate_with_pattern_duplicates(batch_size)
            unique_samples.update(tuple(s.tolist()) for s in seqs)
        return set(random.sample(list(unique_samples), count))

    def _generate_without_pattern_unique(self, count: int, existing_samples: Set[Tuple[int, ...]]) -> Set[Tuple[int, ...]]:
        unique_samples: Set[Tuple[int, ...]] = set()
        while len(unique_samples) < count:
            batch_size = int((count - len(unique_samples)) * 1.5) + 10
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), dtype=torch.long, device=self.device)
            for seq in candidates:
                if not self._contains_pattern(seq):
                    seq_tuple = tuple(seq.tolist())
                    if seq_tuple not in existing_samples:
                        unique_samples.add(seq_tuple)
        return set(random.sample(list(unique_samples), count))

    def _contains_pattern(self, sequence: torch.Tensor) -> bool:
        n, m = len(sequence), len(self.pattern_tensor)
        for i in range(n - m + 1):
            if torch.equal(sequence[i:i+m], self.pattern_tensor):
                return True
        return False


class PalindromeDataGenerator(BaseTensorGenerator):
    """Generates binary sequences for a palindrome task with a 50/50 split."""
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu', allow_duplicates: bool = False):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0: raise ValueError("num_samples must be even.")
        self.allow_duplicates = allow_duplicates
        logger.info(f"PalindromeDataGenerator initialized, allow_duplicates={allow_duplicates}")

    def _generate_raw_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.allow_duplicates:
            palindromes = self._generate_palindromes_duplicates(self.num_positive_samples)
            non_palindromes = self._generate_non_palindromes_duplicates(palindromes)
        else:
            palindromes = self._generate_palindromes_unique(self.num_positive_samples)
            non_palindromes = self._generate_non_palindromes_unique(self.num_negative_samples, palindromes)
        return list(palindromes), list(non_palindromes)

    def _generate_palindromes(self, first_halves: torch.Tensor) -> torch.Tensor:
        L, half_len = self.sequence_length, (self.sequence_length + 1) // 2
        second_halves_rev = torch.flip(first_halves[:, :L // 2], dims=[1])
        return torch.cat([first_halves, second_halves_rev], dim=1)

    def _generate_palindromes_duplicates(self, count: int) -> torch.Tensor:
        half_len = (self.sequence_length + 1) // 2
        first_halves = torch.randint(0, 2, size=(count, half_len), device=self.device, dtype=torch.long)
        return self._generate_palindromes(first_halves)

    def _generate_non_palindromes_duplicates(self, base_palindromes: torch.Tensor) -> torch.Tensor:
        non_palindromes = base_palindromes.clone()
        half_len = (self.sequence_length + 1) // 2
        rows = torch.arange(non_palindromes.size(0), device=self.device)
        cols_to_flip = torch.randint(0, half_len, size=(non_palindromes.size(0),), device=self.device)
        non_palindromes[rows, cols_to_flip] = 1 - non_palindromes[rows, cols_to_flip]
        return non_palindromes

    def _generate_palindromes_unique(self, count: int) -> torch.Tensor:
        half_len = (self.sequence_length + 1) // 2
        if count > 2**half_len:
            raise ValueError(f"Cannot generate {count} unique palindromes; only {2**half_len} exist.")
        
        unique_halves: Set[Tuple[int, ...]] = set()
        while len(unique_halves) < count:
            batch_size = int((count - len(unique_halves)) * 1.5) + 10
            fh = torch.randint(0, 2, size=(batch_size, half_len), device=self.device, dtype=torch.long)
            unique_halves.update(tuple(row.tolist()) for row in fh)
        
        first_halves = torch.tensor(random.sample(list(unique_halves), count), dtype=torch.long, device=self.device)
        return self._generate_palindromes(first_halves)

    def _generate_non_palindromes_unique(self, count: int, existing_palindromes: torch.Tensor) -> torch.Tensor:
        nonpal_set: Set[Tuple[int, ...]] = set()
        pal_set = {tuple(p.tolist()) for p in existing_palindromes}
        while len(nonpal_set) < count:
            batch_size = int((count - len(nonpal_set)) * 1.5) + 10
            candidates = torch.randint(0, 2, (batch_size, self.sequence_length), device=self.device, dtype=torch.long)
            for seq in candidates:
                if not torch.equal(seq, torch.flip(seq, dims=[0])):
                    seq_tuple = tuple(seq.tolist())
                    if seq_tuple not in pal_set and seq_tuple not in nonpal_set:
                        nonpal_set.add(seq_tuple)
        return torch.tensor(random.sample(list(nonpal_set), count), dtype=torch.long, device=self.device)
    

class GraphCycleDataGenerator(BaseDataGenerator):
    def __init__(self, sequence_length: int, num_samples: int, max_vertices: int = 25, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        if sequence_length % 4 != 0:
            raise ValueError("sequence_length must be a multiple of 4 (4 digits per edge).")
        self.max_vertices = min(max_vertices, sequence_length // 4)
        logger.info(f"GraphCycleDataGenerator initialized: max_vertices={self.max_vertices}, sequence_length={sequence_length}")
    
    def _count_sccs(self, edges: List[Tuple[int, int]]) -> int:
        """Count strongly connected components using Kosaraju's algorithm."""
        if not edges:
            return 0
        
        adj_out = {}
        adj_in = {}
        vertex_set = set()
        for u, v in edges:
            vertex_set.add(u)
            vertex_set.add(v)
            if u not in adj_out:
                adj_out[u] = []
            if v not in adj_in:
                adj_in[v] = []
            adj_out[u].append(v)
            adj_in[v].append(u)
        
        visited = set()
        finish_order = []
        
        def dfs1(node):
            visited.add(node)
            for neighbor in adj_out.get(node, []):
                if neighbor not in visited:
                    dfs1(neighbor)
            finish_order.append(node)
        
        for vertex in vertex_set:
            if vertex not in visited:
                dfs1(vertex)
        
        visited.clear()
        scc_count = 0
        
        def dfs2(node):
            visited.add(node)
            for neighbor in adj_in.get(node, []):
                if neighbor not in visited:
                    dfs2(neighbor)
        
        for vertex in reversed(finish_order):
            if vertex not in visited:
                dfs2(vertex)
                scc_count += 1
        
        return scc_count
    
    def _count_cycles(self, edges: List[Tuple[int, int]]) -> int:
        """Count simple disjoint cycles in a directed graph, excluding self-loops."""
        if not edges:
            return 0
        
        adj = {}
        in_deg = {}
        nodes = set()
        for u, v in edges:
            if u == v:
                continue
            adj.setdefault(u, []).append(v)
            in_deg[v] = in_deg.get(v, 0) + 1
            nodes.add(u); nodes.add(v)
        
        visited = set()
        cycle_count = 0
        
        for start_node in nodes:
            if start_node in visited:
                continue
            
            if len(adj.get(start_node, [])) != 1 or in_deg.get(start_node, 0) != 1:
                continue
                
            curr = start_node
            temp_visited = {curr}
            path = [curr]
            is_cycle = False
            
            while True:
                next_nodes = adj.get(curr, [])
                if len(next_nodes) != 1:
                    break
                next_node = next_nodes[0]
                
                if in_deg.get(next_node, 0) != 1:
                    break
                    
                if next_node == start_node:
                    is_cycle = True
                    break
                
                if next_node in temp_visited or next_node in visited:
                    break
                    
                curr = next_node
                temp_visited.add(curr)
                path.append(curr)
                
            if is_cycle:
                cycle_count += 1
                visited.update(temp_visited)
                
        return cycle_count
    
    def _permute_node_labels(self, edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Randomly permute node labels to prevent node ID leakage.
        IMPORTANT: Maps to IDs 1-99 (never 0) so '0000' padding is unambiguous."""
        if not edges:
            return edges
        
        vertex_set = set()
        for u, v in edges:
            vertex_set.add(u)
            vertex_set.add(v)
        
        n = len(vertex_set)
        new_ids = list(range(1, n + 1))
        random.shuffle(new_ids)
        
        old_to_new = {old: new_ids[i] for i, old in enumerate(sorted(vertex_set))}
        return [(old_to_new[u], old_to_new[v]) for u, v in edges]
    
    def _is_cycle_graph(self, edges: List[Tuple[int, int]]) -> bool:
        """Check if a directed graph IS a cycle.
        A directed graph is a cycle if:
        - All vertices have in-degree = 1 and out-degree = 1
        - The graph is connected (all vertices reachable)
        - Number of edges equals number of vertices"""
        if not edges:
            return False
        
        adj_out = {}
        adj_in = {}
        vertex_set = set()
        
        for u, v in edges:
            vertex_set.add(u)
            vertex_set.add(v)
            if u not in adj_out:
                adj_out[u] = []
            if v not in adj_in:
                adj_in[v] = []
            adj_out[u].append(v)
            adj_in[v].append(u)
        
        num_vertices = len(vertex_set)
        num_edges = len(edges)
        
        # Need at least 3 vertices for a cycle
        if num_vertices < 3:
            return False
        
        # Number of edges must equal number of vertices
        if num_edges != num_vertices:
            return False
        
        # All vertices must have in-degree = 1 and out-degree = 1
        for vertex in vertex_set:
            out_deg = len(adj_out.get(vertex, []))
            in_deg = len(adj_in.get(vertex, []))
            if out_deg != 1 or in_deg != 1:
                return False
        
        # Check connectivity: all vertices must be reachable from any vertex
        visited = set()
        
        def dfs(node):
            visited.add(node)
            if node in adj_out and adj_out[node]:
                next_node = adj_out[node][0]
                if next_node not in visited:
                    dfs(next_node)
        
        # Start from any vertex
        start_vertex = list(vertex_set)[0]
        dfs(start_vertex)
        
        # If we visited all vertices, the graph is connected
        return len(visited) == num_vertices
    
    def _has_cycle(self, edges: List[Tuple[int, int]]) -> bool:
        """Check if a directed graph has a cycle using DFS, excluding self-loops."""
        if not edges:
            return False
        
        # Filter out self-loops
        edges = [(u, v) for u, v in edges if u != v]
        
        if not edges:
            return False
        
        adj = {}
        vertex_set = set()
        for u, v in edges:
            vertex_set.add(u)
            vertex_set.add(v)
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Back edge found - cycle exists
                    return True
            
            rec_stack.remove(node)
            return False
        
        for vertex in vertex_set:
            if vertex not in visited:
                if dfs(vertex):
                    return True
        
        return False
    
    def _generate_raw_data(self) -> Tuple[List[str], List[str]]:
        """Generate graphs with > threshold cycles (positive) and <= threshold cycles (negative).
        Both classes use the same distributions of vertices (N) and edges (M) to prevent cheating.
        Number of edges is dynamic based on sequence_length (L/5 to L/4)."""
        positive_samples = []
        negative_samples = []
        m_min = self.sequence_length // 5
        m_max = self.sequence_length // 4
        
        # Adaptive threshold based on sequence length
        # Each cycle needs 3 edges. Max cycles = (L // 4) // 3
        max_possible_cycles = (self.sequence_length // 4) // 3
        # We want a threshold that divides the range of possible cycles
        # If L=300, max_cycles = 75 // 3 = 25. Threshold 8 is fine.
        # If L=100, max_cycles = 25 // 3 = 8. Threshold 8 is impossible (>8).
        # We'll use threshold = max(1, min(8, max_possible_cycles // 2))
        threshold = max(1, min(8, max_possible_cycles // 2))
        
        # Ensure m_min is enough for threshold + 1 cycles
        m_min_required = (threshold + 1) * 3
        m_min = max(m_min, m_min_required)
        m_max = max(m_max, m_min + 1)
        
        if m_max > self.sequence_length // 4:
            m_max = self.sequence_length // 4
            m_min = min(m_min, m_max)
        
        while len(positive_samples) < self.num_positive_samples or len(negative_samples) < self.num_negative_samples:
            # 1. Pick edge count m within the dynamic range
            m = random.randint(m_min, m_max)
            # Pick vertex count n (reasonable range for m edges, e.g. m to m+30)
            n = random.randint(m, min(m + 30, 99))
            
            def generate_with_count(target_cycles, target_edges, target_nodes):
                edges = []
                vertices = list(range(target_nodes))
                random.shuffle(vertices)
                curr_v = 0
                rem_m = target_edges
                
                # Generate cycles (minimum 3 nodes each)
                for c_idx in range(target_cycles):
                    if rem_m < 3 * (target_cycles - c_idx): break
                    max_size = min(rem_m - 3 * (target_cycles - c_idx - 1), target_nodes - curr_v)
                    if max_size < 3: break
                    size = random.randint(3, max_size)
                    cycle_v = vertices[curr_v:curr_v+size]
                    for i in range(size):
                        u, v = cycle_v[i], cycle_v[(i+1)%size]
                        if u != v:
                            edges.append((u, v))
                    curr_v += size
                    rem_m -= size
                
                if len(edges) < target_cycles * 3: return None
                
                # Fill remaining edges with noise (forest) to avoid new cycles and self-loops
                if rem_m > 0:
                    visited_nodes = set()
                    for u, v in edges:
                        visited_nodes.add(u); visited_nodes.add(v)
                    
                    unvisited_nodes = [v for v in vertices if v not in visited_nodes]
                    random.shuffle(unvisited_nodes)
                    
                    # 1. Connect unvisited nodes to existing components (tree growth)
                    while rem_m > 0 and unvisited_nodes and visited_nodes:
                        u = unvisited_nodes.pop()
                        v = random.choice(list(visited_nodes))
                        if u != v:
                            if random.random() < 0.5:
                                edges.append((u, v))
                            else:
                                edges.append((v, u))
                            visited_nodes.add(u)
                            rem_m -= 1
                    
                    # 2. Connect unvisited nodes to each other (new tree roots)
                    while rem_m > 0 and len(unvisited_nodes) >= 2:
                        u = unvisited_nodes.pop()
                        v = unvisited_nodes.pop()
                        if u != v:
                            edges.append((u, v))
                            visited_nodes.add(u)
                            visited_nodes.add(v)
                            rem_m -= 1
                        
                    # 3. Last resort: if still have edges, we must add them between nodes
                    # in DIFFERENT components or simply stop if we can't avoid cycles.
                    # Given the task is about disjoint cycles, we'll just fail and retry.
                    if rem_m > 0: return None
                
                return edges

            # Generate positive
            if len(positive_samples) < self.num_positive_samples:
                c_count = random.randint(threshold + 1, threshold + 7)
                edges = generate_with_count(c_count, m, n)
                if edges:
                    edges = self._permute_node_labels(edges)
                    edge_str = self._format_edge_list(edges)
                    edge_str = edge_str[:self.sequence_length].ljust(self.sequence_length, '0')
                    if self._count_cycles(edges) == c_count and edge_str not in positive_samples:
                        positive_samples.append(edge_str)
            
            # Generate negative
            if len(negative_samples) < self.num_negative_samples:
                c_count = random.randint(1, threshold)
                edges = generate_with_count(c_count, m, n)
                if edges:
                    edges = self._permute_node_labels(edges)
                    edge_str = self._format_edge_list(edges)
                    edge_str = edge_str[:self.sequence_length].ljust(self.sequence_length, '0')
                    if self._count_cycles(edges) == c_count and edge_str not in negative_samples:
                        negative_samples.append(edge_str)
        
        return positive_samples, negative_samples
    
    def _generate_cycle(self, n: int) -> List[Tuple[int, int]]:
        vertices = list(range(n))
        random.shuffle(vertices)
        edges = []
        for i in range(n):
            edges.append((vertices[i], vertices[(i + 1) % n]))
        return edges
    
    def _generate_non_cycle(self, n: int) -> List[Tuple[int, int]]:
        """Generate a directed acyclic graph (DAG) - no cycles allowed."""
        strategy = random.choice(['path', 'tree', 'disconnected_dag', 'star', 'layered'])
        
        if strategy == 'path':
            # Simple path: v0 -> v1 -> v2 -> ... -> v(n-1)
            vertices = list(range(n))
            random.shuffle(vertices)
            return [(vertices[i], vertices[i+1]) for i in range(n-1)] if n > 1 else []
        
        elif strategy == 'tree':
            # Tree structure (DAG)
            vertices = list(range(n))
            random.shuffle(vertices)
            edges = []
            for i in range(1, n):
                parent = random.choice(vertices[:i])
                edges.append((parent, vertices[i]))
            return edges
        
        elif strategy == 'disconnected_dag':
            # Multiple disconnected DAGs (paths or trees)
            if n < 4:
                return self._generate_path(n) if n > 1 else []
            split = n // 2
            # Generate two separate paths (no cycles)
            path1 = self._generate_path(split)
            path2_vertices = list(range(split, n))
            random.shuffle(path2_vertices)
            path2 = [(path2_vertices[i], path2_vertices[i+1]) for i in range(len(path2_vertices)-1)] if len(path2_vertices) > 1 else []
            return path1 + path2
        
        elif strategy == 'star':
            # Star: center -> all other vertices (DAG)
            center = 0
            edges = []
            for i in range(1, n):
                edges.append((center, i))
            return edges
        
        else:  # layered
            # Layered DAG: vertices in layers, edges only go forward
            if n < 2:
                return []
            vertices = list(range(n))
            random.shuffle(vertices)
            num_layers = random.randint(2, min(4, n))
            layer_size = n // num_layers
            edges = []
            for layer in range(num_layers - 1):
                start_idx = layer * layer_size
                end_idx = (layer + 1) * layer_size if layer < num_layers - 2 else n
                next_start = end_idx
                next_end = (layer + 2) * layer_size if layer + 1 < num_layers - 2 else n
                
                # Only create edges if there are vertices in the next layer
                if next_end > next_start:
                    for i in range(start_idx, end_idx):
                        # Connect to 1-2 vertices in next layer
                        max_connections = min(2, next_end - next_start)
                        if max_connections > 0:
                            num_connections = random.randint(1, max_connections)
                            targets = random.sample(range(next_start, next_end), num_connections)
                            for target in targets:
                                edges.append((vertices[i], vertices[target]))
            return edges
    
    def _generate_multiple_sccs(self, num_sccs: int) -> List[Tuple[int, int]]:
        """Generate num_sccs disjoint cycles (each cycle is 1 SCC)."""
        edges = []
        vertex_offset = 0
        remaining_vertices = self.max_vertices
        
        max_possible_sccs = remaining_vertices // 3
        actual_num_sccs = min(num_sccs, max_possible_sccs)
        
        if actual_num_sccs == 0:
            return edges
        
        for i in range(actual_num_sccs):
            if remaining_vertices < 3:
                break
            
            if i == actual_num_sccs - 1:
                cycle_size = remaining_vertices
            else:
                min_size = 3
                max_size = min(8, remaining_vertices - (actual_num_sccs - i - 1) * 3)
                if max_size < min_size:
                    max_size = min_size
                cycle_size = random.randint(min_size, max_size)
            
            cycle_edges = self._generate_cycle(cycle_size)
            cycle_edges_offset = [(u + vertex_offset, v + vertex_offset) 
                                  for u, v in cycle_edges]
            edges.extend(cycle_edges_offset)
            vertex_offset += cycle_size
            remaining_vertices -= cycle_size
        
        return edges
    
    def _generate_multiple_sccs_with_target(self, num_sccs: int, min_target_edges: int, max_target_edges: int) -> List[Tuple[int, int]]:
        """Generate num_sccs disjoint cycles targeting a specific number of edges."""
        target_edges = random.randint(min_target_edges, max_target_edges)
        edges = []
        vertex_offset = 0
        remaining_vertices = self.max_vertices
        
        max_possible_sccs = remaining_vertices // 3
        actual_num_sccs = min(num_sccs, max_possible_sccs)
        
        if actual_num_sccs == 0:
            return edges
        
        # Distribute target_edges among the SCCs
        # Each SCC needs at least 3 vertices (3 edges), so minimum total is 3 * num_sccs
        min_total_edges = 3 * actual_num_sccs
        target_edges = max(min_total_edges, min(target_edges, remaining_vertices))
        
        # Calculate base edges per SCC
        base_edges_per_scc = target_edges // actual_num_sccs
        extra_edges = target_edges % actual_num_sccs
        
        for i in range(actual_num_sccs):
            if remaining_vertices < 3:
                break
            
            # Calculate edges for this SCC
            edges_for_this_scc = base_edges_per_scc + (1 if i < extra_edges else 0)
            # Each edge corresponds to one vertex in a cycle, so cycle_size = edges_for_this_scc
            cycle_size = min(edges_for_this_scc, remaining_vertices)
            cycle_size = max(3, cycle_size)  # At least 3 for a valid cycle
            
            if cycle_size > remaining_vertices:
                cycle_size = remaining_vertices
            
            cycle_edges = self._generate_cycle(cycle_size)
            cycle_edges_offset = [(u + vertex_offset, v + vertex_offset) 
                                  for u, v in cycle_edges]
            edges.extend(cycle_edges_offset)
            vertex_offset += cycle_size
            remaining_vertices -= cycle_size
        
        return edges
    
    def _generate_dense_graph_with_sccs(self, num_sccs: int, min_target_edges: int, max_target_edges: int) -> List[Tuple[int, int]]:
        """Generate a dense graph with num_sccs SCCs by adding extra edges within cycles."""
        # First generate the base cycles
        base_min = max(3 * num_sccs, min_target_edges // 2)
        base_max = max_target_edges // 2
        if base_max < base_min:
            base_max = base_min
        base_edges = self._generate_multiple_sccs_with_target(num_sccs, base_min, base_max)
        
        # Extract vertex sets from base edges
        all_vertices = set()
        for u, v in base_edges:
            all_vertices.add(u)
            all_vertices.add(v)
        
        if not all_vertices:
            return base_edges
        
        # Add random edges within the vertex set to make it dense
        # These extra edges should stay within existing SCCs (won't create new ones if we're careful)
        target_edges = random.randint(min_target_edges, max_target_edges)
        current_edges = len(base_edges)
        edge_set = set(base_edges)
        
        # Add random edges, but avoid creating new SCCs
        # For simplicity, we'll add edges that connect vertices within the same component
        attempts = 0
        max_attempts = (max_target_edges - current_edges) * 10
        
        while current_edges < target_edges and len(base_edges) < max_target_edges and attempts < max_attempts:
            u = random.choice(list(all_vertices))
            v = random.choice(list(all_vertices))
            if u != v and (u, v) not in edge_set:
                base_edges.append((u, v))
                edge_set.add((u, v))
                current_edges += 1
            attempts += 1
        
        return base_edges
    
    def _generate_mixed_structure(self, min_target_edges: int, max_target_edges: int) -> List[Tuple[int, int]]:
        """Generate a mixed structure (cycles + paths/trees) with < 4 SCCs."""
        num_cycles = random.randint(1, 2)
        edges = self._generate_multiple_sccs(num_cycles)
        
        # Add a path or tree component
        used_vertices = set(v for edge in edges for v in edge)
        remaining_vertices = self.max_vertices - len(used_vertices)
        
        if remaining_vertices >= 2:
            path_size = random.randint(2, min(remaining_vertices, 8))
            path_edges = self._generate_path(path_size)
            offset = max(used_vertices) + 1 if used_vertices else 0
            edges.extend([(u + offset, v + offset) for u, v in path_edges])
        
        # Pad to target if needed
        target_edges = random.randint(min_target_edges, max_target_edges)
        all_vertices = set(v for edge in edges for v in edge)
        edge_set = set(edges)
        
        # Add more edges to reach target
        attempts = 0
        max_attempts = (max_target_edges - len(edges)) * 10
        
        while len(edges) < target_edges and len(edges) < max_target_edges and attempts < max_attempts:
            if len(all_vertices) < 2:
                break
            u = random.choice(list(all_vertices))
            v = random.choice(list(all_vertices))
            if u != v and (u, v) not in edge_set:
                edges.append((u, v))
                edge_set.add((u, v))
            attempts += 1
            if len(edges) >= target_edges:
                break
        
        return edges
    
    def _generate_path(self, n: int) -> List[Tuple[int, int]]:
        if n < 2:
            return []
        vertices = list(range(n))
        random.shuffle(vertices)
        return [(vertices[i], vertices[i+1]) for i in range(n-1)]
    
    def _format_edge_list(self, edges: List[Tuple[int, int]]) -> str:
        result = []
        for u, v in edges:
            result.append(f"{u:02d}{v:02d}")
        return ''.join(result)
    
    def _format_input(self, sample: str) -> np.ndarray:
        sample = sample[:self.sequence_length].ljust(self.sequence_length, '0')
        return np.array(list(sample), dtype='<U1')


class CycleDetectionDataGenerator(BaseDataGenerator):
    """fn_aa: Cycle Density task - does cycle count exceed threshold?
    
    Both classes have IDENTICAL: node count V, edge count E.
    Cycle count = E - V + C (cyclomatic complexity, where C = connected components).
    
    Class 1 (High cycles): cycle_count > threshold (more components)
    Class 0 (Low cycles): cycle_count <= threshold (fewer components)
    
    Format: u12v23 style (6 chars per edge: 'u' + 2-digit source + 'v' + 2-digit dest)
    This defeats TabPFN by fixing V and E, so only topology matters."""
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        # Each edge = 6 chars: u + 2 digits + v + 2 digits (e.g., "u12v23")
        self.num_edges = sequence_length // 6
        if self.num_edges < 4:
            raise ValueError(f"sequence_length {sequence_length} too short for graph tasks (need >= 24)")
        
        # Fix V = E (same vertices as edges)
        self.num_vertices = self.num_edges
        
        # Threshold for cycle count: cycle_count = E - V + C = C (when V = E)
        # So threshold is on number of connected components
        self.threshold = max(1, self.num_edges // 6)
        
        logger.info(f"CycleDetectionDataGenerator: E={self.num_edges}, V={self.num_vertices}, threshold={self.threshold}")

    def _count_components(self, edges: List[Tuple[int, int]], vertices: Set[int]) -> int:
        """Count connected components using Union-Find."""
        if not vertices:
            return 0
        parent = {v: v for v in vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        for u, v in edges:
            pu, pv = find(u), find(v)
            if pu != pv:
                parent[pu] = pv
        
        return len(set(find(v) for v in vertices))

    def _cycle_count(self, edges: List[Tuple[int, int]], vertices: Set[int]) -> int:
        """Calculate cyclomatic complexity = E - V + C."""
        E = len(edges)
        V = len(vertices)
        C = self._count_components(edges, vertices)
        return E - V + C

    def _generate_high_cycle_graph(self) -> List[Tuple[int, int]]:
        """Generate graph with high cycle count (many small components)."""
        E, V = self.num_edges, self.num_vertices
        vertices = list(range(V))
        edges = []
        edge_set = set()
        
        # Target: many small dense components (triangles, small cycles)
        # Each triangle uses 3 vertices and 3 edges, gives cycle_count contribution of 1
        target_components = self.threshold + random.randint(2, max(3, E // 8))
        
        # Create small cycles (triangles or small loops)
        v_idx = 0
        remaining_edges = E
        
        while remaining_edges >= 3 and v_idx + 2 < V:
            # Create a triangle
            cycle_size = min(3, remaining_edges, V - v_idx)
            if cycle_size < 3:
                break
            
            cycle_verts = vertices[v_idx:v_idx + cycle_size]
            for i in range(cycle_size):
                edge = (cycle_verts[i], cycle_verts[(i + 1) % cycle_size])
                edge_key = tuple(sorted(edge))
                if edge_key not in edge_set:
                    edges.append(edge)
                    edge_set.add(edge_key)
                    remaining_edges -= 1
            
            v_idx += cycle_size
            if len(edges) >= E:
                break
        
        # Fill remaining edges randomly within existing components (avoids connecting them)
        used_vertices = set()
        for u, v in edges:
            used_vertices.add(u)
            used_vertices.add(v)
        
        unused_vertices = [v for v in vertices if v not in used_vertices]
        
        # Add remaining vertices as isolated edges or small components
        while remaining_edges > 0 and len(unused_vertices) >= 2:
            u, v = unused_vertices.pop(), unused_vertices.pop()
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_set:
                edges.append((u, v))
                edge_set.add(edge_key)
                remaining_edges -= 1
        
        # If still need edges, add within existing components
        attempts = 0
        while len(edges) < E and attempts < E * 10:
            u, v = random.sample(vertices[:v_idx] if v_idx > 1 else vertices, 2)
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_set:
                edges.append((u, v))
                edge_set.add(edge_key)
            attempts += 1
        
        return edges[:E]

    def _generate_low_cycle_graph(self) -> List[Tuple[int, int]]:
        """Generate graph with low cycle count (one or few connected components)."""
        E, V = self.num_edges, self.num_vertices
        vertices = list(range(V))
        edges = []
        edge_set = set()
        
        # Build a spanning tree first (V-1 edges, 1 component, 0 cycles)
        random.shuffle(vertices)
        for i in range(1, V):
            parent = vertices[random.randint(0, i - 1)]
            edge = (parent, vertices[i])
            edges.append(edge)
            edge_set.add(tuple(sorted(edge)))
        
        # Add remaining edges (E - V + 1 edges) to create exactly that many cycles
        # But keep it as ONE component, so cycles = E - V + 1
        remaining = E - len(edges)
        attempts = 0
        while len(edges) < E and attempts < E * 10:
            u, v = random.sample(vertices, 2)
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_set:
                edges.append((u, v))
                edge_set.add(edge_key)
            attempts += 1
        
        return edges[:E]

    def _format_graph(self, edges: List[Tuple[int, int]]) -> str:
        """Format graph with ID scrambling, edge shuffling. Format: u12v23 per edge."""
        if not edges:
            return "u00v00" * self.num_edges
        
        # Get actual vertices used
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        
        # Scramble node IDs (use random IDs from 01-99)
        n = len(vertices)
        new_ids = random.sample(range(1, 100), min(n, 99))
        id_map = {old: new_ids[i] for i, old in enumerate(sorted(vertices))}
        
        # Format edges with random direction: u12v23 format
        edge_strs = []
        for u, v in edges:
            u_new, v_new = id_map[u], id_map[v]
            if random.random() > 0.5:
                edge_strs.append(f"u{u_new:02d}v{v_new:02d}")
            else:
                edge_strs.append(f"u{v_new:02d}v{u_new:02d}")
        
        random.shuffle(edge_strs)
        
        # Pad with u00v00 if needed
        while len(edge_strs) < self.num_edges:
            edge_strs.append("u00v00")
        
        return "".join(edge_strs[:self.num_edges])

    def _format_input(self, sample: str) -> np.ndarray:
        return np.array(list(sample), dtype='<U1')

    def _generate_raw_data(self) -> Tuple[List[str], List[str]]:
        positive_samples = []  # High cycle count
        negative_samples = []  # Low cycle count
        seen = set()
        
        while len(positive_samples) < self.num_positive_samples or len(negative_samples) < self.num_negative_samples:
            
            if len(positive_samples) < self.num_positive_samples:
                edges = self._generate_high_cycle_graph()
                vertices = set(v for e in edges for v in e)
                cc = self._cycle_count(edges, vertices)
                if cc > self.threshold:
                    edge_str = self._format_graph(edges)
                    if edge_str not in seen:
                        seen.add(edge_str)
                        positive_samples.append(edge_str)
            
            if len(negative_samples) < self.num_negative_samples:
                edges = self._generate_low_cycle_graph()
                vertices = set(v for e in edges for v in e)
                cc = self._cycle_count(edges, vertices)
                if cc <= self.threshold:
                    edge_str = self._format_graph(edges)
                    if edge_str not in seen:
                        seen.add(edge_str)
                        negative_samples.append(edge_str)
        
        return positive_samples, negative_samples

class CDCDiabetesDataGenerator(BaseDataGenerator):
    
    RAW_FEATURE_NAMES = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
    ]
    
    NUMERIC_INDICES = {3, 13, 14, 15, 18, 19, 20}
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        self._data_cache = None
        self._category_maps: Dict[int, Dict[str, str]] = {}
        self._maps_initialized = False
        numeric_indices_list = sorted(list(self.NUMERIC_INDICES))
        n_numeric = len(numeric_indices_list)
        self._A_diag = np.random.normal(0, 1, n_numeric)
        self._b = np.random.normal(0, 1, n_numeric)
        self._numeric_indices_list = numeric_indices_list
        logger.info(f"CDCDiabetesDataGenerator initialized for {num_samples} samples")
    
    def _init_category_maps(self, all_samples: List[Dict[str, str]]) -> None:
        if self._maps_initialized:
            return
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            if idx not in self.NUMERIC_INDICES:
                unique_vals = sorted(set(s[name] for s in all_samples))
                max_categories = len(unique_vals)
                pad_width = len(str(max_categories - 1))
                self._category_maps[idx] = {v: f"c{i:0{pad_width}d}" for i, v in enumerate(unique_vals)}
        self._maps_initialized = True
    
    def _transform_sample(self, raw: Dict[str, str]) -> Dict[str, str]:
        transformed = {}
        numeric_values = []
        for numeric_idx in self._numeric_indices_list:
            val = raw[self.RAW_FEATURE_NAMES[numeric_idx]]
            try:
                num_val = float(val)
                numeric_values.append(num_val)
            except ValueError:
                numeric_values.append(0.0)
        
        numeric_array = np.array(numeric_values)
        transformed_numeric = self._A_diag * numeric_array + self._b
        
        numeric_to_transformed = {idx: val for idx, val in zip(self._numeric_indices_list, transformed_numeric)}
        
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            new_key = f"x{idx}"
            if idx in self.NUMERIC_INDICES:
                transformed[new_key] = f"{numeric_to_transformed[idx]:.2f}"
            else:
                val = raw[name]
                transformed[new_key] = self._category_maps[idx].get(val, "c_unk")
        return transformed
    
    def _load_dataset(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if self._data_cache is not None:
            return self._data_cache
        
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            raise ImportError("Please install ucimlrepo: pip install ucimlrepo")
        
        logger.info("Fetching CDC Diabetes dataset from UCI repository...")
        cdc_diabetes = fetch_ucirepo(id=891)
        X = cdc_diabetes.data.features
        y = cdc_diabetes.data.targets
        
        all_raw_samples = []
        positive_raw = []
        negative_raw = []
        
        for idx in range(len(X)):
            features = {}
            for name in self.RAW_FEATURE_NAMES:
                if name in X.columns:
                    features[name] = str(X.iloc[idx][name])
            
            label = int(y.iloc[idx]['Diabetes_binary'])
            
            all_raw_samples.append(features)
            if label == 1:
                positive_raw.append(features)
            else:
                negative_raw.append(features)
        
        self._init_category_maps(all_raw_samples)
        
        positive_samples = [self._transform_sample(s) for s in positive_raw]
        negative_samples = [self._transform_sample(s) for s in negative_raw]
        
        logger.info(f"Loaded and transformed {len(positive_samples)} positive and {len(negative_samples)} negative samples")
        self._data_cache = (positive_samples, negative_samples)
        return self._data_cache
    
    def _generate_raw_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_positive, all_negative = self._load_dataset()
        
        if len(all_positive) < self.num_positive_samples:
            raise ValueError(f"Not enough positive samples: need {self.num_positive_samples}, have {len(all_positive)}")
        if len(all_negative) < self.num_negative_samples:
            raise ValueError(f"Not enough negative samples: need {self.num_negative_samples}, have {len(all_negative)}")
        
        sampled_positive = random.sample(all_positive, self.num_positive_samples)
        sampled_negative = random.sample(all_negative, self.num_negative_samples)
        
        return sampled_positive, sampled_negative
    
    def _format_input(self, sample: Dict[str, str]) -> np.ndarray:
        formatted = ','.join(f"{k}:{v}" for k, v in sample.items())
        return np.array([formatted])


class HTRU2DataGenerator(BaseDataGenerator):
    
    RAW_FEATURE_NAMES = [
        'Profile_mean', 'Profile_stdev', 'Profile_skewness', 'Profile_kurtosis',
        'DM_mean', 'DM_stdev', 'DM_skewness', 'DM_kurtosis'
    ]
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        self._data_cache = None
        n_numeric = len(self.RAW_FEATURE_NAMES)
        self._A_diag = np.random.normal(0, 1, n_numeric)
        self._b = np.random.normal(0, 1, n_numeric)
        logger.info(f"HTRU2DataGenerator initialized for {num_samples} samples")
    
    def _load_dataset(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if self._data_cache is not None:
            return self._data_cache
        
        import os
        import pandas as pd
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, 'HTRU_2.csv')
        
        if not os.path.exists(data_path):
            logger.info(f"Fetching HTRU2 dataset via ucimlrepo...")
            htru2 = fetch_ucirepo(id=372)
            X = htru2.data.features
            y = htru2.data.targets
            df = X.copy()
            df['class'] = y
            df.to_csv(data_path, index=False)
            logger.info(f"Saved dataset to {data_path}")
        
        df = pd.read_csv(data_path)
        all_raw_samples = []
        labels = []
        
        for _, row in df.iterrows():
            features = {}
            for i, name in enumerate(self.RAW_FEATURE_NAMES):
                features[name] = str(row[name])
            label = int(row['class'])
            all_raw_samples.append(features)
            labels.append(label)
        
        numeric_values_list = []
        for sample in all_raw_samples:
            numeric_values = [float(sample[name]) for name in self.RAW_FEATURE_NAMES]
            numeric_values_list.append(numeric_values)
        
        numeric_array = np.array(numeric_values_list)
        transformed_numeric = numeric_array * self._A_diag + self._b
        
        positive_samples = []
        negative_samples = []
        
        for sample_idx, raw_sample in enumerate(all_raw_samples):
            features = {}
            for feat_idx, name in enumerate(self.RAW_FEATURE_NAMES):
                features[f"x{feat_idx}"] = f"{transformed_numeric[sample_idx][feat_idx]:.2f}"
            
            if labels[sample_idx] == 1:
                positive_samples.append(features)
            else:
                negative_samples.append(features)
        
        logger.info(f"Loaded and transformed {len(positive_samples)} positive (pulsar) and {len(negative_samples)} negative (non-pulsar) samples")
        self._data_cache = (positive_samples, negative_samples)
        return self._data_cache
    
    def _generate_raw_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_positive, all_negative = self._load_dataset()
        
        if len(all_positive) < self.num_positive_samples:
            raise ValueError(f"Not enough positive samples: need {self.num_positive_samples}, have {len(all_positive)}")
        if len(all_negative) < self.num_negative_samples:
            raise ValueError(f"Not enough negative samples: need {self.num_negative_samples}, have {len(all_negative)}")
        
        sampled_positive = random.sample(all_positive, self.num_positive_samples)
        sampled_negative = random.sample(all_negative, self.num_negative_samples)
        
        return sampled_positive, sampled_negative
    
    def _format_input(self, sample: Dict[str, str]) -> np.ndarray:
        formatted = ','.join(f"{k}:{v}" for k, v in sample.items())
        return np.array([formatted])


class ChessDataGenerator(BaseDataGenerator):
    
    RAW_FEATURE_NAMES = [f'f{i}' for i in range(35)]
    
    NUMERIC_INDICES = set()
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        self._data_cache = None
        self._category_maps: Dict[int, Dict[str, str]] = {}
        self._maps_initialized = False
        logger.info(f"ChessDataGenerator initialized for {num_samples} samples")
    
    def _init_category_maps(self, all_samples: List[Dict[str, str]]) -> None:
        if self._maps_initialized:
            return
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            if idx not in self.NUMERIC_INDICES:
                unique_vals = sorted(set(s[name] for s in all_samples))
                max_categories = len(unique_vals)
                pad_width = len(str(max_categories - 1))
                self._category_maps[idx] = {v: f"c{i:0{pad_width}d}" for i, v in enumerate(unique_vals)}
        self._maps_initialized = True
    
    def _transform_sample(self, raw: Dict[str, str]) -> Dict[str, str]:
        transformed = {}
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            new_key = f"x{idx}"
            if idx in self.NUMERIC_INDICES:
                transformed[new_key] = raw[name]
            else:
                val = raw[name]
                transformed[new_key] = self._category_maps[idx].get(val, "c_unk")
        return transformed
    
    def _load_dataset(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if self._data_cache is not None:
            return self._data_cache
        
        import os
        import pandas as pd
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, 'kr-vs-kp.data')
        
        # Validate cached file format or fetch if missing
        needs_fetch = not os.path.exists(data_path)
        if not needs_fetch:
            # Quick validation: check first non-empty line has 36 fields and valid labels
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 36:
                        needs_fetch = True
                        logger.warning(f"Cached file has wrong format (expected 36 fields, got {len(parts)}), re-fetching...")
                        break
                    label_str = parts[35].strip()
                    if label_str not in ('won', 'nowin'):
                        needs_fetch = True
                        logger.warning(f"Cached file has wrong label format (expected 'won'/'nowin', got '{label_str}'), re-fetching...")
                        break
                    # Found one valid line, format looks correct
                    break
        
        if needs_fetch:
            logger.info(f"Fetching Chess dataset via ucimlrepo...")
            chess = fetch_ucirepo(id=22)
            X = chess.data.features
            y = chess.data.targets
            df = X.copy()
            df['class'] = y
            df.to_csv(data_path, header=False, index=False)
            logger.info(f"Saved dataset to {data_path}")
        
        all_raw_samples = []
        positive_raw = []
        negative_raw = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 36:
                    continue
                
                features = {}
                for i, name in enumerate(self.RAW_FEATURE_NAMES):
                    features[name] = parts[i]
                
                label_str = parts[35].strip()
                
                all_raw_samples.append(features)
                if label_str == 'won':
                    positive_raw.append(features)
                elif label_str == 'nowin':
                    negative_raw.append(features)
        
        self._init_category_maps(all_raw_samples)
        
        positive_samples = [self._transform_sample(s) for s in positive_raw]
        negative_samples = [self._transform_sample(s) for s in negative_raw]
        
        logger.info(f"Loaded and transformed {len(positive_samples)} positive (won) and {len(negative_samples)} negative (nowin) samples")
        self._data_cache = (positive_samples, negative_samples)
        return self._data_cache
    
    def _generate_raw_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_positive, all_negative = self._load_dataset()
        
        if len(all_positive) < self.num_positive_samples:
            raise ValueError(f"Not enough positive samples: need {self.num_positive_samples}, have {len(all_positive)}")
        if len(all_negative) < self.num_negative_samples:
            raise ValueError(f"Not enough negative samples: need {self.num_negative_samples}, have {len(all_negative)}")
        
        sampled_positive = random.sample(all_positive, self.num_positive_samples)
        sampled_negative = random.sample(all_negative, self.num_negative_samples)
        
        return sampled_positive, sampled_negative
    
    def _format_input(self, sample: Dict[str, str]) -> np.ndarray:
        formatted = ','.join(f"{k}:{v}" for k, v in sample.items())
        return np.array([formatted])


class MushroomDataGenerator(BaseDataGenerator):
    """Generates transformed data from UCI Secondary Mushroom dataset with anonymized features."""
    
    RAW_FEATURE_NAMES = [
        'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
        'gill-attachment', 'gill-spacing', 'gill-color', 'stem-height', 'stem-width',
        'stem-root', 'stem-surface', 'stem-color', 'veil-type', 'veil-color',
        'has-ring', 'ring-type', 'spore-print-color', 'habitat', 'season'
    ]
    
    NUMERIC_INDICES = {0, 8, 9}
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        self._data_cache = None
        self._category_maps: Dict[int, Dict[str, str]] = {}
        self._maps_initialized = False
        numeric_indices_list = sorted(list(self.NUMERIC_INDICES))
        n_numeric = len(numeric_indices_list)
        self._A_diag = np.random.normal(0, 1, n_numeric)
        self._b = np.random.normal(0, 1, n_numeric)
        self._numeric_indices_list = numeric_indices_list
        logger.info(f"MushroomDataGenerator initialized for {num_samples} samples")
    
    def _init_category_maps(self, all_samples: List[Dict[str, str]]) -> None:
        if self._maps_initialized:
            return
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            if idx not in self.NUMERIC_INDICES:
                unique_vals = sorted(set(s[name] for s in all_samples))
                max_categories = len(unique_vals)
                pad_width = len(str(max_categories - 1))
                self._category_maps[idx] = {v: f"c{i:0{pad_width}d}" for i, v in enumerate(unique_vals)}
        self._maps_initialized = True
    
    def _transform_sample(self, raw: Dict[str, str]) -> Dict[str, str]:
        transformed = {}
        numeric_values = []
        for numeric_idx in self._numeric_indices_list:
            val = raw[self.RAW_FEATURE_NAMES[numeric_idx]]
            try:
                num_val = float(val)
                numeric_values.append(num_val)
            except ValueError:
                numeric_values.append(0.0)
        
        numeric_array = np.array(numeric_values)
        transformed_numeric = self._A_diag * numeric_array + self._b
        
        numeric_to_transformed = {idx: val for idx, val in zip(self._numeric_indices_list, transformed_numeric)}
        
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            new_key = f"x{idx}"
            if idx in self.NUMERIC_INDICES:
                transformed[new_key] = f"{numeric_to_transformed[idx]:.2f}"
            else:
                val = raw[name]
                transformed[new_key] = self._category_maps[idx].get(val, "c_unk")
        return transformed
    
    def _load_dataset(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if self._data_cache is not None:
            return self._data_cache
        
        import os
        import pandas as pd
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        csv_path = os.path.join(cache_dir, 'mushroom_secondary_data.csv')
        
        if not os.path.exists(csv_path):
            logger.info(f"Fetching Mushroom dataset via ucimlrepo...")
            mushroom = fetch_ucirepo(id=848)
            X = mushroom.data.features
            y = mushroom.data.targets
            df = X.copy()
            df['class'] = y
            df.to_csv(csv_path, index=False, sep=';')
            logger.info(f"Saved dataset to {csv_path}")
        
        all_raw_samples = []
        positive_raw = []
        negative_raw = []
        
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            label_str = str(row['class']).strip()
            if label_str not in ['e', 'p']:
                continue
            
            features = {}
            for name in self.RAW_FEATURE_NAMES:
                features[name] = str(row.get(name, '')).strip()
            
            all_raw_samples.append(features)
            if label_str == 'e':
                positive_raw.append(features)
            else:
                negative_raw.append(features)
        
        self._init_category_maps(all_raw_samples)
        
        positive_samples = [self._transform_sample(s) for s in positive_raw]
        negative_samples = [self._transform_sample(s) for s in negative_raw]
        
        logger.info(f"Loaded and transformed {len(positive_samples)} positive and {len(negative_samples)} negative samples")
        self._data_cache = (positive_samples, negative_samples)
        return self._data_cache
    
    def _generate_raw_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_positive, all_negative = self._load_dataset()
        
        if len(all_positive) < self.num_positive_samples:
            raise ValueError(f"Not enough positive samples: need {self.num_positive_samples}, have {len(all_positive)}")
        if len(all_negative) < self.num_negative_samples:
            raise ValueError(f"Not enough negative samples: need {self.num_negative_samples}, have {len(all_negative)}")
        
        sampled_positive = random.sample(all_positive, self.num_positive_samples)
        sampled_negative = random.sample(all_negative, self.num_negative_samples)
        
        return sampled_positive, sampled_negative
    
    def _format_input(self, sample: Dict[str, str]) -> np.ndarray:
        formatted = ','.join(f"{k}:{v}" for k, v in sample.items())
        return np.array([formatted])


class AdultIncomeDataGenerator(BaseDataGenerator):
    """Generates transformed data from UCI Adult dataset with anonymized features."""
    
    RAW_FEATURE_NAMES = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]
    
    NUMERIC_INDICES = {0, 2, 4, 10, 11, 12}
    
    def __init__(self, sequence_length: int, num_samples: int, device: str = 'cpu'):
        super().__init__(sequence_length, num_samples, device)
        if num_samples % 2 != 0:
            raise ValueError("num_samples must be even for a 50/50 split.")
        self._data_cache = None
        self._category_maps: Dict[int, Dict[str, str]] = {}
        self._maps_initialized = False
        numeric_indices_list = sorted(list(self.NUMERIC_INDICES))
        n_numeric = len(numeric_indices_list)
        self._A_diag = np.random.normal(0, 1, n_numeric)
        self._b = np.random.normal(0, 1, n_numeric)
        self._numeric_indices_list = numeric_indices_list
        logger.info(f"AdultIncomeDataGenerator initialized for {num_samples} samples")
    
    def _init_category_maps(self, all_samples: List[Dict[str, str]]) -> None:
        if self._maps_initialized:
            return
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            if idx not in self.NUMERIC_INDICES:
                unique_vals = sorted(set(s[name] for s in all_samples))
                max_categories = len(unique_vals)
                pad_width = len(str(max_categories - 1))
                self._category_maps[idx] = {v: f"c{i:0{pad_width}d}" for i, v in enumerate(unique_vals)}
        self._maps_initialized = True
    
    def _transform_sample(self, raw: Dict[str, str]) -> Dict[str, str]:
        transformed = {}
        numeric_values = []
        for numeric_idx in self._numeric_indices_list:
            val = raw[self.RAW_FEATURE_NAMES[numeric_idx]]
            try:
                num_val = float(val)
                numeric_values.append(num_val)
            except ValueError:
                numeric_values.append(0.0)
        
        numeric_array = np.array(numeric_values)
        transformed_numeric = self._A_diag * numeric_array + self._b
        
        numeric_to_transformed = {idx: val for idx, val in zip(self._numeric_indices_list, transformed_numeric)}
        
        for idx, name in enumerate(self.RAW_FEATURE_NAMES):
            new_key = f"x{idx}"
            if idx in self.NUMERIC_INDICES:
                transformed[new_key] = f"{numeric_to_transformed[idx]:.2f}"
            else:
                val = raw[name]
                transformed[new_key] = self._category_maps[idx].get(val, "c_unk")
        return transformed
    
    def _load_dataset(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        if self._data_cache is not None:
            return self._data_cache
        
        import os
        import pandas as pd
        
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data_cache')
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, 'adult.data')
        
        if not os.path.exists(data_path):
            logger.info(f"Fetching Adult Income dataset via ucimlrepo...")
            adult = fetch_ucirepo(id=2)
            X = adult.data.features
            y = adult.data.targets
            df = X.copy()
            df['class'] = y
            df.to_csv(data_path, header=False, index=False)
            logger.info(f"Saved dataset to {data_path}")
        
        all_raw_samples = []
        positive_raw = []
        negative_raw = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('|'):
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 15:
                    continue
                
                label_str = parts[-1].rstrip('.')
                if label_str not in ['>50K', '<=50K']:
                    continue
                
                features = {}
                for i, name in enumerate(self.RAW_FEATURE_NAMES):
                    features[name] = parts[i]
                
                all_raw_samples.append(features)
                if label_str == '>50K':
                    positive_raw.append(features)
                else:
                    negative_raw.append(features)
        
        self._init_category_maps(all_raw_samples)
        
        positive_samples = [self._transform_sample(s) for s in positive_raw]
        negative_samples = [self._transform_sample(s) for s in negative_raw]
        
        logger.info(f"Loaded and transformed {len(positive_samples)} positive and {len(negative_samples)} negative samples")
        self._data_cache = (positive_samples, negative_samples)
        return self._data_cache
    
    def _generate_raw_data(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        all_positive, all_negative = self._load_dataset()
        
        if len(all_positive) < self.num_positive_samples:
            raise ValueError(f"Not enough positive samples: need {self.num_positive_samples}, have {len(all_positive)}")
        if len(all_negative) < self.num_negative_samples:
            raise ValueError(f"Not enough negative samples: need {self.num_negative_samples}, have {len(all_negative)}")
        
        sampled_positive = random.sample(all_positive, self.num_positive_samples)
        sampled_negative = random.sample(all_negative, self.num_negative_samples)
        
        return sampled_positive, sampled_negative
    
    def _format_input(self, sample: Dict[str, str]) -> np.ndarray:
        formatted = ','.join(f"{k}:{v}" for k, v in sample.items())
        return np.array([formatted])


def get_data_generator(target_name: str, sequence_length: int, num_samples: int) -> BaseDataGenerator:
    if target_name == 'dyck2':
        if sequence_length == 20:
            return Dyck2DataGenerator(sequence_length, num_samples, allow_duplicates=True)
        else:
            return Dyck2DataGenerator(sequence_length, num_samples)
    
    if target_name in ['patternmatch1', 'patternmatch2']:
        if target_name == 'patternmatch2':
            return PatternBasedDataGenerator(sequence_length, num_samples, pattern_string='00111111', allow_duplicates=True)
        else:
            return PatternBasedDataGenerator(sequence_length, num_samples)
    
    if target_name == "palindrome":
        if sequence_length == 20:
            return PalindromeDataGenerator(sequence_length, num_samples, allow_duplicates=True)
        else:
            return PalindromeDataGenerator(sequence_length, num_samples, allow_duplicates=True)
        
    if target_name == "prime_decimal":
        return PrimeDataGenerator(sequence_length, num_samples)
        
    if target_name == "prime_decimal_tf_check":
        return PrimeDecimalTailRestrictedDataGenerator(sequence_length, num_samples, allow_leading_zeros=False)
    
    if target_name == "prime_plus_47":
        return PrimePlus47DataGenerator(sequence_length, num_samples, allowed_nonprime_last_digits=(1, 3, 7, 9))
    
    if target_name == "graph_has_cycle":
        return CycleDetectionDataGenerator(sequence_length, num_samples)
    
    if target_name == "adult_income":
        return AdultIncomeDataGenerator(sequence_length, num_samples)
    
    if target_name == "mushroom":
        return MushroomDataGenerator(sequence_length, num_samples)
    
    if target_name == "cdc_diabetes":
        return CDCDiabetesDataGenerator(sequence_length, num_samples)
    
    if target_name == "htru2":
        return HTRU2DataGenerator(sequence_length, num_samples)
    
    if target_name == "chess":
        return ChessDataGenerator(sequence_length, num_samples)
    
    if target_name in TARGET_FUNCTIONS:
        return BinaryDataGenerator(target_name, sequence_length, num_samples)

    raise ValueError(f"No data generator found for target function '{target_name}'")

def create_stratified_splits(
    all_samples: List[Dict[str, Any]],
    train_size: int,
    val_size: int,
    test_size: int,
    device: str = 'cpu'
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates stratified train, validation, and test splits from a generated dataset.

    This utility ensures that each split maintains a balanced 50/50 class distribution,
    which is critical for the experiments.

    Args:
        all_samples: A list of generated data samples, where each sample is a
                     dictionary like {'Input': np.array, 'Output': '0' or '1'}.
        train_size: The desired number of samples in the training set.
        val_size: The desired number of samples in the validation set.
        test_size: The desired number of samples in the test set.
        device: The torch device to use for tensor operations.

    Returns:
        A tuple containing three lists of dictionaries:
        (train_split, validation_split, test_split).
    """
    if train_size % 2 != 0:
        raise ValueError("train_size must be even for a balanced split.")
    if val_size % 2 != 0:
        raise ValueError("val_size must be even for a balanced split.")

    # Convert the list of dicts to a format that's easier to split
    original_indices = list(range(len(all_samples)))
    all_labels = torch.tensor([int(s['Output']) for s in all_samples], device=device)

    # Separate indices by class
    indices_0 = torch.where(all_labels == 0)[0]
    indices_1 = torch.where(all_labels == 1)[0]

    # Deterministic shuffle of indices for each class
    shuffled_indices_0 = indices_0[torch.randperm(len(indices_0), device=device)]
    shuffled_indices_1 = indices_1[torch.randperm(len(indices_1), device=device)]

    # Calculate samples per class for each split
    train_per_class = train_size // 2
    val_per_class = val_size // 2

    if len(shuffled_indices_0) < train_per_class + val_per_class:
        raise ValueError("Not enough samples of class 0 for the requested train/val split size.")
    if len(shuffled_indices_1) < train_per_class + val_per_class:
        raise ValueError("Not enough samples of class 1 for the requested train/val split size.")

    # Create train indices
    train_indices_0 = shuffled_indices_0[:train_per_class]
    train_indices_1 = shuffled_indices_1[:train_per_class]
    train_indices = torch.cat([train_indices_0, train_indices_1])
    # Final shuffle to mix classes within the training set
    train_indices = train_indices[torch.randperm(len(train_indices), device=device)]

    # Create validation indices
    val_indices_0 = shuffled_indices_0[train_per_class : train_per_class + val_per_class]
    val_indices_1 = shuffled_indices_1[train_per_class : train_per_class + val_per_class]
    val_indices = torch.cat([val_indices_0, val_indices_1])

    # Create test indices from the remainder
    test_indices_0 = shuffled_indices_0[train_per_class + val_per_class:]
    test_indices_1 = shuffled_indices_1[train_per_class + val_per_class:]
    test_indices = torch.cat([test_indices_0, test_indices_1])

    # Reconstruct the splits using the original list and the selected indices
    train_split = [all_samples[i] for i in train_indices.tolist()]
    val_split = [all_samples[i] for i in val_indices.tolist()]
    test_split = [all_samples[i] for i in test_indices.tolist()]

    # Sanity checks
    assert len(train_split) == train_size
    assert len(val_split) == val_size
    assert len(test_split) == test_size

    return train_split, val_split, test_split