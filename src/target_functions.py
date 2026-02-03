# /src/target_functions.py
"""
Defines a canonical set of target functions for generating ground truth data.
Each function takes a PyTorch tensor and a device string, returning a tensor of 0s and 1s.
"""
import torch
import torch.nn.functional as F
import hashlib
import math
import random
from itertools import combinations
from typing import Callable, Dict, Any, Tuple
from sympy import isprime

# --- Core Implementations ---

def parity_all(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of all coordinates."""
    return (v.sum(dim=1) % 2).long()

def parity_first_half(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of the first half of the coordinates."""
    return (v[:, :v.shape[1]//2].sum(dim=1) % 2).long()

def automata_parity(v: torch.Tensor, device: str) -> torch.Tensor:
    """Applies a rule-based transformation on a sliding window of size 3 (Rule 30-like)."""
    inds = F.pad(v.float(), (1, 1), 'constant', 0).unfold(1, 3, 1).matmul(torch.tensor([4, 2, 1], device=device, dtype=torch.float))
    rule = torch.tensor([0, 1, 1, 1, 1, 0, 0, 0], device=device, dtype=torch.float)
    return (rule[inds.long()].sum(1) % 2).long()

def sha256_parity(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of '1' bits in the SHA-256 hash of the binary string."""
    results = []
    for row in v:
        row_str = ''.join(map(str, row.int().tolist()))
        hashed = hashlib.sha256(row_str.encode()).hexdigest()
        binary_hash = bin(int(hashed, 16))[2:]
        results.append(binary_hash.count('1') % 2)
    return torch.tensor(results, device=device, dtype=torch.long)

def palindrome(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if each binary sequence in the batch is a palindrome."""
    flipped_v = torch.flip(v, dims=[1])
    return torch.all(v == flipped_v, dim=1).long()

def dyck2(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for valid Dyck-2 sequences, e.g., '()[]'."""
    pmap = {"00": "(", "01": ")", "10": "[", "11": "]"}
    match = {')': '(', ']': '['}
    def to_paren(row):
        return "".join(pmap.get(f"{row[i]}{row[i+1]}", "?") for i in range(0, len(row), 2))

    def is_valid(s):
        stack = []
        for c in s:
            if c in match.values(): stack.append(c)
            elif not stack or stack.pop() != match[c]: return 0
        return int(not stack)

    return torch.tensor([is_valid(to_paren(r.tolist())) for r in v], device=device, dtype=torch.long)

def parity_rand_3(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of 3 random but fixed coordinates."""
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    idx = torch.randperm(v.shape[1], generator=generator, device=device)[:3]
    return (v[:, idx].sum(dim=1) % 2).long()

def parity_rand_10(v: torch.Tensor, device: str) -> torch.Tensor:
    """Parity of 10 random but fixed coordinates."""
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    idx = torch.randperm(v.shape[1], generator=generator, device=device)[:10]
    return (v[:, idx].sum(dim=1) % 2).long()

def patternmatch1(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for the presence of the pattern '10101010'."""
    pattern_str = '10101010'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def patternmatch2(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks for the presence of the pattern '00111111'."""
    pattern_str = '00111111'
    return torch.tensor([pattern_str in "".join(map(str, row.tolist())) for row in v], dtype=torch.long, device=device)

def prime_decimal(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if a decimal number is prime."""
    numbers = [int("".join(map(str, row.tolist()))) for row in v]
    return torch.tensor([isprime(n) for n in numbers], dtype=torch.long, device=device)

def prime_decimal_tf_check(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if a decimal number is prime (identical to prime_decimal, used for different generator)."""
    return prime_decimal(v, device)

def two_primes(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if two numbers have the same primality status.
    Input format: concatenated decimal digits of two numbers, each of length seq_len.
    Returns 1 if both are prime OR both are non-prime, 0 otherwise."""
    results = []
    for row in v:
        digits = ''.join(map(str, row.int().tolist()))
        seq_len = len(digits) // 2
        n1_str = digits[:seq_len]
        n2_str = digits[seq_len:]
        n1 = int(n1_str) if n1_str else 0
        n2 = int(n2_str) if n2_str else 0
        if n1 < 2 or n2 < 2:
            results.append(0)
            continue
        n1_prime = isprime(n1)
        n2_prime = isprime(n2)
        results.append(1 if n1_prime == n2_prime else 0)
    return torch.tensor(results, device=device, dtype=torch.long)

def prime_minus_5(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if n-5 is prime for each number n."""
    results = []
    for row in v:
        n = int("".join(map(str, row.tolist())))
        if n - 5 < 2:
            results.append(0)
            continue
        results.append(1 if isprime(n - 5) else 0)
    return torch.tensor(results, device=device, dtype=torch.long)

def prime_plus_47(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if n-47 is prime for each number n."""
    results = []
    for row in v:
        n = int("".join(map(str, row.tolist())))
        if n - 47 < 2:
            results.append(0)
            continue
        results.append(1 if isprime(n - 47) else 0)
    return torch.tensor(results, device=device, dtype=torch.long)

def gcd_range(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if GCD(n1, n2) is in [20, 51].
    Input format: concatenated decimal digits of two numbers, each of length seq_len."""
    results = []
    for row in v:
        digits = ''.join(map(str, row.int().tolist()))
        seq_len = len(digits) // 2
        n1_str = digits[:seq_len]
        n2_str = digits[seq_len:]
        n1 = int(n1_str) if n1_str else 0
        n2 = int(n2_str) if n2_str else 0
        if n1 == 0 or n2 == 0:
            results.append(0)
            continue
        g = math.gcd(n1, n2)
        if 20 <= g <= 51:
            results.append(1)
        else:
            results.append(0)
    return torch.tensor(results, device=device, dtype=torch.long)

_poly_cache = {}

def _generate_monomials(n: int, k: int, d: int, T: int, seed: int) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[int, ...]]:
    rng = random.Random(seed)
    relevant = tuple(sorted(rng.sample(range(n), k)))
    
    all_possible = []
    for deg in range(1, d + 1):
        for combo in combinations(relevant, deg):
            all_possible.append(combo)
    
    if len(all_possible) < T:
        T = len(all_possible)
    
    monomials = tuple(sorted(rng.sample(all_possible, T)))
    return monomials, relevant

def poly_f2_deg3(v: torch.Tensor, device: str) -> torch.Tensor:
    """Evaluates a random 3rd degree polynomial over GF(2) on binary inputs."""
    n = v.shape[1]
    
    if n not in _poly_cache:
        seed = hash(f"poly_deg3_n{n}") % (2**31)
        k = min(10, max(5, n // 10))
        T = 3
        monomials, relevant = _generate_monomials(n, k, 3, T, seed)
        _poly_cache[n] = (monomials, relevant)
    
    monomials, relevant = _poly_cache[n]
    
    results = []
    for row in v:
        x = row.int().tolist()
        result = 0
        for mono in monomials:
            term = 1
            for idx in mono:
                term &= x[idx]
                if term == 0:
                    break
            result ^= term
        results.append(result)
    
    return torch.tensor(results, device=device, dtype=torch.long)

def graph_has_cycle(v: torch.Tensor, device: str) -> torch.Tensor:
    """Checks if graph has high cycle density (cycle count > threshold).
    Input format: u12v23 style (6 chars per edge: 'u' + 2-digit src + 'v' + 2-digit dst).
    Accepts both character format (for data generation) and numeric token format (for LLM training).
    Numeric tokens: 'u'=10, 'v'=11, digits 0-9=themselves.
    Cycle count = E - V + C (cyclomatic complexity).
    Returns 1 if cycle_count > threshold, 0 otherwise."""
    results = []
    for row in v:
        tokens = row.int().tolist()
        
        # Decode based on format:
        # - Numeric tokens (LLM training): 10='u', 11='v', 0-9=digits
        # - ASCII codes (data generation): 117='u', 118='v', 48-57='0'-'9'
        # - Direct digits: 0-9 as-is
        decoded = []
        for t in tokens:
            if t == 10:  # Numeric token for 'u'
                decoded.append('u')
            elif t == 11:  # Numeric token for 'v'
                decoded.append('v')
            elif t == 117:  # ASCII 'u'
                decoded.append('u')
            elif t == 118:  # ASCII 'v'
                decoded.append('v')
            elif 0 <= t <= 9:  # Direct digit
                decoded.append(str(t))
            elif 48 <= t <= 57:  # ASCII digit '0'-'9'
                decoded.append(chr(t))
            # Skip other values (padding, etc.)
        raw = ''.join(decoded)
        
        edges = []
        vertices = set()
        # Parse u12v23 format: each edge is 6 chars
        import re
        edge_pattern = re.compile(r'u(\d{2})v(\d{2})')
        for match in edge_pattern.finditer(raw):
            v1, v2 = int(match.group(1)), int(match.group(2))
            if v1 == 0 and v2 == 0:
                continue
            edges.append((v1, v2))
            vertices.add(v1)
            vertices.add(v2)
        
        if not edges or not vertices:
            results.append(0)
            continue
        
        E = len(edges)
        V = len(vertices)
        
        # Count connected components using Union-Find
        parent = {v: v for v in vertices}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        for u, w in edges:
            pu, pw = find(u), find(w)
            if pu != pw:
                parent[pu] = pw
        
        C = len(set(find(v) for v in vertices))
        
        # Cyclomatic complexity = E - V + C
        cycle_count = E - V + C
        
        # Threshold based on number of edges
        threshold = max(1, E // 6)
        
        results.append(1 if cycle_count > threshold else 0)
    
    return torch.tensor(results, device=device, dtype=torch.long)

def tabular_dummy(v: torch.Tensor, device: str) -> torch.Tensor:
    """Dummy function for tabular datasets (labels come from data generators)."""
    return torch.zeros(v.size(0), dtype=torch.long, device=device)

TARGET_FUNCTIONS: Dict[str, Callable[[torch.Tensor, str], torch.Tensor]] = {
    'parity_all': parity_all,
    'parity_first_half': parity_first_half,
    'patternmatch1': patternmatch1,
    'patternmatch2': patternmatch2,
    'parity_rand_3': parity_rand_3,
    'parity_rand_10': parity_rand_10,
    'palindrome': palindrome,
    'dyck2': dyck2,
    'prime_decimal': prime_decimal,
    'automata_parity': automata_parity,
    'prime_decimal_tf_check': prime_decimal_tf_check,
    'sha256_parity': sha256_parity,
    'two_primes': two_primes,
    'prime_minus_5': prime_minus_5,
    'prime_plus_47': prime_plus_47,
    'gcd_range': gcd_range,
    'poly_f2_deg3': poly_f2_deg3,
    'graph_has_cycle': graph_has_cycle,
    'adult_income': tabular_dummy,
    'mushroom': tabular_dummy,
    'cdc_diabetes': tabular_dummy,
    'breast_cancer': tabular_dummy,
    'htru2': tabular_dummy,
    'chess': tabular_dummy,
}

EXPERIMENT_FUNCTION_MAPPING: Dict[str, str] = {
    "fn_a": "parity_all",
    "fn_b": "parity_first_half",
    "fn_c": "patternmatch1",
    "fn_d": "patternmatch2",
    "fn_e": "parity_rand_3",
    "fn_f": "parity_rand_10",
    "fn_g": "palindrome",
    "fn_h": "dyck2",
    "fn_i": "prime_decimal",
    "fn_j": "automata_parity",
    "fn_k": "prime_decimal_tf_check",
    "fn_l": "sha256_parity",
    "fn_v": "prime_plus_47",
    # Tabular datasets
    "fn_m": "adult_income",
    "fn_n": "mushroom",
    "fn_o": "cdc_diabetes",
    "fn_p": "htru2",
    "fn_q": "chess",
    # Graph problems
    "fn_aa": "graph_has_cycle",
}

EXPERIMENT_FUNCTION_METADATA: Dict[str, Dict[str, Any]] = {
    "fn_h": {
        "lengths": [100, 80, 60, 40, 20]
    },
    "fn_m": {
        "lengths": [14],
        "tabular": True
    },
    "fn_n": {
        "lengths": [20],
        "tabular": True
    },
    "fn_o": {
        "lengths": [21],
        "tabular": True
    },
    "fn_p": {
        "lengths": [8],
        "tabular": True
    },
    "fn_q": {
        "lengths": [35],
        "tabular": True
    },
}