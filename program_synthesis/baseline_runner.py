"""Baseline ML models for comparison with LLM-ERM."""

import os, sys, json, csv, time, argparse, random, hashlib, pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import logging
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

from src.data_handler import get_data_generator, create_stratified_splits
from src.target_functions import EXPERIMENT_FUNCTION_MAPPING, EXPERIMENT_FUNCTION_METADATA

FUNCTION_NAME_MAPPING = EXPERIMENT_FUNCTION_MAPPING
TABULAR_FNS = {"adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}
BOOLEAN_FNS = {"parity_all", "parity_first_half", "parity_rand_3", "parity_rand_10", 
               "automata_parity", "palindrome", "dyck2", "patternmatch1", "patternmatch2",
               "prime_decimal", "prime_decimal_tf_check", "sha256_parity", "prime_plus_47", "collatz_steps_parity",
               "graph_has_cycle", "graph_connected",
               "adult_income", "mushroom", "cdc_diabetes", "htru2", "chess"}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("baseline_runner")

OPS_UNARY = ["NOT"]
OPS_BINARY = ["XOR", "AND", "OR"]
OPS_ALL = OPS_UNARY + OPS_BINARY


@dataclass
class Node:
    op: str
    idx: Optional[int] = None
    val: Optional[int] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    
    def is_leaf(self) -> bool:
        return self.op in ("VAR", "CONST")
    
    def size(self) -> int:
        if self.is_leaf():
            return 1
        if self.op in OPS_UNARY:
            return 1 + self.left.size()
        return 1 + self.left.size() + self.right.size()
    
    def depth(self) -> int:
        """Returns the maximum depth of the tree (number of edges from root to deepest leaf)."""
        if self.is_leaf():
            return 0
        if self.op in OPS_UNARY:
            return 1 + self.left.depth()
        return 1 + max(self.left.depth(), self.right.depth())
    
    def clone(self) -> "Node":
        if self.is_leaf():
            return Node(self.op, self.idx, self.val)
        if self.op in OPS_UNARY:
            return Node(self.op, left=self.left.clone())
        return Node(self.op, left=self.left.clone(), right=self.right.clone())
    
    def eval(self, X: np.ndarray) -> np.ndarray:
        if self.op == "VAR":
            return X[:, self.idx].astype(bool)
        if self.op == "CONST":
            return np.full(X.shape[0], bool(self.val), dtype=bool)
        if self.op == "NOT":
            return np.logical_not(self.left.eval(X))
        a = self.left.eval(X)
        b = self.right.eval(X)
        if self.op == "XOR":
            return np.logical_xor(a, b)
        if self.op == "AND":
            return np.logical_and(a, b)
        if self.op == "OR":
            return np.logical_or(a, b)
        raise ValueError(f"Unknown op: {self.op}")
    
    def to_str(self) -> str:
        if self.op == "VAR":
            return f"x{self.idx}"
        if self.op == "CONST":
            return str(self.val)
        if self.op == "NOT":
            return f"(~{self.left.to_str()})"
        sym = {"XOR": "^", "AND": "&", "OR": "|"}[self.op]
        return f"({self.left.to_str()} {sym} {self.right.to_str()})"


def random_leaf(d: int, p_const: float = 0.1) -> Node:
    if random.random() < p_const:
        return Node("CONST", val=random.randint(0, 1))
    return Node("VAR", idx=random.randrange(d))


def random_tree(d: int, max_depth: int, p_const: float = 0.1) -> Node:
    if max_depth <= 0 or random.random() < 0.25:
        return random_leaf(d, p_const=p_const)
    op = random.choice(OPS_ALL)
    if op in OPS_UNARY:
        return Node(op, left=random_tree(d, max_depth - 1, p_const=p_const))
    return Node(
        op,
        left=random_tree(d, max_depth - 1, p_const=p_const),
        right=random_tree(d, max_depth - 1, p_const=p_const),
    )


def iter_nodes_with_parents(root: Node):
    stack = [(root, None, None)]
    while stack:
        node, parent, is_left = stack.pop()
        yield node, parent, is_left
        if node.is_leaf():
            continue
        if node.op in OPS_UNARY:
            stack.append((node.left, node, True))
        else:
            stack.append((node.right, node, False))
            stack.append((node.left, node, True))


def pick_random_subtree(root: Node) -> Tuple[Node, Optional[Node], Optional[bool]]:
    nodes = list(iter_nodes_with_parents(root))
    return random.choice(nodes)


def replace_child(parent: Node, is_left: bool, new_child: Node):
    if parent.op in OPS_UNARY:
        parent.left = new_child
    else:
        if is_left:
            parent.left = new_child
        else:
            parent.right = new_child


def crossover(a: Node, b: Node) -> Tuple[Node, Node]:
    a2 = a.clone()
    b2 = b.clone()
    na, pa, is_left_a = pick_random_subtree(a2)
    nb, pb, is_left_b = pick_random_subtree(b2)
    if pa is None and pb is None:
        return b2, a2
    if pa is None:
        a2 = nb.clone()
    else:
        replace_child(pa, is_left_a, nb.clone())
    if pb is None:
        b2 = na.clone()
    else:
        replace_child(pb, is_left_b, na.clone())
    return a2, b2


def mutate(root: Node, d: int, max_depth: int, p_const: float = 0.1) -> Node:
    r = root.clone()
    n, p, is_left = pick_random_subtree(r)
    new_sub = random_tree(d, max_depth=max_depth, p_const=p_const)
    if p is None:
        return new_sub
    replace_child(p, is_left, new_sub)
    return r


def ga_accuracy(pred_bool: np.ndarray, y: np.ndarray) -> float:
    yb = y.astype(bool)
    return float(np.mean(pred_bool == yb))


class GeneticAlgorithmClassifier:
    def __init__(
        self,
        pop_size: int = 300,
        generations: int = 80,
        max_depth_init: int = 4,
        max_depth_mut: int = 4,
        tournament_k: int = 5,
        cx_prob: float = 0.6,
        mut_prob: float = 0.3,
        elite_frac: float = 0.05,
        size_penalty: float = 1e-4,
        fitness_subsample: int = 4000,
        random_state: int = 0,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth_init = max_depth_init
        self.max_depth_mut = max_depth_mut
        self.tournament_k = tournament_k
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite_frac = elite_frac
        self.size_penalty = size_penalty
        self.fitness_subsample = fitness_subsample
        self.random_state = random_state
        self.best_node_ = None
        self.logs_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        X = X.astype(bool)
        y = y.astype(int)
        n, d = X.shape
        
        def eval_fitness(node: Node) -> float:
            if self.fitness_subsample is not None and self.fitness_subsample < n:
                idx = np.random.choice(n, size=self.fitness_subsample, replace=False)
                pred = node.eval(X[idx])
                acc = ga_accuracy(pred, y[idx])
            else:
                pred = node.eval(X)
                acc = ga_accuracy(pred, y)
            return acc - self.size_penalty * node.size()
        
        def tournament_select(pop: List[Node], fits: List[float]) -> Node:
            cand = random.sample(range(len(pop)), k=min(self.tournament_k, len(pop)))
            best_i = max(cand, key=lambda i: fits[i])
            return pop[best_i]
        
        pop = [random_tree(d, self.max_depth_init) for _ in range(self.pop_size)]
        logs = {"best_train_acc": [], "best_val_acc": [], "best_size": []}
        best_node = None
        best_fit = -1e9
        
        for gen in range(self.generations):
            fits = [eval_fitness(ind) for ind in pop]
            gen_best_i = int(np.argmax(fits))
            gen_best = pop[gen_best_i]
            gen_best_fit = fits[gen_best_i]
            
            if gen_best_fit > best_fit:
                best_fit = gen_best_fit
                best_node = gen_best.clone()
            
            train_acc = ga_accuracy(best_node.eval(X), y)
            if X_val is not None and y_val is not None:
                X_val_b = X_val.astype(bool)
                y_val_i = y_val.astype(int)
                val_acc = ga_accuracy(best_node.eval(X_val_b), y_val_i)
            else:
                val_acc = float("nan")
            
            logs["best_train_acc"].append(train_acc)
            logs["best_val_acc"].append(val_acc)
            logs["best_size"].append(best_node.size())
            
            elite_n = max(1, int(self.elite_frac * self.pop_size))
            elite_idx = np.argsort(fits)[-elite_n:]
            new_pop = [pop[i].clone() for i in elite_idx]
            
            while len(new_pop) < self.pop_size:
                r = random.random()
                if r < self.cx_prob and len(new_pop) + 1 < self.pop_size:
                    p1 = tournament_select(pop, fits)
                    p2 = tournament_select(pop, fits)
                    c1, c2 = crossover(p1, p2)
                    new_pop.extend([c1, c2])
                elif r < self.cx_prob + self.mut_prob:
                    p = tournament_select(pop, fits)
                    c = mutate(p, d=d, max_depth=self.max_depth_mut)
                    new_pop.append(c)
                else:
                    p = tournament_select(pop, fits)
                    new_pop.append(p.clone())
            
            pop = new_pop[:self.pop_size]
        
        self.best_node_ = best_node
        self.logs_ = logs
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_node_ is None:
            raise RuntimeError("Model not fitted yet")
        X_bool = X.astype(bool)
        return self.best_node_.eval(X_bool).astype(int)


@dataclass
class Config:
    functions: List[str] = field(default_factory=lambda: [
        "fn_a", "fn_b", "fn_c", "fn_d", "fn_e", "fn_f",
        "fn_g", "fn_h", "fn_i", "fn_j", "fn_k", "fn_l", "fn_v", "fn_t",
        "fn_m", "fn_n", "fn_o", "fn_p", "fn_q", "fn_aa",
    ])
    lengths: List[int] = field(default_factory=lambda: [100, 50, 30, 25, 20])
    train_size: int = int(os.getenv("TRAIN_SIZE", "100"))
    val_size: int = int(os.getenv("VAL_SIZE", "100"))
    test_size: int = int(os.getenv("TEST_SIZE", "10000"))
    seed: int = int(os.getenv("GLOBAL_SEED", "42"))
    num_trials: int = int(os.getenv("NUM_TRIALS", "10"))
    dataset_dir: str = os.getenv("DATASET_DIR", "program_synthesis/datasets")
    out_jsonl: str = "program_synthesis/baseline_results.jsonl"
    out_csv: str = "program_synthesis/baseline_results.csv"


class DatasetStore:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

    def _paths(self, target_name: str, L: int, derived_seed: int) -> Dict[str, str]:
        base = os.path.join(self.cfg.dataset_dir, target_name, f"L{L}", f"seed{derived_seed}")
        return {
            "dir": base,
            "train": os.path.join(base, "train.txt"),
            "val": os.path.join(base, "val.txt"),
            "test": os.path.join(base, "test.txt"),
        }

    def _stable_derived_seed(self, fn: str, L: int) -> int:
        key = f"{fn}|L={L}|train={self.cfg.train_size+self.cfg.val_size}|test={self.cfg.test_size}|base_seed={self.cfg.seed}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") & 0x7FFFFFFF

    def _read_lines(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]

    def get(self, fn: str, L: int) -> Tuple[List[str], List[str], List[str]]:
        target_name = FUNCTION_NAME_MAPPING[fn]
        derived_seed = self._stable_derived_seed(fn, L)
        paths = self._paths(target_name, L, derived_seed)

        if all(os.path.exists(paths[k]) for k in ["train", "val", "test"]):
            return self._read_lines(paths["train"]), self._read_lines(paths["val"]), self._read_lines(paths["test"])

        self._set_seed(derived_seed)
        total_samples = self.cfg.train_size + self.cfg.val_size + self.cfg.test_size
        generator = get_data_generator(target_name, L, total_samples)
        all_samples = generator.generate_data()

        train_split, val_split, test_split = create_stratified_splits(
            all_samples, self.cfg.train_size, self.cfg.val_size, self.cfg.test_size, device='cpu'
        )

        os.makedirs(paths["dir"], exist_ok=True)
        for split_name, split_data in [("train", train_split), ("val", val_split), ("test", test_split)]:
            with open(paths[split_name], "w") as f:
                for s in split_data:
                    input_val = s['Input']
                    if isinstance(input_val, np.ndarray):
                        input_str = input_val.item() if input_val.size == 1 else ''.join(input_val.tolist())
                    else:
                        input_str = ''.join(input_val) if isinstance(input_val, (list, tuple)) else str(input_val)
                    f.write(f"{input_str} -> {s['Output']}\n")

        return self._read_lines(paths["train"]), self._read_lines(paths["val"]), self._read_lines(paths["test"])


def parse_data(lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for line in lines:
        seq, label = line.split("->")
        # Encode characters: digits 0-9 stay as-is, 'u' -> 10, 'v' -> 11, filter out spaces/others
        encoded = []
        for c in seq.strip():
            if c.isdigit():
                encoded.append(int(c))
            elif c == 'u':
                encoded.append(10)
            elif c == 'v':
                encoded.append(11)
            # Skip spaces and other characters
        X.append(encoded)
        y.append(int(label.strip()))
    return np.array(X), np.array(y)


class TabularDataParser:
    ADULT_INCOME_NUMERIC = {'x0', 'x2', 'x4', 'x10', 'x11', 'x12'}
    MUSHROOM_NUMERIC = {'x0', 'x8', 'x9'}
    CDC_DIABETES_NUMERIC = {'x3', 'x13', 'x14', 'x15', 'x18', 'x19', 'x20'}
    BREAST_CANCER_NUMERIC = {f'x{i}' for i in range(30)}
    HTRU2_NUMERIC = {f'x{i}' for i in range(8)}
    CHESS_NUMERIC = set()
    
    def __init__(self, dataset_name: str = 'adult_income'):
        self.feature_names: List[str] = []
        self._fitted = False
        self.numeric_indices: List[int] = []
        self.categorical_indices: List[int] = []
        if dataset_name == 'mushroom':
            self.numeric_features = self.MUSHROOM_NUMERIC
        elif dataset_name == 'cdc_diabetes':
            self.numeric_features = self.CDC_DIABETES_NUMERIC
        elif dataset_name == 'htru2':
            self.numeric_features = self.HTRU2_NUMERIC
        elif dataset_name == 'chess':
            self.numeric_features = self.CHESS_NUMERIC
        else:
            self.numeric_features = self.ADULT_INCOME_NUMERIC
    
    def _parse_line(self, line: str) -> Tuple[Dict[str, str], int]:
        features_str, label = line.split("->")
        label = int(label.strip())
        features = {}
        for pair in features_str.strip().split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                features[k.strip()] = v.strip()
        return features, label
    
    def fit_transform(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        parsed = [self._parse_line(line) for line in lines]
        if not parsed:
            return np.array([]), np.array([])
        
        self.feature_names = list(parsed[0][0].keys())
        self.numeric_indices = [i for i, f in enumerate(self.feature_names) if f in self.numeric_features]
        self.categorical_indices = [i for i, f in enumerate(self.feature_names) if f not in self.numeric_features]
        self._fitted = True
        return self._transform(parsed)
    
    def transform(self, lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Parser must be fitted before transform")
        parsed = [self._parse_line(line) for line in lines]
        return self._transform(parsed)
    
    def _transform(self, parsed: List[Tuple[Dict[str, str], int]]) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for features, label in parsed:
            row = []
            for feat in self.feature_names:
                val = features.get(feat, '?')
                if feat in self.numeric_features:
                    try:
                        row.append(float(val))
                    except ValueError:
                        row.append(0.0)
                else:
                    if val.startswith('c') and val[1:].isdigit():
                        row.append(float(val[1:]))
                    else:
                        row.append(-1.0)
            X.append(row)
            y.append(label)
        return np.array(X), np.array(y)


class GABinarizer:
    def __init__(self):
        self.numeric_thresholds = None
        self.category_values = {}
    
    def fit_transform(self, X: np.ndarray, numeric_indices: List[int], categorical_indices: List[int]) -> np.ndarray:
        parts = []
        
        if numeric_indices:
            X_num = X[:, numeric_indices]
            self.numeric_thresholds = np.median(X_num, axis=0)
            parts.append((X_num > self.numeric_thresholds).astype(int))
        
        for idx in categorical_indices:
            col = X[:, idx].astype(int)
            unique_vals = np.unique(col)
            self.category_values[idx] = unique_vals
            one_hot = (col[:, None] == unique_vals[None, :]).astype(int)
            parts.append(one_hot)
        
        return np.hstack(parts) if parts else np.zeros((X.shape[0], 0), dtype=int)
    
    def transform(self, X: np.ndarray, numeric_indices: List[int], categorical_indices: List[int]) -> np.ndarray:
        parts = []
        
        if numeric_indices:
            X_num = X[:, numeric_indices]
            parts.append((X_num > self.numeric_thresholds).astype(int))
        
        for idx in categorical_indices:
            col = X[:, idx].astype(int)
            unique_vals = self.category_values[idx]
            one_hot = (col[:, None] == unique_vals[None, :]).astype(int)
            parts.append(one_hot)
        
        return np.hstack(parts) if parts else np.zeros((X.shape[0], 0), dtype=int)


def get_param_grids(tabular: bool = False, boolean: bool = False) -> Dict[str, Dict[str, List[Any]]]:
    grids = {
        "svm": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "sigmoid"]
        },
        "random_forest": {
            "n_estimators": [64, 128, 256],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10]
        },
        "decision_tree": {
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ["gini", "entropy"]
        },
        "xgboost": {
            "n_estimators": [100, 128, 256],
            "max_depth": [5, 6, 7],
            "learning_rate": [0.1, 0.3],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }
    }
    
    if boolean:
        grids["genetic_algorithm"] = {
            "pop_size": [300, 500],     
            "generations": [80],         
            "max_depth_init": [6, 10],    
            "max_depth_mut": [4, 6],      
            "cx_prob": [0.7],             
            "mut_prob": [0.25]            
        }
    
    grids["tabpfn"] = {}
    
    return grids

def get_base_models(seed: int, include_ga: bool = False, include_tabpfn: bool = False) -> Dict[str, Any]:
    models = {
        "svm": SVC(random_state=seed),
        "random_forest": RandomForestClassifier(random_state=seed),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "xgboost": XGBClassifier(
           booster='gbtree', 
           random_state=seed, 
           objective='binary:logistic', 
           verbosity=0,
           tree_method='hist',
           device='cpu'
        ),
    }
    
    if include_ga:
        models["genetic_algorithm"] = GeneticAlgorithmClassifier(random_state=seed)
    
    if include_tabpfn:
        models["tabpfn"] = TabPFNClassifier()
    
    return models


class BaselineRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ds = DatasetStore(cfg)
        self.best_decision_trees = {}
        self.best_decision_tree_accs = {}
        self.best_decision_tree_infos = {}
        self.best_ga_models = {}
        self.best_ga_accs = {}

    def run(self) -> List[Dict[str, Any]]:
        all_rows = []
        
        os.makedirs(os.path.dirname(self.cfg.out_jsonl) if os.path.dirname(self.cfg.out_jsonl) else ".", exist_ok=True)
        jsonl_file = open(self.cfg.out_jsonl, "w", encoding="utf-8")
        
        try:
            for fn in self.cfg.functions:
                if fn not in FUNCTION_NAME_MAPPING:
                    continue

                task_meta = EXPERIMENT_FUNCTION_METADATA.get(fn, {})
                current_lengths = task_meta.get("lengths", self.cfg.lengths)

                for L in current_lengths:
                    train_lines, val_lines, test_lines = self.ds.get(fn, L)
                    target_name = FUNCTION_NAME_MAPPING[fn]
                    
                    is_tabular = target_name in TABULAR_FNS
                    is_boolean = target_name in BOOLEAN_FNS
                    
                    base_models = get_base_models(self.cfg.seed, include_ga=is_boolean, include_tabpfn=True)
                    
                    if is_tabular:
                        parser = TabularDataParser(dataset_name=target_name)
                        X_train, y_train = parser.fit_transform(train_lines)
                        X_val, y_val = parser.transform(val_lines)
                        X_test, y_test = parser.transform(test_lines)
                    else:
                        X_train, y_train = parse_data(train_lines)
                        X_val, y_val = parse_data(val_lines)
                        X_test, y_test = parse_data(test_lines)

                    print(f"{fn} L={L}: train={len(train_lines)}, val={len(val_lines)}, test={len(test_lines)}")

                    param_grids = get_param_grids(tabular=is_tabular, boolean=is_boolean)

                    for model_name in base_models.keys():
                        print(f"  Starting {model_name} (T={self.cfg.num_trials} trials)...", flush=True)
                        try:
                            if model_name not in param_grids:
                                logger.warning(f"{fn} L={L} {model_name}: not found in param_grids, skipping")
                                continue
                            
                            test_accuracies = []
                            val_accuracies = []
                            durations = []
                            all_best_params = []
                            
                            for trial in range(self.cfg.num_trials):
                                print(f"    Trial {trial+1}/{self.cfg.num_trials}...", flush=True)
                                trial_seed = self.cfg.seed + trial
                                
                                base_model = get_base_models(trial_seed, include_ga=is_boolean, include_tabpfn=True)[model_name]
                                param_grid = param_grids[model_name]
                                
                                t0 = time.perf_counter()
                                
                                if model_name == "tabpfn":
                                    base_model.fit(X_train, y_train)
                                    val_pred = base_model.predict(X_val)
                                    test_pred = base_model.predict(X_test)
                                    best_val_acc = accuracy_score(y_val, val_pred)
                                    test_acc = accuracy_score(y_test, test_pred)
                                    best_params = {}
                                    best_model = base_model
                                elif model_name == "genetic_algorithm":
                                    if is_tabular:
                                        binarizer = GABinarizer()
                                        X_train_ga = binarizer.fit_transform(X_train, parser.numeric_indices, parser.categorical_indices)
                                        X_val_ga = binarizer.transform(X_val, parser.numeric_indices, parser.categorical_indices)
                                        X_test_ga = binarizer.transform(X_test, parser.numeric_indices, parser.categorical_indices)
                                    else:
                                        X_train_ga, X_val_ga, X_test_ga = X_train, X_val, X_test
                                    
                                    best_val_acc = -1
                                    best_params = None
                                    best_model = None
                                    
                                    from itertools import product
                                    keys = list(param_grid.keys())
                                    values = [param_grid[k] for k in keys]
                                    
                                    for combo in product(*values):
                                        params = dict(zip(keys, combo))
                                        params['random_state'] = trial_seed
                                        
                                        model = GeneticAlgorithmClassifier(**params)
                                        model.fit(X_train_ga, y_train, X_val_ga, y_val)
                                        
                                        val_pred = model.predict(X_val_ga)
                                        val_acc = accuracy_score(y_val, val_pred)
                                        
                                        if val_acc > best_val_acc:
                                            best_val_acc = val_acc
                                            best_params = params
                                            best_model = model
                                    
                                    test_acc = accuracy_score(y_test, best_model.predict(X_test_ga))
                                    
                                    # Track best GA model across trials (by test accuracy)
                                    task_key = (fn, L)
                                    current_best_acc = self.best_ga_accs.get(task_key, -1.0)
                                    if test_acc >= current_best_acc:
                                        self.best_ga_accs[task_key] = float(test_acc)
                                        # Clone the best node to preserve it
                                        if best_model.best_node_ is not None:
                                            cloned_node = best_model.best_node_.clone()
                                            self.best_ga_models[task_key] = {
                                                "node": cloned_node,
                                                "depth": cloned_node.depth(),
                                                "size": cloned_node.size(),
                                                "expr": cloned_node.to_str(),
                                                "test_acc": float(test_acc),
                                                "params": best_params.copy(),
                                            }
                                            print(f"        New best GA tree: {cloned_node.to_str()} (depth={cloned_node.depth()}, size={cloned_node.size()}, test_acc={test_acc:.4f})", flush=True)
                                else:
                                    grid_search = GridSearchCV(
                                        base_model,
                                        param_grid,
                                        cv=5,
                                        scoring='accuracy',
                                        n_jobs=1,
                                        verbose=0
                                    )
                                    grid_search.fit(X_train, y_train)
                                    best_model = grid_search.best_estimator_
                                    val_acc = accuracy_score(y_val, best_model.predict(X_val))
                                    
                                    if is_tabular and model_name == "svm":
                                        test_subset_size = min(1000, len(X_test))
                                        test_indices = np.random.RandomState(trial_seed).choice(len(X_test), test_subset_size, replace=False)
                                        test_acc = accuracy_score(y_test[test_indices], best_model.predict(X_test[test_indices]))
                                    else:
                                        test_acc = accuracy_score(y_test, best_model.predict(X_test))
                                    
                                    best_params = grid_search.best_params_
                                    best_val_acc = val_acc

                                    if model_name == "decision_tree":
                                        task_key = (fn, L)
                                        current_best_acc = self.best_decision_tree_accs.get(task_key, -1.0)
                                        if test_acc > current_best_acc:
                                            self.best_decision_tree_accs[task_key] = float(test_acc)
                                            self.best_decision_trees[task_key] = best_model
                                            self.best_decision_tree_infos[task_key] = {
                                                "fn": fn,
                                                "length": L,
                                                "trial": trial,
                                                "val_acc": float(best_val_acc),
                                                "test_acc": float(test_acc),
                                                "best_params": best_params,
                                            }
                                
                                duration_ms = int((time.perf_counter() - t0) * 1000)
                                test_accuracies.append(test_acc)
                                val_accuracies.append(best_val_acc)
                                durations.append(duration_ms)
                                all_best_params.append(best_params)
                                
                                print(f"      Trial {trial+1}: test_acc={test_acc:.4f}, val_acc={best_val_acc:.4f}", flush=True)
                            
                            # Compute statistics
                            test_acc_mean = float(np.mean(test_accuracies))
                            test_acc_std = float(np.std(test_accuracies))
                            val_acc_mean = float(np.mean(val_accuracies))
                            val_acc_std = float(np.std(val_accuracies))
                            total_duration_ms = int(np.sum(durations))
                            
                            # Use first best_params (could use mode if needed)
                            best_params = all_best_params[0]
                            
                            print(f"    {model_name} completed: test_acc={test_acc_mean:.4f}±{test_acc_std:.4f} (mean±std over {self.cfg.num_trials} trials)", flush=True)
                            
                            row = {
                                "fn": fn,
                                "length": L,
                                "model": model_name,
                                "duration_ms": total_duration_ms,
                                "val_acc": val_acc_mean,
                                "val_acc_std": val_acc_std,
                                "test_acc": test_acc_mean,
                                "test_acc_std": test_acc_std,
                                "best_params": json.dumps(best_params),
                                "best_cv_score": val_acc_mean,
                                "num_trials": self.cfg.num_trials,
                            }
                            
                            # Add tree depth info for GA models (depth of best-performing tree)
                            if model_name == "genetic_algorithm":
                                task_key = (fn, L)
                                best_ga_info = self.best_ga_models.get(task_key)
                                if best_ga_info is not None:
                                    row["tree_depth"] = float(best_ga_info["depth"])
                                    row["tree_size"] = float(best_ga_info["size"])
                                    row["tree_expr"] = best_ga_info["expr"]
                                    print(f"      Best tree: {best_ga_info['expr']}", flush=True)
                                    print(f"      Depth: {best_ga_info['depth']}, Size: {best_ga_info['size']}", flush=True)
                            all_rows.append(row)
                            jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            jsonl_file.flush()
                            logger.info(
                                f"{fn} L={L} {model_name}: test_acc={test_acc_mean:.4f}±{test_acc_std:.4f} "
                                f"(mean±std over {self.cfg.num_trials} trials)"
                            )
                        except Exception as e:
                            print(f"    Error processing {model_name}: {str(e)}", flush=True)
                            logger.error(f"{fn} L={L} {model_name}: error - {str(e)}", exc_info=True)
                            continue

            for (fn, length), tree in self.best_decision_trees.items():
                tree_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}.pkl")
                meta_path = os.path.join(current_dir, f"best_decision_tree_{fn}_L{length}_meta.json")
                with open(tree_path, "wb") as f:
                    pickle.dump(tree, f)
                meta = dict(self.best_decision_tree_infos[(fn, length)])
                meta["model_type"] = "decision_tree"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                logger.info(f"Best decision tree for {fn} L={length} saved to {tree_path} with metadata {meta_path}")
        finally:
            jsonl_file.close()

        return all_rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["fn", "length", "model", "duration_ms", "val_acc", "val_acc_std", "test_acc", "test_acc_std", "best_params", "best_cv_score", "num_trials"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline ML models runner")
    p.add_argument("--functions", nargs="*", help="Function IDs (e.g., fn_a fn_b ...)")
    p.add_argument("--lengths", nargs="*", type=int, help="Sequence lengths")
    p.add_argument("--train-size", type=int, help="Train size (default: 100)")
    p.add_argument("--val-size", type=int, help="Validation size (default: 100)")
    p.add_argument("--test-size", type=int, help="Test size (default: 10000)")
    p.add_argument("--seed", type=int, help="Global seed (default: 42)")
    p.add_argument("--num-trials", type=int, help="Number of trials for averaging (default: 10)")
    p.add_argument("--out-jsonl", help="Output JSONL path")
    p.add_argument("--out-csv", help="Output CSV path")

    args = p.parse_args()
    cfg = Config()

    if args.functions: cfg.functions = args.functions
    if args.lengths: cfg.lengths = args.lengths
    if args.train_size: cfg.train_size = args.train_size
    if args.val_size: cfg.val_size = args.val_size
    if args.test_size: cfg.test_size = args.test_size
    if args.seed is not None: cfg.seed = args.seed
    if args.num_trials is not None: cfg.num_trials = args.num_trials
    if args.out_jsonl: cfg.out_jsonl = args.out_jsonl
    if args.out_csv: cfg.out_csv = args.out_csv

    return cfg


def main():
    cfg = parse_args()
    runner = BaselineRunner(cfg)
    rows = runner.run()
    write_csv(cfg.out_csv, rows)
    logger.info(f"Results written to {cfg.out_jsonl} and {cfg.out_csv}")


if __name__ == "__main__":
    main()

