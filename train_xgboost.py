# train_xgboost_functions.py
#
# XGBoost training on function-based features.
# Self-contained: generates synthetic samples, explains each feature,
# trains classifier, prints accuracy and feature importance.

import random
import math
from dataclasses import dataclass, asdict
from typing import List, Dict
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# -----------------------------
# 1. Function-based feature model
# -----------------------------

@dataclass
class FunctionBasedFeatures:
    """
    One row = one binary / sample.
    All fields are numeric and will be fed into XGBoost.
    """

    num_functions: int                # total functions discovered
    avg_func_size: float              # average instructions per function
    max_func_size: int                # size of the largest function
    num_call_instructions: int        # total 'call'-like instructions
    call_density: float               # num_call_instructions / total_instructions
    total_instructions: int           # total instructions in the binary

    avg_func_entropy: float           # average entropy across all functions
    max_func_entropy: float           # highest entropy among functions

    num_loops: int                    # approximate loop count (e.g., from back-edges)
    loop_density: float               # num_loops / num_functions

    num_branches: int                 # conditional branches (if/else)
    branch_density: float             # num_branches / total_instructions

    num_memory_ops: int               # load/store-like instructions
    mem_op_ratio: float               # num_memory_ops / total_instructions

    # opcode histogram style features (very simplified):
    frac_arith_ops: float             # fraction of arithmetic ops (add, sub, mul, xor, etc.)
    frac_logic_ops: float             # fraction of logic ops (and, or, not, shifts)
    frac_data_move_ops: float         # fraction of mov/load/store (data movement)


def describe_features():
    """
    Print human-readable explanation of each feature.
    Call this once at the beginning so you know what the model sees.
    """
    explanations = {
        "num_functions": "Total number of functions detected in the binary.",
        "avg_func_size": "Average number of instructions per function.",
        "max_func_size": "Size (instructions) of the largest function.",
        "num_call_instructions": "Total 'call' instructions across all functions.",
        "call_density": "Call instructions divided by total instructions.",
        "total_instructions": "Total number of instructions in the binary.",
        "avg_func_entropy": "Average entropy of all functions (code randomness).",
        "max_func_entropy": "Maximum entropy over all functions.",
        "num_loops": "Total loops (e.g., from back-edges in CFG).",
        "loop_density": "Loops per function.",
        "num_branches": "Number of branch/conditional jump instructions.",
        "branch_density": "Branches divided by total instructions.",
        "num_memory_ops": "Number of memory-related ops (load/store).",
        "mem_op_ratio": "Memory-related ops divided by total instructions.",
        "frac_arith_ops": "Fraction of arithmetic opcodes among all instructions.",
        "frac_logic_ops": "Fraction of logic/bitwise opcodes.",
        "frac_data_move_ops": "Fraction of data-move operations (mov, load, store).",
    }
    print("\n=== Feature Descriptions ===")
    for name, desc in explanations.items():
        print(f"{name:22s} : {desc}")
    print("============================\n")


# -----------------------------
# 2. Synthetic dataset generator
# -----------------------------

def generate_synthetic_sample(label: int) -> FunctionBasedFeatures:
    """
    Generate one synthetic sample conditioned on label.
    Different labels simulate different 'algorithms' with different function patterns.
    For example:
      label 0: many small functions, low entropy  (like simple hash)
      label 1: fewer big functions, higher entropy (like block cipher)
      label 2: more branches, loops (like complex protocol)
      label 3: memory-heavy, many loads/stores (like parsing code)
    """
    # Base distributions
    if label == 0:
        # many small functions
        num_functions = random.randint(20, 60)
        avg_func_size = random.uniform(10, 30)
        avg_entropy = random.uniform(3.0, 5.0)
        mem_bias = 0.2
        arith_bias = 0.5
        logic_bias = 0.3
    elif label == 1:
        # fewer but larger crypto-style functions
        num_functions = random.randint(5, 25)
        avg_func_size = random.uniform(40, 120)
        avg_entropy = random.uniform(5.0, 7.5)
        mem_bias = 0.3
        arith_bias = 0.6
        logic_bias = 0.1
    elif label == 2:
        # control-flow heavy: many branches & loops
        num_functions = random.randint(10, 40)
        avg_func_size = random.uniform(30, 80)
        avg_entropy = random.uniform(4.0, 6.0)
        mem_bias = 0.25
        arith_bias = 0.3
        logic_bias = 0.45
    else:
        # memory-heavy routines
        num_functions = random.randint(8, 30)
        avg_func_size = random.uniform(25, 70)
        avg_entropy = random.uniform(3.5, 5.5)
        mem_bias = 0.5
        arith_bias = 0.25
        logic_bias = 0.25

    total_instructions = int(num_functions * avg_func_size)

    max_func_size = int(avg_func_size * random.uniform(1.1, 2.5))

    # opcode mix
    frac_mem = mem_bias + random.uniform(-0.05, 0.05)
    frac_arith = arith_bias + random.uniform(-0.05, 0.05)
    frac_logic = logic_bias + random.uniform(-0.05, 0.05)

    # normalize to sum to 1 (and clamp)
    s = max(frac_mem, 0) + max(frac_arith, 0) + max(frac_logic, 0)
    frac_mem = max(frac_mem, 0) / s
    frac_arith = max(frac_arith, 0) / s
    frac_logic = max(frac_logic, 0) / s

    num_memory_ops = int(total_instructions * frac_mem)
    num_arith_ops = int(total_instructions * frac_arith)
    num_logic_ops = int(total_instructions * frac_logic)

    # calls, branches, loops
    # tie them loosely to label type
    if label == 1:
        num_call_instructions = random.randint(5, 30)
    else:
        num_call_instructions = random.randint(10, 80)

    if label == 2:
        num_branches = int(total_instructions * random.uniform(0.15, 0.3))
        num_loops = random.randint(10, 40)
    else:
        num_branches = int(total_instructions * random.uniform(0.05, 0.15))
        num_loops = random.randint(3, 15)

    call_density = num_call_instructions / max(total_instructions, 1)
    branch_density = num_branches / max(total_instructions, 1)
    mem_op_ratio = num_memory_ops / max(total_instructions, 1)
    loop_density = num_loops / max(num_functions, 1)

    max_entropy = avg_entropy + random.uniform(0.0, 1.5)

    return FunctionBasedFeatures(
        num_functions=num_functions,
        avg_func_size=avg_func_size,
        max_func_size=max_func_size,
        num_call_instructions=num_call_instructions,
        call_density=call_density,
        total_instructions=total_instructions,
        avg_func_entropy=avg_entropy,
        max_func_entropy=max_entropy,
        num_loops=num_loops,
        loop_density=loop_density,
        num_branches=num_branches,
        branch_density=branch_density,
        num_memory_ops=num_memory_ops,
        mem_op_ratio=mem_op_ratio,
        frac_arith_ops=frac_arith,
        frac_logic_ops=frac_logic,
        frac_data_move_ops=frac_mem,
    )


def build_dataset(
    num_samples: int = 1000,
    num_classes: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a synthetic dataset of function-based features.
    Returns:
      X: [N, D] features
      y: [N] labels
      feature_names: list of column names
    """
    random.seed(seed)

    samples: List[Dict] = []
    labels: List[int] = []

    for _ in range(num_samples):
        label = random.randint(0, num_classes - 1)
        feat = generate_synthetic_sample(label)
        samples.append(asdict(feat))
        labels.append(label)

    feature_names = list(samples[0].keys())
    X = np.array([[s[name] for name in feature_names] for s in samples], dtype=float)
    y = np.array(labels, dtype=int)

    return X, y, feature_names


# -----------------------------
# 3. Train XGBoost
# -----------------------------

def train_xgboost(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """
    Train XGBoost on function-based features, print accuracy and feature importances.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(set(y)),
        eval_metric="mlogloss",
        tree_method="hist",  # good default for CPU
    )

    print("[+] Training XGBoost...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Test Accuracy: {acc*100:.2f}%\n")

    print("[+] Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n[+] Feature Importances (sorted):")
    for name, imp in feat_imp:
        print(f"{name:22s} : {imp:.4f}")

    return model


# -----------------------------
# 4. Main
# -----------------------------

if __name__ == "__main__":
    # 1) Explain each feature in human language
    describe_features()

    # 2) Build synthetic dataset
    X, y, feat_names = build_dataset(num_samples=1200, num_classes=4, seed=42)
    print(f"[+] Built dataset: X shape = {X.shape}, y shape = {y.shape}")
    print(f"[+] Feature names: {feat_names}\n")

    # 3) Train and evaluate XGBoost
    model = train_xgboost(X, y, feat_names)
    print("\n[+] Training complete.")
