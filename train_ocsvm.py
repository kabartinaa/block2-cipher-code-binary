# train_ocsvm_standard_proprietary.py
#
# One-Class SVM training to detect proprietary algorithms as anomalies
# based on "standard vs proprietary" style features.
#
# Completely self-contained: generates synthetic data in memory,
# trains OC-SVM on STANDARD only, tests on mix of STANDARD + PROPRIETARY.


import random
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report


# -----------------------------
# 1. Feature definition
# -----------------------------

@dataclass
class AlgoFeatures:
    """
    One row = one binary/implementation (standard or proprietary).
    All fields are numeric; they simulate real features you might extract
    from code (function-level, entropy, structure, etc.).
    """

    avg_func_size: float          # average instructions per function
    max_func_size: int            # largest function size
    num_functions: int            # number of functions
    total_instructions: int       # total instructions

    avg_entropy: float            # average entropy of functions
    max_entropy: float            # max function entropy

    num_loops: int                # loops (approx. back-edges)
    num_branches: int             # conditional branches
    branch_density: float         # num_branches / total_instructions

    num_call_instructions: int    # number of function calls
    call_density: float           # num_call_instructions / total_instructions

    frac_arith_ops: float         # fraction of arithmetic ops
    frac_logic_ops: float         # fraction of logical/bitwise ops
    frac_mem_ops: float           # fraction of memory ops (load/store)


def describe_features():
    explanations = {
        "avg_func_size": "Average number of instructions per function.",
        "max_func_size": "Size of the largest function (instructions).",
        "num_functions": "Total number of functions in the binary.",
        "total_instructions": "Total instructions across all functions.",
        "avg_entropy": "Average entropy across functions (code randomness).",
        "max_entropy": "Max function entropy (most 'random' function).",
        "num_loops": "Total loops in the code.",
        "num_branches": "Total branch/conditional instructions.",
        "branch_density": "Branches divided by total instructions.",
        "num_call_instructions": "Total 'call' instructions.",
        "call_density": "Calls divided by total instructions.",
        "frac_arith_ops": "Fraction of arithmetic operations.",
        "frac_logic_ops": "Fraction of logical/bitwise operations.",
        "frac_mem_ops": "Fraction of memory operations (load/store).",
    }
    print("\n=== Feature Descriptions ===")
    for k, v in explanations.items():
        print(f"{k:20s} : {v}")
    print("=" * 35 + "\n")


# -----------------------------
# 2. Synthetic data generator
# -----------------------------

def generate_sample(is_standard: bool) -> AlgoFeatures:
    """
    Generate synthetic sample for:
      - standard algorithms: more consistent, typical distributions
      - proprietary algorithms: slightly different distributions (more 'odd')
    """

    if is_standard:
        # Standard algorithms: well-known, polished implementations
        num_functions = random.randint(10, 40)
        avg_func_size = random.uniform(30, 80)
        avg_entropy = random.uniform(4.5, 6.5)
        # opcode mix ~ balanced
        frac_mem = 0.3 + random.uniform(-0.05, 0.05)
        frac_arith = 0.4 + random.uniform(-0.05, 0.05)
        frac_logic = 0.3 + random.uniform(-0.05, 0.05)
        branch_factor = random.uniform(0.08, 0.18)
        loop_factor = random.uniform(0.05, 0.15)
        call_factor = random.uniform(0.03, 0.10)
    else:
        # Proprietary: more irregular, maybe obfuscated / unusual structure
        num_functions = random.randint(5, 60)
        avg_func_size = random.uniform(10, 120)
        avg_entropy = random.uniform(5.5, 8.0)  # on average higher
        # opcode mix shifted / noisier
        frac_mem = 0.4 + random.uniform(-0.15, 0.15)
        frac_arith = 0.25 + random.uniform(-0.10, 0.10)
        frac_logic = 0.35 + random.uniform(-0.10, 0.10)
        branch_factor = random.uniform(0.15, 0.35)
        loop_factor = random.uniform(0.10, 0.30)
        call_factor = random.uniform(0.01, 0.20)

    # normalize opcode fractions so they sum to 1
    s = max(frac_mem, 0) + max(frac_arith, 0) + max(frac_logic, 0)
    frac_mem = max(frac_mem, 0) / s
    frac_arith = max(frac_arith, 0) / s
    frac_logic = max(frac_logic, 0) / s

    total_instructions = int(num_functions * avg_func_size)
    total_instructions = max(total_instructions, 1)

    max_func_size = int(avg_func_size * random.uniform(1.1, 2.5))
    max_entropy = avg_entropy + random.uniform(0.0, 1.5)

    num_branches = int(total_instructions * branch_factor)
    num_loops = int(num_functions * loop_factor)
    num_calls = int(total_instructions * call_factor)

    branch_density = num_branches / total_instructions
    call_density = num_calls / total_instructions

    num_mem_ops = int(total_instructions * frac_mem)

    return AlgoFeatures(
        avg_func_size=avg_func_size,
        max_func_size=max_func_size,
        num_functions=num_functions,
        total_instructions=total_instructions,
        avg_entropy=avg_entropy,
        max_entropy=max_entropy,
        num_loops=num_loops,
        num_branches=num_branches,
        branch_density=branch_density,
        num_call_instructions=num_calls,
        call_density=call_density,
        frac_arith_ops=frac_arith,
        frac_logic_ops=frac_logic,
        frac_mem_ops=frac_mem,
    )


def build_datasets(
    n_standard_train: int = 400,
    n_standard_test: int = 100,
    n_proprietary_test: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Build:
      - X_train: only STANDARD samples (for OC-SVM training)
      - X_test: STANDARD + PROPRIETARY
      - y_test: +1 for standard, -1 for proprietary (for evaluation)
    """
    random.seed(seed)

    # training: only standard
    train_samples: List[Dict] = []
    for _ in range(n_standard_train):
        feat = generate_sample(is_standard=True)
        train_samples.append(asdict(feat))

    # test: mix of standard + proprietary
    test_samples: List[Dict] = []
    test_labels: List[int] = []

    for _ in range(n_standard_test):
        feat = generate_sample(is_standard=True)
        test_samples.append(asdict(feat))
        test_labels.append(+1)  # standard → +1

    for _ in range(n_proprietary_test):
        feat = generate_sample(is_standard=False)
        test_samples.append(asdict(feat))
        test_labels.append(-1)  # proprietary → -1

    feature_names = list(train_samples[0].keys())

    X_train = np.array([[s[f] for f in feature_names] for s in train_samples], dtype=float)
    X_test = np.array([[s[f] for f in feature_names] for s in test_samples], dtype=float)
    y_test = np.array(test_labels, dtype=int)

    return X_train, X_test, y_test, feature_names


# -----------------------------
# 3. Train OC-SVM and evaluate
# -----------------------------

def train_ocsvm(X_train: np.ndarray) -> OneClassSVM:
    """
    Train One-Class SVM on standard-only data.
    """
    # nu = approx fraction of outliers you expect in training (we set small)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    print("[+] Training One-Class SVM on STANDARD samples only...")
    model.fit(X_train)
    return model


def evaluate_ocsvm(model: OneClassSVM, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate OCSVM on test set containing:
      +1 = standard (should be predicted as inliers)
      -1 = proprietary (should be predicted as outliers)
    """
    # OneClassSVM predicts +1 (inlier) or -1 (outlier)
    y_pred = model.predict(X_test)

    print("\n[+] Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred, labels=[+1, -1])
    print("Labels: [ +1 (standard), -1 (proprietary) ]")
    print(cm)

    print("\n[+] Classification Report (treat +1 as standard, -1 as proprietary):")
    print(classification_report(y_test, y_pred, target_names=["proprietary(-1)", "standard(+1)"]))


# -----------------------------
# 4. Main
# -----------------------------

if __name__ == "__main__":
    # 1) Explain features
    describe_features()

    # 2) Build datasets
    X_train, X_test, y_test, feat_names = build_datasets(
        n_standard_train=400,
        n_standard_test=100,
        n_proprietary_test=100,
        seed=42,
    )

    print(f"[+] X_train shape: {X_train.shape} (STANDARD only)")
    print(f"[+] X_test shape:  {X_test.shape} (STANDARD + PROPRIETARY)")
    print(f"[+] Feature names: {feat_names}\n")

    # 3) Train OC-SVM
    model = train_ocsvm(X_train)

    # 4) Evaluate
    evaluate_ocsvm(model, X_test, y_test)

    print("\n[+] Done.")
