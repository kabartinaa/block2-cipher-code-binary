# feature_extractor.py

import os
import math
import json
import csv
from collections import Counter

import angr
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
import torch
from torch_geometric.data import Data


# ---------- BASIC UTILITIES ----------

def entropy(data: bytes) -> float:
    """Shannon entropy of raw bytes."""
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    ent = 0.0
    for f in freq:
        if f > 0:
            p = f / len(data)
            ent -= p * math.log2(p)
    return ent


def get_project(filepath: str):
    """
    ALWAYS load as a raw blob with x86 arch.
    This avoids all CLE backend/arch issues for synthetic .bin files.
    """
    print(f"[!] Forcing blob loader (x86) for {filepath}")
    proj = angr.Project(
        filepath,
        main_opts={
            "backend": "blob",
            "arch": "x86",
            "base_addr": 0x400000,
            "entry_point": 0x400000,
        },
        auto_load_libs=False,
    )
    return proj


# ---------- FEATURE EXTRACTION PER BASIC BLOCK ----------

def extract_block_features(block, md: Cs) -> list:
    """
    Node feature vector for one basic block:
      [block_size, block_entropy, mov, add, xor, sub, push, pop, cmp, call]
    """
    code = block.bytes
    ops = [ins.mnemonic for ins in md.disasm(code, block.addr)]
    hist = Counter(ops)

    return [
        block.size,
        entropy(code),
        hist["mov"],
        hist["add"],
        hist["xor"],
        hist["sub"],
        hist["push"],
        hist["pop"],
        hist["cmp"],
        hist["call"],
    ]


# ---------- BINARY -> CFG GRAPH ----------

def extract_graph_from_binary(
    filepath: str,
    algo_label_id: int,
    arch_str: str,            # kept only for logging
    out_dir: str = "graphs",
):
    """
    Build a PyTorch Geometric Data object from one binary:
      - nodes: basic blocks
      - edges: CFG edges
      - node features: simple opcode stats
      - label: algorithm ID
    """
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return None

    print(f"[+] Processing {filepath}  (algo_id={algo_label_id}, arch={arch_str})")

    proj = get_project(filepath)      # <--- always blob x86
    cfg = proj.analyses.CFGFast()

    # Disassembler: x86_64 generic (OK for synthetic blobs)
    md = Cs(CS_ARCH_X86, CS_MODE_64)

    node_features = []
    node_index = {}
    idx = 0
    edges = []

    # Nodes
    for node in cfg.graph.nodes():
        block = getattr(node, "block", None)
        if block is None:
            continue
        feats = extract_block_features(block, md)
        node_features.append(feats)
        node_index[node] = idx
        idx += 1

    if len(node_features) == 0:
        print(f"[!] No basic blocks found for {filepath}, skipping.")
        return None

    # Edges
    for src, dst in cfg.graph.edges():
        if src in node_index and dst in node_index:
            edges.append([node_index[src], node_index[dst]])

    if len(edges) == 0:
        print(f"[!] No edges found for {filepath}, skipping.")
        return None

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([algo_label_id], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(filepath) + ".pt")
    torch.save(data, out_path)
    print(f"[+] Saved graph to {out_path}")
    return out_path


# ---------- BATCH OVER ALL SAMPLES ----------

def batch_process(
    samples_dir: str,
    metadata_csv: str,
    out_dir: str = "graphs",
    label_map_path: str = "label_map.json",
):
    """
    Reads metadata.csv and creates .pt graphs for each sample.
    metadata.csv must have at least: file, algo, arch
    """
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"metadata.csv not found at: {metadata_csv}")

    rows = []
    with open(metadata_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build label map algo -> int
    label_map = {}
    next_label_id = 0
    for row in rows:
        algo = row.get("algo", "").strip()
        if not algo:
            continue
        if algo not in label_map:
            label_map[algo] = next_label_id
            next_label_id += 1

    print("[+] Algorithm label map:", label_map)
    with open(label_map_path, "w") as f:
        json.dump(label_map, f)
    print(f"[+] Saved label map to {label_map_path}")

    # Process each row
    for row in rows:
        fname = row.get("file")
        algo = row.get("algo", "").strip()
        arch = row.get("arch", "x86")
        if not fname or not algo:
            continue

        file_path = os.path.join(samples_dir, fname)
        algo_id = label_map[algo]
        extract_graph_from_binary(file_path, algo_id, arch, out_dir=out_dir)


if __name__ == "__main__":
    SAMPLES_DIR = "samples"       # folder with .bin files
    METADATA_CSV = "metadata.csv" # labels + arch per file

    batch_process(SAMPLES_DIR, METADATA_CSV)
