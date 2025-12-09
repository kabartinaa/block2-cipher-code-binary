import os
import sys
import math
import json
import random
import hashlib
from datetime import datetime
import tempfile

import numpy as np
import joblib

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

# ------- NEW: Flask for single website -------
from flask import Flask, request, render_template_string

# =========================================================
# 0. GLOBAL LABEL SPACES
# =========================================================

ALGO_ID_TO_NAME = {
    0: "AES-CBC",
    1: "AES-CTR",
    2: "AES-GCM",
    3: "AES-CCM",
    4: "AES-XTS",
    5: "3DES-CBC",
    6: "3DES-ECB",
    7: "CHACHA20",
    8: "CHACHA20-POLY1305",
    9: "BLOWFISH-CBC",
    10: "SERPENT-CBC",
    11: "TWOFISH-CBC",
    12: "CAMELLIA-CBC",
    13: "RSA-2048",
    14: "RSA-4096",
    15: "ECDSA-P256",
    16: "ECDSA-P384",
    17: "ECDH-P256",
    18: "HMAC-SHA1",
    19: "HMAC-SHA256",
    20: "SHA1",
    21: "SHA256",
    22: "SHA3-256",
}

ARCH_ID_TO_NAME = {
    0: "x86",
    1: "x86_64",
    2: "arm",
    3: "armv7",
    4: "aarch64",
    5: "mips",
    6: "mips64",
    7: "riscv32",
    8: "riscv64",
    9: "avr",
    10: "powerpc",
}

PROTOCOL_ID_TO_NAME = {
    0: "TLS",
    1: "DTLS",
    2: "SSH",
    3: "WiFi",
    4: "Bluetooth",
    5: "IPSec",
    6: "HTTP",
    7: "MQTT",
    8: "CoAP",
    9: "CustomProto",
}

PROTOCOL_SEQUENCE_JSON = "protocol_sequences.json"
VOCAB_SIZE = 50  # for LSTM "opcode" IDs (demo)

BLOCKCHAIN_JSON = "blockchain.json"
MODEL_METRICS_JSON = "model_metrics.json"  # metrics file


# =========================================================
# 0.1 SIMPLE BLOCKCHAIN
# =========================================================

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # must be JSON-serializable
        self.previous_hash = previous_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(
            {
                "index": self.index,
                "timestamp": self.timestamp,
                "data": self.data,
                "previous_hash": self.previous_hash,
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(block_string).hexdigest()


class Blockchain:
    def __init__(self, filename=BLOCKCHAIN_JSON):
        self.filename = filename
        self.chain = []
        self.load_chain()

    def create_genesis_block(self):
        genesis_data = {"message": "Genesis Block"}
        genesis_block = Block(
            index=0,
            timestamp=datetime.utcnow().isoformat() + "Z",
            data=genesis_data,
            previous_hash="0",
        )
        self.chain = [genesis_block]
        self.save_chain()

    def load_chain(self):
        if not os.path.exists(self.filename):
            self.create_genesis_block()
            return

        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
            self.chain = []
            for b in data:
                block = Block(
                    index=b["index"],
                    timestamp=b["timestamp"],
                    data=b["data"],
                    previous_hash=b["previous_hash"],
                )
                block.hash = b["hash"]
                self.chain.append(block)
            if not self.chain:
                self.create_genesis_block()
        except Exception:
            self.create_genesis_block()

    def save_chain(self):
        data = []
        for b in self.chain:
            data.append(
                {
                    "index": b.index,
                    "timestamp": b.timestamp,
                    "data": b.data,
                    "previous_hash": b.previous_hash,
                    "hash": b.hash,
                }
            )
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

    def add_block(self, data):
        if not self.chain:
            self.create_genesis_block()
        last_block = self.chain[-1]
        new_block = Block(
            index=last_block.index + 1,
            timestamp=datetime.utcnow().isoformat() + "Z",
            data=data,
            previous_hash=last_block.hash,
        )
        self.chain.append(new_block)
        self.save_chain()
        return new_block


# =========================================================
# 1. MODELS
# =========================================================

class GNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        g = global_mean_pool(h, batch)
        out = self.fc(g)
        return out


class OpcodeLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, extra_feat_dim, num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim + extra_feat_dim, num_classes)

    def forward(self, seqs, lengths, extra_feats):
        emb = self.embedding(seqs)
        packed = pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        seq_emb = h_n[0]
        combined = torch.cat([seq_emb, extra_feats], dim=1)
        logits = self.fc(combined)
        return logits


# =========================================================
# 2. FEATURE EXTRACTION
# =========================================================

def shannon_entropy(byte_arr: bytes) -> float:
    if len(byte_arr) == 0:
        return 0.0
    freq = [0] * 256
    for b in byte_arr:
        freq[b] += 1
    ent = 0.0
    for f in freq:
        if f > 0:
            p = f / len(byte_arr)
            ent -= p * math.log2(p)
    return ent


def build_features_from_bytes(data: bytes):
    if len(data) == 0:
        data = b"\x00"

    # ---- LSTM sequence ----
    opcode_ids = [b % VOCAB_SIZE for b in data]
    seq = torch.tensor(opcode_ids, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(opcode_ids)], dtype=torch.long)

    # extra features (normalized)
    f_len = len(data)
    f_ent = shannon_entropy(data)
    f_mean = float(sum(data)) / len(data)
    f_std = float(np.std(np.frombuffer(data, dtype=np.uint8)))

    extra_feats = torch.tensor(
        [[f_len / 10000.0, f_ent / 8.0, f_mean / 255.0]],
        dtype=torch.float,
    )

    # ---- GNN graph ----
    block_size = 16
    num_blocks = max(1, len(data) // block_size)
    node_features = []
    for i in range(num_blocks):
        block = data[i * block_size: (i + 1) * block_size]
        if len(block) == 0:
            block = b"\x00"
        mean_b = float(sum(block)) / len(block)
        ent_b = shannon_entropy(block)
        node_features.append([mean_b / 255.0, ent_b / 8.0])

    x = torch.tensor(node_features, dtype=torch.float)
    edges = []
    for i in range(num_blocks - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    if not edges:
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    batch_vec = torch.zeros(x.size(0), dtype=torch.long)
    gnn_data = Data(x=x, edge_index=edge_index, batch=batch_vec)

    # ---- Tabular features ----
    unique_bytes = len(set(data))
    tabular = np.array(
        [[
            f_len,
            unique_bytes,
            f_ent,
            f_mean,
            f_std,
        ]],
        dtype=float,
    )

    feature_dict = {
        "length": f_len,
        "unique_bytes": unique_bytes,
        "entropy": f_ent,
        "mean_byte": f_mean,
        "std_byte": f_std,
    }

    lstm_input = (seq, lengths, extra_feats)
    return gnn_data, lstm_input, tabular, feature_dict


def build_features_from_file(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return build_features_from_bytes(data)


# =========================================================
# 3. SYNTHETIC DATASET FOR TRAINING
# =========================================================

def synthetic_sample(algo_id: int):
    random.seed()
    base_len = 500 + 50 * algo_id
    length = random.randint(base_len, base_len + 200)
    center = (algo_id * 11) % 256
    data = bytes(
        max(0, min(255, int(random.gauss(center, 25))))
        for _ in range(length)
    )
    return data


def build_synthetic_dataset(n_samples=150):
    """
    Smaller n_samples for faster training.
    """
    num_algos = len(ALGO_ID_TO_NAME)
    num_archs = len(ARCH_ID_TO_NAME)
    num_protos = len(PROTOCOL_ID_TO_NAME)

    # store for GNN/LSTM training
    gnn_graphs = []
    lstm_seqs = []
    lstm_lengths = []
    lstm_extras = []
    algo_labels = []

    # store for XGB/OC-SVM
    tabular_list = []
    algo_labels_tab = []
    arch_labels_tab = []
    proto_labels_tab = []

    per_algo_features = {aid: [] for aid in ALGO_ID_TO_NAME}

    proto_sequences = {}
    for proto_id, proto_name in PROTOCOL_ID_TO_NAME.items():
        proto_sequences[proto_name] = [
            f"{proto_name}_STEP1",
            f"{proto_name}_STEP2",
            f"{proto_name}_STEP3",
        ]

    for _ in range(n_samples):
        algo_id = random.randint(0, num_algos - 1)
        arch_id = algo_id % num_archs
        proto_id = algo_id % num_protos

        data = synthetic_sample(algo_id)
        gnn_data, lstm_input, tabular, _ = build_features_from_bytes(data)
        seq, lengths, extra_feats = lstm_input

        gnn_data.y = torch.tensor([algo_id], dtype=torch.long)
        gnn_graphs.append(gnn_data)

        lstm_seqs.append(seq.squeeze(0))
        lstm_lengths.append(lengths.squeeze(0))
        lstm_extras.append(extra_feats.squeeze(0))
        algo_labels.append(algo_id)

        tabular_list.append(tabular[0])
        algo_labels_tab.append(algo_id)
        arch_labels_tab.append(arch_id)
        proto_labels_tab.append(proto_id)

        per_algo_features[algo_id].append(list(tabular[0]))

    algo_feature_means = {}
    for aid, rows in per_algo_features.items():
        if rows:
            algo_feature_means[aid] = list(np.mean(np.array(rows), axis=0))
        else:
            algo_feature_means[aid] = None

    with open(PROTOCOL_SEQUENCE_JSON, "w") as f:
        json.dump(proto_sequences, f, indent=4)

    dataset = {
        "gnn_graphs": gnn_graphs,
        "lstm_seqs": lstm_seqs,
        "lstm_lengths": lstm_lengths,
        "lstm_extras": lstm_extras,
        "algo_labels": algo_labels,
        "tabular": np.array(tabular_list, dtype=float),
        "algo_labels_tab": np.array(algo_labels_tab, dtype=int),
        "arch_labels_tab": np.array(arch_labels_tab, dtype=int),
        "proto_labels_tab": np.array(proto_labels_tab, dtype=int),
        "algo_feature_means": algo_feature_means,
    }
    return dataset


# =========================================================
# 4. TRAIN ALL MODELS + METRICS
# =========================================================

def compute_and_save_metrics(dataset, device, gnn_model, lstm_model, xgb_algo, xgb_arch, xgb_proto, ocsvm):
    """
    Compute accuracy, precision, recall, and false positive rate
    for each model, plus overall ensemble metrics and OC-SVM stats.
    """
    tabular = dataset["tabular"]
    y_algo = dataset["algo_labels_tab"]
    y_arch = dataset["arch_labels_tab"]
    y_proto = dataset["proto_labels_tab"]

    # ---------- GNN metrics (algo) ----------
    gnn_graphs = dataset["gnn_graphs"]
    gnn_model.eval()
    gnn_true = []
    gnn_pred = []
    loader = DataLoader(gnn_graphs, batch_size=32, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = gnn_model(batch.x, batch.edge_index, batch.batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            gnn_pred.extend(preds.tolist())
            gnn_true.extend(labels.tolist())
    gnn_true = np.array(gnn_true)
    gnn_pred = np.array(gnn_pred)
    gnn_acc = float(accuracy_score(gnn_true, gnn_pred))
    gnn_recall = float(recall_score(gnn_true, gnn_pred, average="macro"))
    gnn_prec = float(precision_score(gnn_true, gnn_pred, average="macro", zero_division=0))
    gnn_fpr_overall = float(1.0 - gnn_acc)  # treat overall error as FP-ish rate

    # ---------- LSTM metrics (algo) ----------
    lstm_model.eval()
    lstm_seqs = dataset["lstm_seqs"]
    lstm_lengths = dataset["lstm_lengths"]
    lstm_extras = dataset["lstm_extras"]
    seqs_padded = pad_sequence(lstm_seqs, batch_first=True, padding_value=0)
    lengths_t = torch.stack(lstm_lengths)
    extras_t = torch.stack(lstm_extras)
    labels_t = torch.tensor(dataset["algo_labels"], dtype=torch.long)

    with torch.no_grad():
        logits = lstm_model(
            seqs_padded.to(device),
            lengths_t.to(device),
            extras_t.to(device),
        )
        lstm_preds = torch.argmax(logits, dim=1).cpu().numpy()
        lstm_true = labels_t.cpu().numpy()
    lstm_acc = float(accuracy_score(lstm_true, lstm_preds))
    lstm_recall = float(recall_score(lstm_true, lstm_preds, average="macro"))
    lstm_prec = float(precision_score(lstm_true, lstm_preds, average="macro", zero_division=0))
    lstm_fpr_overall = float(1.0 - lstm_acc)

    # ---------- XGBoost metrics ----------
    y_algo_pred = xgb_algo.predict(tabular)
    y_arch_pred = xgb_arch.predict(tabular)
    y_proto_pred = xgb_proto.predict(tabular)

    xgb_algo_acc = float(accuracy_score(y_algo, y_algo_pred))
    xgb_algo_recall = float(recall_score(y_algo, y_algo_pred, average="macro"))
    xgb_algo_prec = float(precision_score(y_algo, y_algo_pred, average="macro", zero_division=0))
    xgb_algo_fpr_overall = float(1.0 - xgb_algo_acc)

    xgb_arch_acc = float(accuracy_score(y_arch, y_arch_pred))
    xgb_arch_recall = float(recall_score(y_arch, y_arch_pred, average="macro"))
    xgb_arch_prec = float(precision_score(y_arch, y_arch_pred, average="macro", zero_division=0))
    xgb_arch_fpr_overall = float(1.0 - xgb_arch_acc)

    xgb_proto_acc = float(accuracy_score(y_proto, y_proto_pred))
    xgb_proto_recall = float(recall_score(y_proto, y_proto_pred, average="macro"))
    xgb_proto_prec = float(precision_score(y_proto, y_proto_pred, average="macro", zero_division=0))
    xgb_proto_fpr_overall = float(1.0 - xgb_proto_acc)

    # ---------- OVERALL ENSEMBLE ----------
    N = len(y_algo)
    ensemble_preds = []
    for i in range(N):
        votes = [int(gnn_pred[i]), int(lstm_preds[i]), int(y_algo_pred[i])]
        final = max(set(votes), key=votes.count)
        ensemble_preds.append(final)
    ensemble_preds = np.array(ensemble_preds)
    ensemble_true = y_algo

    ensemble_acc = float(accuracy_score(ensemble_true, ensemble_preds))
    ensemble_recall = float(recall_score(ensemble_true, ensemble_preds, average="macro"))
    ensemble_prec = float(precision_score(ensemble_true, ensemble_preds, average="macro", zero_division=0))
    ensemble_fpr_overall = float(1.0 - ensemble_acc)

    # ---------- OC-SVM ----------
    oc_pred = ocsvm.predict(tabular)  # +1 normal, -1 outlier
    num_algos = len(ALGO_ID_TO_NAME)
    standard_mask = y_algo < (num_algos // 2)
    nonstandard_mask = ~standard_mask

    fp = int(np.sum((standard_mask) & (oc_pred == -1)))
    tp_normal = int(np.sum((standard_mask) & (oc_pred == 1)))
    n_standard = int(np.sum(standard_mask))
    fp_rate = float(fp / n_standard) if n_standard > 0 else 0.0

    fn = int(np.sum((nonstandard_mask) & (oc_pred == 1)))
    tp_outlier = int(np.sum((nonstandard_mask) & (oc_pred == -1)))
    n_nonstandard = int(np.sum(nonstandard_mask))
    fn_rate = float(fn / n_nonstandard) if n_nonstandard > 0 else 0.0

    total = n_standard + n_nonstandard
    oc_acc = float((tp_normal + tp_outlier) / total) if total > 0 else 0.0
    prec_denom = tp_outlier + fp
    rec_denom = tp_outlier + fn
    oc_prec = float(tp_outlier / prec_denom) if prec_denom > 0 else 0.0
    oc_rec = float(tp_outlier / rec_denom) if rec_denom > 0 else 0.0

    metrics = {
        "gnn_algo": {
            "accuracy": gnn_acc,
            "macro_recall": gnn_recall,
            "macro_precision": gnn_prec,
            "false_positive_rate_overall": gnn_fpr_overall,
        },
        "lstm_algo": {
            "accuracy": lstm_acc,
            "macro_recall": lstm_recall,
            "macro_precision": lstm_prec,
            "false_positive_rate_overall": lstm_fpr_overall,
        },
        "xgb_algo": {
            "accuracy": xgb_algo_acc,
            "macro_recall": xgb_algo_recall,
            "macro_precision": xgb_algo_prec,
            "false_positive_rate_overall": xgb_algo_fpr_overall,
        },
        "xgb_arch": {
            "accuracy": xgb_arch_acc,
            "macro_recall": xgb_arch_recall,
            "macro_precision": xgb_arch_prec,
            "false_positive_rate_overall": xgb_arch_fpr_overall,
        },
        "xgb_proto": {
            "accuracy": xgb_proto_acc,
            "macro_recall": xgb_proto_recall,
            "macro_precision": xgb_proto_prec,
            "false_positive_rate_overall": xgb_proto_fpr_overall,
        },
        "overall_ensemble": {
            "accuracy": ensemble_acc,
            "macro_recall": ensemble_recall,
            "macro_precision": ensemble_prec,
            "false_positive_rate_overall": ensemble_fpr_overall,
        },
        "ocsvm": {
            "accuracy": oc_acc,
            "precision": oc_prec,
            "recall": oc_rec,
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "n_standard": n_standard,
            "n_nonstandard": n_nonstandard,
            "false_positives": fp,
            "false_negatives": fn,
            "true_standard_normal": tp_normal,
            "true_nonstandard_outlier": tp_outlier,
        },
    }

    with open(MODEL_METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=4)

    print("[+] Saved model_metrics.json")


def load_model_metrics(path=MODEL_METRICS_JSON):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def train_all_models():
    print("[+] Building synthetic training dataset...")
    dataset = build_synthetic_dataset(n_samples=150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- GNN ----------
    print("[+] Training GNN...")
    gnn_graphs = dataset["gnn_graphs"]
    gnn_in_dim = gnn_graphs[0].x.shape[1]
    gnn_num_classes = len(ALGO_ID_TO_NAME)
    gnn_model = GNNClassifier(gnn_in_dim, hidden_dim=32, num_classes=gnn_num_classes).to(device)

    loader = DataLoader(gnn_graphs, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(gnn_model.parameters(), lr=0.001)

    for epoch in range(1, 4):  # 3 epochs
        gnn_model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = gnn_model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        print(f"[GNN] Epoch {epoch:02d}, loss = {total_loss / len(gnn_graphs):.4f}")

    torch.save(gnn_model.state_dict(), "gnn_model.pth")
    print("[+] Saved gnn_model.pth")

    # ---------- LSTM ----------
    print("[+] Training LSTM...")
    lstm_seqs = dataset["lstm_seqs"]
    lstm_lengths = dataset["lstm_lengths"]
    lstm_extras = dataset["lstm_extras"]
    algo_labels = dataset["algo_labels"]

    seqs_padded = pad_sequence(lstm_seqs, batch_first=True, padding_value=0)
    lengths_t = torch.stack(lstm_lengths)
    extras_t = torch.stack(lstm_extras)
    labels_t = torch.tensor(algo_labels, dtype=torch.long)

    lstm_num_classes = len(ALGO_ID_TO_NAME)
    lstm_model = OpcodeLSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=16,
        hidden_dim=32,
        extra_feat_dim=3,
        num_classes=lstm_num_classes,
    ).to(device)

    opt_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    N = seqs_padded.shape[0]
    idxs = list(range(N))
    for epoch in range(1, 4):  # 3 epochs
        random.shuffle(idxs)
        total_loss = 0.0
        lstm_model.train()
        for i in range(0, N, 32):
            batch_idx = idxs[i: i + 32]
            b_seqs = seqs_padded[batch_idx].to(device)
            b_lens = lengths_t[batch_idx].to(device)
            b_extras = extras_t[batch_idx].to(device)
            b_labels = labels_t[batch_idx].to(device)

            opt_lstm.zero_grad()
            logits = lstm_model(b_seqs, b_lens, b_extras)
            loss = F.cross_entropy(logits, b_labels)
            loss.backward()
            opt_lstm.step()
            total_loss += loss.item() * b_labels.size(0)
        print(f"[LSTM] Epoch {epoch:02d}, loss = {total_loss / N:.4f}")

    torch.save(lstm_model.state_dict(), "lstm_opcode_model.pth")
    print("[+] Saved lstm_opcode_model.pth")

    # ---------- XGBoost ----------
    tabular = dataset["tabular"]
    y_algo = dataset["algo_labels_tab"]
    y_arch = dataset["arch_labels_tab"]
    y_proto = dataset["proto_labels_tab"]

    print("[+] Training XGBoost (algo)...")
    xgb_algo = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(ALGO_ID_TO_NAME),
        eval_metric="mlogloss",
        tree_method="hist",
    )
    xgb_algo.fit(tabular, y_algo)
    xgb_algo.save_model("xgb_algo_model.json")
    print("[+] Saved xgb_algo_model.json")

    print("[+] Training XGBoost (arch)...")
    xgb_arch = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(ARCH_ID_TO_NAME),
        eval_metric="mlogloss",
        tree_method="hist",
    )
    xgb_arch.fit(tabular, y_arch)
    xgb_arch.save_model("xgb_arch_model.json")
    print("[+] Saved xgb_arch_model.json")

    print("[+] Training XGBoost (proto)...")
    xgb_proto = XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=len(PROTOCOL_ID_TO_NAME),
        eval_metric="mlogloss",
        tree_method="hist",
    )
    xgb_proto.fit(tabular, y_proto)
    xgb_proto.save_model("xgb_proto_model.json")
    print("[+] Saved xgb_proto_model.json")

    # ---------- OC-SVM ----------
    print("[+] Training OC-SVM...")
    standard_mask = y_algo < (len(ALGO_ID_TO_NAME) // 2)
    X_standard = tabular[standard_mask]

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    ocsvm.fit(X_standard)
    joblib.dump(ocsvm, "ocsvm_model.pkl")
    print("[+] Saved ocsvm_model.pkl")

    with open("algo_feature_means.json", "w") as f:
        json.dump(dataset["algo_feature_means"], f, indent=4)
    print("[+] Saved algo_feature_means.json")

    # ---------- METRICS ----------
    print("[+] Computing metrics on synthetic dataset...")
    compute_and_save_metrics(
        dataset=dataset,
        device=device,
        gnn_model=gnn_model,
        lstm_model=lstm_model,
        xgb_algo=xgb_algo,
        xgb_arch=xgb_arch,
        xgb_proto=xgb_proto,
        ocsvm=ocsvm,
    )

    print("[+] Training of all models complete.")


# =========================================================
# 5. LOAD MODELS
# =========================================================

def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_in_dim = 2
    gnn_model = GNNClassifier(gnn_in_dim, 32, len(ALGO_ID_TO_NAME)).to(device)
    gnn_model.load_state_dict(torch.load("gnn_model.pth", map_location=device))
    gnn_model.eval()

    lstm_model = OpcodeLSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=16,
        hidden_dim=32,
        extra_feat_dim=3,
        num_classes=len(ALGO_ID_TO_NAME),
    ).to(device)
    lstm_model.load_state_dict(torch.load("lstm_opcode_model.pth", map_location=device))
    lstm_model.eval()

    xgb_algo = XGBClassifier()
    xgb_algo.load_model("xgb_algo_model.json")

    xgb_arch = XGBClassifier()
    xgb_arch.load_model("xgb_arch_model.json")

    xgb_proto = XGBClassifier()
    xgb_proto.load_model("xgb_proto_model.json")

    ocsvm = joblib.load("ocsvm_model.pkl")

    with open("algo_feature_means.json", "r") as f:
        algo_feature_means = json.load(f)

    if os.path.exists(PROTOCOL_SEQUENCE_JSON):
        with open(PROTOCOL_SEQUENCE_JSON, "r") as f:
            proto_sequences = json.load(f)
    else:
        proto_sequences = {}

    return (
        device,
        gnn_model,
        lstm_model,
        xgb_algo,
        xgb_arch,
        xgb_proto,
        ocsvm,
        algo_feature_means,
        proto_sequences,
    )


# =========================================================
# 5.1 EXPLANATION HELPERS
# =========================================================

def explain_protocol_sequence(proto_name, proto_seq):
    explanations = []
    for idx, step in enumerate(proto_seq):
        text = None
        if proto_name == "MQTT":
            if idx == 0:
                text = "Client opens connection and sends CONNECT."
            elif idx == 1:
                text = "Broker replies CONNACK and establishes session."
            else:
                text = "Broker and client exchange PUBLISH/SUBSCRIBE messages."
        elif proto_name in ("TLS", "DTLS"):
            if idx == 0:
                text = "ClientHello: negotiate cipher suites and version."
            elif idx == 1:
                text = "ServerHello + certificate: authenticate server and share key material."
            else:
                text = "Finished + encrypted application data."
        elif proto_name == "SSH":
            if idx == 0:
                text = "Version exchange and algorithm negotiation."
            elif idx == 1:
                text = "Key exchange and host authentication."
            else:
                text = "User auth and interactive session/port forwarding."
        elif proto_name == "HTTP":
            if idx == 0:
                text = "Client sends HTTP request (e.g., GET /)."
            elif idx == 1:
                text = "Server processes request and prepares response."
            else:
                text = "Response body (HTML/JSON) returned to client."
        else:
            if idx == 0:
                text = "Initial connection / handshake stage."
            elif idx == 1:
                text = "Session setup / negotiation of parameters."
            else:
                text = "Application data transfer over the established channel."
        explanations.append({"step": step, "description": text})
    return explanations


def build_prediction_explanations(final_algo_id, arch_name, proto_name, is_proprietary, features):
    algo_name = ALGO_ID_TO_NAME.get(final_algo_id, f"algo_{final_algo_id}")
    length = features.get("length")
    entropy = features.get("entropy")
    mean_byte = features.get("mean_byte")
    std_byte = features.get("std_byte")

    algo_reason = (
        "The algorithm prediction is made by an ensemble of three models (GNN, LSTM, XGBoost) "
        "using byte-level patterns and statistical features. "
        f"For this file, its feature profile (length={length}, entropy≈{entropy:.2f}, "
        f"mean_byte≈{mean_byte:.2f}, std_byte≈{std_byte:.2f}) is closest to the synthetic "
        f"cluster for algorithm ID {final_algo_id} ({algo_name}), "
        "and the final decision is by majority vote."
    )

    arch_reason = (
        "Architecture is predicted by XGBoost from the same tabular features "
        "(length, entropy, unique byte count, mean and std of bytes). "
        f"The feature profile best matches the training distribution for {arch_name}."
    )

    proto_reason = (
        "Protocol is predicted by XGBoost trained on synthetic samples labelled with TLS/DTLS/MQTT/etc. "
        f"The current feature vector aligns most with the cluster labelled {proto_name}."
    )

    if is_proprietary:
        prop_reason = (
            "The One-Class SVM is trained only on synthetic 'standard' algorithms, learning their normal feature region. "
            "This binary lies outside that region, so it is flagged as an outlier → treated as a proprietary/unusual crypto."
        )
    else:
        prop_reason = (
            "The One-Class SVM sees this feature vector as inside the normal region of known standard algorithms, "
            "so it is marked as standard-like (not proprietary)."
        )

    return {
        "algo_reason": algo_reason,
        "arch_reason": arch_reason,
        "proto_reason": proto_reason,
        "prop_reason": prop_reason,
    }


# =========================================================
# 6. INFERENCE FOR ONE BINARY
# =========================================================

def predict_for_binary(bin_path: str):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Binary not found: {bin_path}")

    (
        device,
        gnn_model,
        lstm_model,
        xgb_algo,
        xgb_arch,
        xgb_proto,
        ocsvm,
        algo_feature_means,
        proto_sequences,
    ) = load_all_models()

    gnn_data, lstm_input, tabular, feat_dict = build_features_from_file(bin_path)
    seq, lengths, extra_feats = lstm_input

    with torch.no_grad():
        gnn_data = gnn_data.to(device)
        logits = gnn_model(gnn_data.x, gnn_data.edge_index, gnn_data.batch)
        probs = torch.softmax(logits, dim=1)
        gnn_pred = int(torch.argmax(probs, dim=1).item())

    with torch.no_grad():
        seq = seq.to(device)
        lengths = lengths.to(device)
        extra_feats = extra_feats.to(device)
        logits = lstm_model(seq, lengths, extra_feats)
        probs = torch.softmax(logits, dim=1)
        lstm_pred = int(torch.argmax(probs, dim=1).item())

    xgb_algo_pred = int(xgb_algo.predict(tabular)[0])
    xgb_arch_pred = int(xgb_arch.predict(tabular)[0])
    xgb_proto_pred = int(xgb_proto.predict(tabular)[0])

    oc_pred = int(ocsvm.predict(tabular)[0])
    is_proprietary = (oc_pred == -1)

    # Majority vote for algorithm
    algo_votes = [gnn_pred, lstm_pred, xgb_algo_pred]
    final_algo_id = max(set(algo_votes), key=algo_votes.count)
    algo_name = ALGO_ID_TO_NAME.get(final_algo_id, f"algo_{final_algo_id}")

    arch_name = ARCH_ID_TO_NAME.get(xgb_arch_pred, f"arch_{xgb_arch_pred}")
    proto_name = PROTOCOL_ID_TO_NAME.get(xgb_proto_pred, f"proto_{xgb_proto_pred}")

    proto_seq = proto_sequences.get(proto_name, ["<no sequence in dataset>"])

    algo_mean_feats = (
        algo_feature_means.get(str(final_algo_id))
        or algo_feature_means.get(final_algo_id)
    )

    protocol_step_explanations = explain_protocol_sequence(proto_name, proto_seq)
    explanations = build_prediction_explanations(
        final_algo_id=final_algo_id,
        arch_name=arch_name,
        proto_name=proto_name,
        is_proprietary=is_proprietary,
        features=feat_dict,
    )

    result = {
        "file": os.path.basename(bin_path),
        "algo_id": final_algo_id,
        "algo_name": algo_name,
        "arch_name": arch_name,
        "protocol_name": proto_name,
        "protocol_sequence": proto_seq,
        "gnn_pred_algo_id": gnn_pred,
        "lstm_pred_algo_id": lstm_pred,
        "xgb_algo_pred_algo_id": xgb_algo_pred,
        "xgb_arch_pred_id": xgb_arch_pred,
        "xgb_proto_pred_id": xgb_proto_pred,
        "ocsvm_is_proprietary": is_proprietary,
        "input_features": feat_dict,
        "algo_mean_features": algo_mean_feats,
        "protocol_step_explanations": protocol_step_explanations,
        "explanations": explanations,
    }
    return result


# =========================================================
# 7. WEB APP (SINGLE WEBSITE)
# =========================================================

app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Crypto Binary Analyzer</title>
  <style>
    body { font-family: system-ui, sans-serif; background:#020617; color:#e5e7eb; margin:0; }
    header { padding:1rem 1.5rem; background:#020617; border-bottom:1px solid #1f2937; }
    h1 { margin:0; font-size:1.3rem; }
    main { max-width:1100px; margin:1.5rem auto; padding:0 1.5rem; }
    .card { background:#020617; border-radius:1rem; border:1px solid #1f2937; padding:1rem 1.25rem; margin-bottom:1rem;}
    label { font-size:0.9rem; }
    input[type=file]{ margin-top:0.4rem; color:#e5e7eb;}
    button { margin-top:0.7rem; padding:0.5rem 1.2rem; border-radius:999px; border:none;
             background:linear-gradient(135deg,#38bdf8,#6366f1); color:#020617; cursor:pointer; font-weight:500;}
    .error { color:#fca5a5; font-size:0.85rem; margin-top:0.4rem;}
    dl { display:grid; grid-template-columns:auto 1fr; column-gap:0.7rem; row-gap:0.2rem; font-size:0.85rem;}
    dt { text-align:right; color:#9ca3af; }
    dd { margin:0; }
    .section-title{ font-size:0.95rem; font-weight:600; margin-top:0.6rem; margin-bottom:0.3rem;}
    .mono{ font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    .pill{ display:inline-block; padding:0.15rem 0.6rem; border-radius:999px; font-size:0.8rem; border:1px solid #1f2937; margin-right:0.3rem; margin-top:0.2rem;}
    .pill.good{ border-color:#22c55e55; color:#4ade80;}
    .pill.bad{ border-color:#ef444455; color:#fca5a5;}
    .pill.neutral{ border-color:#64748b66; color:#9ca3af;}
    .small{ font-size:0.83rem; color:#9ca3af;}
    .hash{ font-size:0.76rem; color:#64748b;}
    .block{ border-top:1px solid #0b1120; padding-top:0.4rem; margin-top:0.4rem;}
  </style>
</head>
<body>
<header>
  <h1>Crypto Binary Analyzer – Web UI</h1>
</header>
<main>

  <div class="card">
    <h2>Upload binary</h2>
    <form method="post" enctype="multipart/form-data">
      <label for="file">Select firmware / binary file:</label><br>
      <input type="file" name="file" id="file" required><br>
      <button type="submit">Analyze</button>
    </form>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    <p class="small">The file is processed once, features are extracted, models run, and the result is logged into a local blockchain JSON ledger.</p>
  </div>

  {% if prediction %}
  <div class="card">
    <h2>Prediction summary</h2>
    <p>
      File: <span class="mono">{{ prediction.file }}</span><br>
    </p>
    <div>
      <span class="pill neutral">Algorithm: {{ prediction.algo_name }} (id={{ prediction.algo_id }})</span>
      <span class="pill neutral">Architecture: {{ prediction.arch_name }}</span>
      <span class="pill neutral">Protocol: {{ prediction.protocol_name }}</span>
      {% if prediction.ocsvm_is_proprietary %}
        <span class="pill bad">Proprietary (OC-SVM outlier)</span>
      {% else %}
        <span class="pill good">Standard-like</span>
      {% endif %}
    </div>

    <div class="section-title">Model votes (algorithm id)</div>
    <dl>
      <dt>GNN</dt><dd>{{ prediction.gnn_pred_algo_id }}</dd>
      <dt>LSTM</dt><dd>{{ prediction.lstm_pred_algo_id }}</dd>
      <dt>XGBoost (algo)</dt><dd>{{ prediction.xgb_algo_pred_algo_id }}</dd>
    </dl>

    <div class="section-title">Input features</div>
    <dl>
      <dt>Length</dt><dd>{{ prediction.input_features.length }}</dd>
      <dt>Unique bytes</dt><dd>{{ prediction.input_features.unique_bytes }}</dd>
      <dt>Entropy</dt><dd>{{ "%.4f"|format(prediction.input_features.entropy) }}</dd>
      <dt>Mean byte</dt><dd>{{ "%.4f"|format(prediction.input_features.mean_byte) }}</dd>
      <dt>Std byte</dt><dd>{{ "%.4f"|format(prediction.input_features.std_byte) }}</dd>
    </dl>

    <div class="section-title">Typical features for this algorithm</div>
    {% if prediction.algo_mean_features %}
      <dl>
        <dt>Length</dt><dd>{{ "%.4f"|format(prediction.algo_mean_features[0]) }}</dd>
        <dt>Unique bytes</dt><dd>{{ "%.4f"|format(prediction.algo_mean_features[1]) }}</dd>
        <dt>Entropy</dt><dd>{{ "%.4f"|format(prediction.algo_mean_features[2]) }}</dd>
        <dt>Mean byte</dt><dd>{{ "%.4f"|format(prediction.algo_mean_features[3]) }}</dd>
        <dt>Std byte</dt><dd>{{ "%.4f"|format(prediction.algo_mean_features[4]) }}</dd>
      </dl>
    {% else %}
      <p class="small">No stats available for this algorithm.</p>
    {% endif %}

    <div class="section-title">Protocol sequence</div>
    <div class="small">
      {% for idx, step in enumerate(prediction.protocol_sequence) %}
        <div>
          <span class="mono">{{ step }}</span>
          {% if prediction.protocol_step_explanations and idx < prediction.protocol_step_explanations|length %}
             – {{ prediction.protocol_step_explanations[idx].description }}
          {% endif %}
        </div>
      {% endfor %}
    </div>

    <div class="section-title">Reasoning</div>
    <div class="small">
      <strong>Algorithm:</strong> {{ prediction.explanations.algo_reason }}<br><br>
      <strong>Architecture:</strong> {{ prediction.explanations.arch_reason }}<br><br>
      <strong>Protocol:</strong> {{ prediction.explanations.proto_reason }}<br><br>
      <strong>Proprietary / OC-SVM:</strong> {{ prediction.explanations.prop_reason }}
    </div>
  </div>
  {% endif %}

  {% if metrics and metrics.overall_ensemble %}
  <div class="card">
    <h2>Overall ensemble metrics (synthetic dataset)</h2>
    <dl>
      <dt>Accuracy</dt><dd>{{ "%.4f"|format(metrics.overall_ensemble.accuracy) }}</dd>
      <dt>Macro recall</dt><dd>{{ "%.4f"|format(metrics.overall_ensemble.macro_recall) }}</dd>
      <dt>Macro precision</dt><dd>{{ "%.4f"|format(metrics.overall_ensemble.macro_precision) }}</dd>
      <dt>False positive rate</dt><dd>{{ "%.4f"|format(metrics.overall_ensemble.false_positive_rate_overall) }}</dd>
    </dl>
  </div>
  {% endif %}

  {% if blockchain and blockchain.last_blocks %}
  <div class="card">
    <h2>Blockchain log (last {{ blockchain.last_blocks|length }} blocks)</h2>
    {% for b in blockchain.last_blocks %}
      <div class="block">
        <div>#{{ b.index }} – {{ b.timestamp }}</div>
        {% if b.data and b.data.file %}
          <div class="small">
            File <span class="mono">{{ b.data.file }}</span>,
            Algo {{ b.data.algo_name }} (id={{ b.data.algo_id }}),
            Arch {{ b.data.arch_name }},
            Proto {{ b.data.protocol_name }},
            {% if b.data.is_proprietary %}Proprietary{% else %}Standard-like{% endif %}
          </div>
        {% endif %}
        <div class="hash">hash={{ b.hash[:16] }}… prev={{ b.previous_hash[:12] }}…</div>
      </div>
    {% endfor %}
  </div>
  {% endif %}

</main>
</body>
</html>
"""

def ensure_models_trained():
    """Check for model files; train if missing."""
    needed = [
        "gnn_model.pth",
        "lstm_opcode_model.pth",
        "xgb_algo_model.json",
        "xgb_arch_model.json",
        "xgb_proto_model.json",
        "ocsvm_model.pkl",
        "algo_feature_means.json",
        MODEL_METRICS_JSON,
    ]
    if not all(os.path.exists(f) for f in needed):
        print("[*] Some model files are missing. Training all models now...")
        train_all_models()
    else:
        print("[+] All model files found. Skipping training.")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    blockchain_info = None
    error = None

    # always try to load metrics (may be None)
    metrics = load_model_metrics() or {}

    if request.method == "POST":
        uploaded = request.files.get("file")
        if not uploaded or uploaded.filename == "":
            error = "Please choose a binary file."
        else:
            # save temporary file
            fd, tmp_path = tempfile.mkstemp()
            os.close(fd)
            try:
                uploaded.save(tmp_path)
                ensure_models_trained()
                prediction = predict_for_binary(tmp_path)

                # update blockchain
                chain = Blockchain()
                block_data = {
                    "file": prediction["file"],
                    "algo_id": prediction["algo_id"],
                    "algo_name": prediction["algo_name"],
                    "arch_name": prediction["arch_name"],
                    "protocol_name": prediction["protocol_name"],
                    "is_proprietary": prediction["ocsvm_is_proprietary"],
                    "features": prediction["input_features"],
                    "overall_metrics": metrics.get("overall_ensemble") if isinstance(metrics, dict) else None,
                }
                new_block = chain.add_block(block_data)
                last_blocks = chain.chain[-5:]
                blockchain_info = {
                    "new_block": new_block,
                    "last_blocks": last_blocks,
                }
            except Exception as e:
                error = f"Error during analysis: {e}"
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    return render_template_string(
        HTML_PAGE,
        prediction=prediction,
        metrics=metrics,
        blockchain=blockchain_info,
        error=error,
        enumerate=enumerate,  # for Jinja loop
    )


if __name__ == "__main__":
    # Make sure models exist before serving web
    ensure_models_trained()
    app.run(host="0.0.0.0", port=5000, debug=True)
