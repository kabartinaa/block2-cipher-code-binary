import os
import sys
import math
import json
import random
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

    for idx in range(n_samples):
        algo_id = random.randint(0, num_algos - 1)
        arch_id = algo_id % num_archs
        proto_id = algo_id % num_protos

        data = synthetic_sample(algo_id)
        gnn_data, lstm_input, tabular, feat_dict = build_features_from_bytes(data)
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
# 4. TRAIN ALL FOUR MODELS (FAST VERSION)
# =========================================================

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

    algo_votes = [gnn_pred, lstm_pred, xgb_algo_pred]
    final_algo_id = max(set(algo_votes), key=algo_votes.count)
    algo_name = ALGO_ID_TO_NAME.get(final_algo_id, f"algo_{final_algo_id}")

    arch_name = ARCH_ID_TO_NAME.get(xgb_arch_pred, f"arch_{xgb_arch_pred}")
    proto_name = PROTOCOL_ID_TO_NAME.get(xgb_proto_pred, f"proto_{xgb_proto_pred}")

    proto_seq = proto_sequences.get(proto_name, ["<no sequence in dataset>"])

    algo_mean_feats = algo_feature_means.get(str(final_algo_id)) or algo_feature_means.get(final_algo_id)

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
    }
    return result


# =========================================================
# 7. MAIN ENTRY
# =========================================================

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(__file__)} <binary_file>")
        sys.exit(1)

    bin_path = sys.argv[1]

    needed = [
        "gnn_model.pth",
        "lstm_opcode_model.pth",
        "xgb_algo_model.json",
        "xgb_arch_model.json",
        "xgb_proto_model.json",
        "ocsvm_model.pkl",
        "algo_feature_means.json",
    ]
    if not all(os.path.exists(f) for f in needed):
        print("[*] Some model files are missing. Training all models now...")
        train_all_models()
    else:
        print("[+] All model files found. Skipping training.")

    res = predict_for_binary(bin_path)

    print("\n==================== FINAL PREDICTION ====================")
    print(f"File               : {res['file']}")
    print(f"Algorithm          : {res['algo_name']} (id={res['algo_id']})")
    print(f"Architecture       : {res['arch_name']}")
    print(f"Protocol           : {res['protocol_name']}")
    print(f"Proprietary?       : {'YES (OC-SVM outlier)' if res['ocsvm_is_proprietary'] else 'NO (standard-like)'}")

    print("\nProtocol sequence (from dataset):")
    for step in res["protocol_sequence"]:
        print(f"  - {step}")

    print("\nModel votes (algorithm id):")
    print(f"  GNN   : {res['gnn_pred_algo_id']}")
    print(f"  LSTM  : {res['lstm_pred_algo_id']}")
    print(f"  XGBoost (algo): {res['xgb_algo_pred_algo_id']}")

    print("\nInput features (computed from this binary):")
    for k, v in res["input_features"].items():
        print(f"  {k:12s}: {v}")

    print("\nTypical features for this algorithm (mean over training dataset):")
    if res["algo_mean_features"] is not None:
        feat_names = ["length", "unique_bytes", "entropy", "mean_byte", "std_byte"]
        for name, val in zip(feat_names, res["algo_mean_features"]):
            print(f"  {name:12s}: {val}")
    else:
        print("  <no stats available>")

    print("==========================================================\n")


if __name__ == "__main__":
    main()
