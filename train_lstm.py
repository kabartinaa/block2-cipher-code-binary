# train_lstm.py
#
# Standalone LSTM trainer on opcode sequences + extra features.
# Does NOT need angr or external files. Synthetic data is generated in memory.

import random
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader


# ----------------- Synthetic dataset -----------------

class OpcodeSequenceDataset(Dataset):
    """
    Each sample:
      - opcode_seq: [seq_len] (list of ints, token ids)
      - extra_feats: [F] (tensor of floats, global features for the whole binary)
      - label: int (class id, e.g., algo type)
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 4,
        vocab_size: int = 50,
        min_len: int = 20,
        max_len: int = 80,
        extra_feat_dim: int = 3,
        seed: int = 42,
    ):
        super().__init__()
        random.seed(seed)
        self.samples = []

        # define "preferred" opcode ranges for each class (to give structure)
        # class 0: opcodes ~ 0-12
        # class 1: opcodes ~ 10-24
        # class 2: opcodes ~ 20-36
        # class 3: opcodes ~ 30-49
        class_ranges = [
            (0, 12),
            (10, 24),
            (20, 36),
            (30, vocab_size - 1),
        ]

        for _ in range(num_samples):
            label = random.randint(0, num_classes - 1)
            lo, hi = class_ranges[label]

            seq_len = random.randint(min_len, max_len)
            opcode_seq = []
            for _ in range(seq_len):
                # mostly sample from class-specific range, sometimes random
                if random.random() < 0.8:
                    opcode = random.randint(lo, hi)
                else:
                    opcode = random.randint(0, vocab_size - 1)
                opcode_seq.append(opcode)

            # --- compute some "related" features ---
            # feature 1: normalized length
            f_len = seq_len / float(max_len)

            # feature 2: unique opcode ratio
            unique_ops = len(set(opcode_seq))
            f_unique = unique_ops / float(vocab_size)

            # feature 3: simple entropy-like measure (over opcode ids)
            # not true Shannon entropy, but enough to vary by pattern
            freq = {}
            for op in opcode_seq:
                freq[op] = freq.get(op, 0) + 1
            ent = 0.0
            for c in freq.values():
                p = c / float(seq_len)
                ent -= p * math.log2(p)
            f_entropy = ent / math.log2(min(vocab_size, seq_len))  # normalized

            extra_feats = torch.tensor([f_len, f_unique, f_entropy], dtype=torch.float)

            self.samples.append(
                {
                    "opcode_seq": opcode_seq,
                    "extra_feats": extra_feats,
                    "label": label,
                }
            )

        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.extra_feat_dim = extra_feat_dim

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # opcode_seq as LongTensor
        seq = torch.tensor(s["opcode_seq"], dtype=torch.long)
        extra = s["extra_feats"]  # already tensor
        label = torch.tensor(s["label"], dtype=torch.long)
        return seq, extra, label


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """
    batch is a list of (seq, extra_feats, label)
    We need to:
      - pad sequences to max length in batch
      - keep original lengths
      - stack extra_feats and labels
    """
    seqs, extras, labels = zip(*batch)  # tuples of tensors

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    # pad sequences: result shape [batch_size, max_len]
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

    extras = torch.stack(extras, dim=0)
    labels = torch.stack(labels, dim=0)

    return padded_seqs, lengths, extras, labels


# ----------------- LSTM model -----------------

class OpcodeLSTMClassifier(nn.Module):
    """
    Model:
      - Embedding over opcode IDs
      - LSTM over sequence
      - Use last hidden state as sequence embedding
      - Concatenate with extra features
      - Fully-connected classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        extra_feat_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_dim + extra_feat_dim, num_classes)

    def forward(self, seqs, lengths, extra_feats):
        """
        seqs: [batch, max_len] LongTensor of opcode ids
        lengths: [batch] original lengths
        extra_feats: [batch, extra_feat_dim]
        """
        # [batch, max_len, embed_dim]
        emb = self.embedding(seqs)

        # pack for LSTM
        packed = pack_padded_sequence(
            emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # output: packed_out, (h_n, c_n)
        _, (h_n, _) = self.lstm(packed)
        # h_n: [num_layers * num_directions, batch, hidden_dim]
        # we only have 1 layer, 1 direction => h_n[0] is [batch, hidden_dim]
        seq_emb = h_n[0]

        # concat extra features
        combined = torch.cat([seq_emb, extra_feats], dim=1)  # [batch, hidden_dim + extra_feat_dim]

        logits = self.fc(combined)  # [batch, num_classes]
        return logits


# ----------------- Train / eval -----------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for seqs, lengths, extras, labels in loader:
        seqs = seqs.to(device)
        lengths = lengths.to(device)
        extras = extras.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(seqs, lengths, extras)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / float(total_samples)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for seqs, lengths, extras, labels in loader:
            seqs = seqs.to(device)
            lengths = lengths.to(device)
            extras = extras.to(device)
            labels = labels.to(device)

            logits = model(seqs, lengths, extras)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / float(total) if total > 0 else 0.0
    return acc


# ----------------- Main -----------------

if __name__ == "__main__":
    # 1) create synthetic opcode dataset
    NUM_SAMPLES = 1000
    NUM_CLASSES = 4
    VOCAB_SIZE = 50
    EXTRA_FEAT_DIM = 3

    dataset = OpcodeSequenceDataset(
        num_samples=NUM_SAMPLES,
        num_classes=NUM_CLASSES,
        vocab_size=VOCAB_SIZE,
        min_len=20,
        max_len=80,
        extra_feat_dim=EXTRA_FEAT_DIM,
        seed=42,
    )

    # train/test split
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # 2) build model
    EMBED_DIM = 32
    HIDDEN_DIM = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpcodeLSTMClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        extra_feat_dim=EXTRA_FEAT_DIM,
        num_classes=NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3) train
    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # 4) save model
    torch.save(model.state_dict(), "lstm_opcode_model.pth")
    print("[+] LSTM model saved as lstm_opcode_model.pth")
