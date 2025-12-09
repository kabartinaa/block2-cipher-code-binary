# train_gnn.py
#
# Standalone GNN training script using synthetic graph data.
# Does NOT depend on feature_extractor.py, graphs/, or label_map.json.

import random
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# ---------- GNN MODEL ----------

class GNNClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim,num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        # x: [num_nodes, in_dim]
        # edge_index: [2, num_edges]
        # batch: graph id for each node
        h = self.conv1(x, edge_index)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)

        # Pool node embeddings to graph embedding
        g = global_mean_pool(h, batch)   # [num_graphs, hidden_dim]

        out = self.fc(g)                 # [num_graphs, num_classes]
        return out


# ---------- SYNTHETIC DATASET GENERATION ----------

def make_synthetic_graph(num_nodes: int, feature_dim: int, num_classes: int) -> Data:
    """
    Create one random graph:
      - num_nodes nodes
      - random edges (with at least a chain for connectivity)
      - random node features
      - random label in [0, num_classes-1]
    """

    # node features: standard normal
    x = torch.randn(num_nodes, feature_dim)

    # build a simple connected chain: 0-1-2-...-(n-1)
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])   # make it undirected-ish

    # add some random extra edges
    extra_edges = num_nodes  # you can tune this
    for _ in range(extra_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u != v:
            edges.append([u, v])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # random label
    y = torch.tensor([random.randint(0, num_classes - 1)], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def make_synthetic_dataset(
    num_graphs: int = 500,
    num_classes: int = 4,
    min_nodes: int = 5,
    max_nodes: int = 25,
    feature_dim: int = 16,
):
    dataset = []
    for _ in range(num_graphs):
        n = random.randint(min_nodes, max_nodes)
        g = make_synthetic_graph(n, feature_dim, num_classes)
        dataset.append(g)
    return dataset, feature_dim, num_classes


# ---------- TRAIN / EVAL HELPERS ----------

def split_dataset(dataset, train_ratio=0.8, seed=42):
    random.Random(seed).shuffle(dataset)
    n = len(dataset)
    n_train = int(train_ratio * n)
    train_set = dataset[:n_train]
    test_set = dataset[n_train:]
    return train_set, test_set


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total += batch.num_graphs
    acc = correct / total if total > 0 else 0.0
    return acc


# ---------- MAIN ----------

if __name__ == "__main__":
    # 1) Create synthetic dataset in memory
    NUM_GRAPHS = 500
    NUM_CLASSES = 4
    MIN_NODES = 5
    MAX_NODES = 25
    FEATURE_DIM = 16

    dataset, feature_dim, num_classes = make_synthetic_dataset(
        num_graphs=NUM_GRAPHS,
        num_classes=NUM_CLASSES,
        min_nodes=MIN_NODES,
        max_nodes=MAX_NODES,
        feature_dim=FEATURE_DIM,
    )

    print(f"[+] Created synthetic dataset with {len(dataset)} graphs")
    print(f"[+] Node feature dimension: {feature_dim}")
    print(f"[+] Number of classes: {num_classes}")

    # 2) Split into train/test
    train_set, test_set = split_dataset(dataset, train_ratio=0.8, seed=42)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # 3) Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNClassifier(in_dim=feature_dim, hidden_dim=64, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4) Train
    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    # 5) Save model
    torch.save(model.state_dict(), "gnn_model_synthetic.pth")
    print("[+] Model saved as gnn_model_synthetic.pth")
