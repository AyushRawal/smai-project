# height_classifier_pt.py
#
# Predict shot‑height category ("low", "medium", "high") from
#   • velocity_before_bounce  (m/s)
#   • velocity_after_bounce   (m/s)
#   • bounce_x, bounce_y      (court coordinates, m)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# ────────────────────────────────────────────────────────────
# 1. LOAD & PREP DATA ── replace with your own CSV / DB
# ────────────────────────────────────────────────────────────
df = pd.read_csv("rally_samples.csv")                          # v_before, v_after, bx, by, height_cat

X = df[["v_before", "v_after", "bx", "by"]].values.astype("float32")
y = df["height_cat"].map({"low": 0, "medium": 1, "high": 2}).values.astype("int64")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)

# ────────────────────────────────────────────────────────────
# 2. TORCH DATASET / DATALOADER
# ────────────────────────────────────────────────────────────
class RallyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_loader = DataLoader(RallyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(RallyDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

# ────────────────────────────────────────────────────────────
# 3. MODEL DEFINITION
# ────────────────────────────────────────────────────────────
class HeightNet(nn.Module):
    def __init__(self, in_dim=4, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):                     # x: (batch, 4)
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = HeightNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-3)

# ────────────────────────────────────────────────────────────
# 4. TRAINING LOOP
# ────────────────────────────────────────────────────────────
epochs          = 150
best_val_acc    = 0
patience        = 10
epochs_no_improve= 0

for epoch in range(1, epochs+1):
    # ---- train ---
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()

    # ---- validate ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds  = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch:3d} | val_acc={val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), "height_net_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# ────────────────────────────────────────────────────────────
# 5. INFERENCE EXAMPLE
# ────────────────────────────────────────────────────────────
label_map = {0: "low", 1: "medium", 2: "high"}

def predict(sample_raw):
    """
    sample_raw: 1×4 numpy array with v_before, v_after, bx, by
    """
    sample_std = scaler.transform(sample_raw.astype("float32"))
    with torch.no_grad():
        x = torch.from_numpy(sample_std).to(device)
        probs = torch.softmax(model(x), dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
    return label_map[label_idx], probs.cpu().numpy()

# Example usage
new_shot = np.array([[31.2, 18.7, 5.4, 1.3]])
cls, probs = predict(new_shot)
print("Predicted category:", cls, "| class probabilities:", probs.round(3))

