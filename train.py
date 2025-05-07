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
df = pd.read_csv("dataset.csv")                          # v_before, v_after, bx, by, height_cat

# bounce_frame,vx_before,vy_before,bounce_x,bounce_y,height_category
X = df[["vx_before", "vy_before", "bounce_x", "bounce_y"]].values.astype("float32")
y = df["height_category"].map({"low": 0, "mid": 1, "high": 2}).values.astype("int64")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)

# ────────────────────────────────────────────────────────────
# 2. TORCH DATASET / DATALOADER
# ────────────────────────────────────────────────────────────
class HeightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_loader = DataLoader(HeightDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(HeightDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

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
epochs            = 8
best_val_acc      = 0
patience          = 3
epochs_no_improve = 0

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
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            
            # Store predictions and true labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            
    val_acc = correct / total
    
    print(f"Epoch {epoch:3d} | val_acc={val_acc:.4f}")
    
    # For a nicer visual if you want to save it (optional)
    if epoch == epochs or (epochs_no_improve + 1 >= patience and val_acc <= best_val_acc):
        # Print confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix:")
        print(cm)
        
        # Print classification report
        class_labels = ['low', 'medium', 'high']
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_labels))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix.png')
        plt.close()


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


