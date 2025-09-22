# bc_mlp_fullbatch_best.py
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

set_seed(42)
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr, ytr = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device).long()
Xte, yte = torch.tensor(Xte, device=device), torch.tensor(yte, device=device).long()

model = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # tente 0 se precisar

for _ in range(150):  # pode usar 100–200
    opt.zero_grad(); F.cross_entropy(model(Xtr), ytr).backward(); opt.step()

with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean().item()
print(f"acuracia: {acc*100:.2f}%")
