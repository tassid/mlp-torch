# nome: tassiane anzolin
# MLP mínima em Torch + adaptação
# dataset: câncer de mama (sklearn)

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# dados
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X).astype("float32")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# tensores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Xtr, ytr = torch.tensor(Xtr).to(device), torch.tensor(ytr).long().to(device)
Xte, yte = torch.tensor(Xte).to(device), torch.tensor(yte).long().to(device)

# modelo
model = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# treino
for _ in range(100):
    opt.zero_grad()
    loss = F.cross_entropy(model(Xtr), ytr)
    loss.backward()
    opt.step()

# teste
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean().item()
print(f"acuracia: {acc*100:.2f}%")