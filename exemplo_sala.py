# MLP (RMSProp) no dataset Breast Cancer Wisconsin (scikit-learn)
# - Split estratificado train/val/test (64/16/20)
# - Padronização sem vazamento (fit no train)
# - MLP com BatchNorm + SiLU
# - RMSProp + ReduceLROnPlateau
# - Early stopping por val_loss; restaura melhor estado e avalia no teste

import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

SEED = 42
EPOCHS = 200
LR = 1e-3
ALPHA = 0.9           # decay do RMSProp
WEIGHT_DECAY = 5e-5
PATIENCE = 20
CLIP_NORM = 1.0
H1, H2 = 128, 64

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Dados ----------------
X, y = load_breast_cancer(return_X_y=True)
# 64/16/20
X_trv, X_te, y_trv, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_tr,  X_va, y_tr,  y_va = train_test_split(X_trv, y_trv, test_size=0.20, stratify=y_trv, random_state=SEED)

sc = StandardScaler().fit(X_tr)
X_tr = sc.transform(X_tr).astype("float32")
X_va = sc.transform(X_va).astype("float32")
X_te = sc.transform(X_te).astype("float32")

Xtr = torch.from_numpy(X_tr).to(device)
Xva = torch.from_numpy(X_va).to(device)
Xte = torch.from_numpy(X_te).to(device)
ytr = torch.from_numpy(y_tr).long().to(device)
yva = torch.from_numpy(y_va).long().to(device)
yte = torch.from_numpy(y_te).long().to(device)

# ---------------- Modelo ----------------
class MLP(nn.Module):
    def __init__(self, d_in, h1, h2, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.SiLU(),
            nn.Linear(h1, h2),   nn.BatchNorm1d(h2), nn.SiLU(),
            nn.Linear(h2, d_out)
        )
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x)

model = MLP(Xtr.shape[1], H1, H2).to(device)

opt = torch.optim.RMSprop(model.parameters(), lr=LR, alpha=ALPHA, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)

best_state, best_val, stale = None, float('inf'), 0

for epoch in range(1, EPOCHS+1):
    # treino full-batch
    model.train()
    opt.zero_grad(set_to_none=True)
    logits = model(Xtr)
    loss = crit(logits, ytr)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORm:=CLIP_NORM)
    opt.step()

    # validação
    model.eval()
    with torch.no_grad():
        v_logits = model(Xva)
        v_loss = crit(v_logits, yva).item()
    scheduler.step(v_loss)

    if v_loss < best_val - 1e-8:
        best_val = v_loss
        best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1

    if epoch % 10 == 0 or stale == 0:
        v_acc = (v_logits.argmax(1) == yva).float().mean().item()
        lr_now = opt.param_groups[0]['lr']
        print(f"epoch {epoch:03d} | train_loss {loss.item():.4f} | val_loss {v_loss:.4f} | val_acc {v_acc*100:.2f}% | lr {lr_now:.2e}")

    if stale >= PATIENCE:
        print(f"early stopping @ epoch {epoch}")
        break

# restaura e avalia no teste
if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    te_logits = model(Xte)
    te_probs  = te_logits.softmax(1)[:,1].detach().cpu().numpy()
    te_pred   = te_logits.argmax(1).detach().cpu().numpy()
    acc = accuracy_score(y_te, te_pred)
    auc = roc_auc_score(y_te, te_probs)
print(f"test_acc {acc*100:.2f}% | test_auc {auc:.4f}")
