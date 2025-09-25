# nome: tassiane anzolin
# MLP mínima em Torch + adaptação
# dataset: câncer de mama (sklearn)

import math, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_recall_curve, roc_curve
)

# ----------------------- utils -----------------------

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)

class WarmupCosine:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float, min_lr: float):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base = base_lr
        self.min = min_lr
        self.t = 0
    def step(self):
        self.t += 1
        if self.t <= self.warmup:
            lr = self.base * self.t / max(1, self.warmup)
        else:
            # cosine on remaining epochs
            tt = self.t - self.warmup
            T = max(1, self.total - self.warmup)
            cos = 0.5 * (1 + math.cos(math.pi * tt / T))
            lr = self.min + (self.base - self.min) * cos
        for g in self.opt.param_groups:
            g['lr'] = lr
        return lr

# retorna threshold ótimo por Youden J e também o melhor por F1

def choose_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    t_you = thr[np.argmax(j)]
    # F1-based
    p, r, th = precision_recall_curve(y_true, y_prob)
    f1 = 2 * p * r / (p + r + 1e-9)
    t_f1 = th[np.argmax(f1[:-1])] if len(th) > 0 else 0.5
    return float(t_you), float(t_f1)

# ----------------------- treino de um split -----------------------

def train_one_split(X_tr, y_tr, X_va, y_va, *,
                    device: torch.device,
                    seeds=(42,1337,777),
                    max_epochs=200,
                    warmup=10,
                    lr=1e-3,
                    min_lr=1e-5,
                    weight_decay=5e-5,
                    patience=20,
                    label_smoothing=0.05,
                    hidden=128):
    # ensemble por múltiplos seeds (snapshot ensemble leve)
    probs_va_members = []

    for sd in seeds:
        set_seed(sd)
        model = MLP(X_tr.shape[1], hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = WarmupCosine(opt, warmup_epochs=warmup, total_epochs=max_epochs, base_lr=lr, min_lr=min_lr)
        crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # full-batch tensors
        xb_tr = torch.from_numpy(X_tr).to(device)
        yb_tr = torch.from_numpy(y_tr).long().to(device)
        xb_va = torch.from_numpy(X_va).to(device)
        yb_va = torch.from_numpy(y_va).long().to(device)

        best_auc, best_state, stale = -1.0, None, 0

        for ep in range(1, max_epochs+1):
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(xb_tr)
            loss = crit(logits, yb_tr)
            loss.backward(); opt.step(); sched.step()

            # validação por AUC
            model.eval()
            with torch.no_grad():
                val_logits = model(xb_va)
                val_prob1 = val_logits.softmax(1)[:,1].detach().cpu().numpy()
                val_auc = roc_auc_score(y_va, val_prob1)

            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1

            if ep % 20 == 0 or stale == 0:
                print(f"seed {sd} | epoch {ep:03d} | train_loss {loss.item():.4f} | val_auc {val_auc:.4f}")
            if stale >= patience:
                print(f"seed {sd}: early stop @ {ep}, best val_auc {best_auc:.4f}")
                break

        # coleta prob das melhores weights
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            probs = model(xb_va).softmax(1)[:,1].detach().cpu().numpy()
        probs_va_members.append(probs)

    # média dos membros (ensemble)
    prob_va = np.mean(np.stack(probs_va_members, axis=0), axis=0)
    t_you, t_f1 = choose_threshold(y_va, prob_va)

    # métricas com cada threshold
    def binarize(th):
        return (prob_va >= th).astype(np.int64)

    acc_you = accuracy_score(y_va, binarize(t_you))
    f1_you = f1_score(y_va, binarize(t_you))
    acc_f1  = accuracy_score(y_va, binarize(t_f1))
    f1_f1   = f1_score(y_va, binarize(t_f1))

    return {
        'val_auc': float(roc_auc_score(y_va, prob_va)),
        'thr_you': t_you, 'thr_f1': t_f1,
        'acc_you': float(acc_you), 'f1_you': float(f1_you),
        'acc_f1': float(acc_f1), 'f1_f1': float(f1_f1),
        'probs_val': prob_va,
    }

# ----------------------- pipeline CV -----------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, y = load_breast_cancer(return_X_y=True)

    # CV estratificado 5x com padronização SEM vazamento (fit só no train de cada fold)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_stats = []

    for fold, (idx_tr, idx_va) in enumerate(kf.split(X, y), start=1):
        X_tr, X_va = X[idx_tr], X[idx_va]
        y_tr, y_va = y[idx_tr], y[idx_va]

        sc = StandardScaler().fit(X_tr)
        X_tr = sc.transform(X_tr).astype('float32')
        X_va = sc.transform(X_va).astype('float32')

        stats = train_one_split(
            X_tr, y_tr, X_va, y_va,
            device=device,
            seeds=(42,1337,777),
            max_epochs=200, warmup=10,
            lr=1e-3, min_lr=1e-5, weight_decay=5e-5,
            patience=20, label_smoothing=0.05,
            hidden=128,
        )
        print(f"FOLD {fold} | AUC {stats['val_auc']:.4f} | acc@Youden {stats['acc_you']:.4f} | f1@Youden {stats['f1_you']:.4f} | acc@F1 {stats['acc_f1']:.4f} | f1@F1 {stats['f1_f1']:.4f}")
        fold_stats.append(stats)

    # agrega métricas
    def mean_std(key):
        arr = np.array([s[key] for s in fold_stats], dtype=float)
        return float(arr.mean()), float(arr.std())

    auc_m, auc_s = mean_std('val_auc')
    accy_m, accy_s = mean_std('acc_you')
    f1y_m, f1y_s   = mean_std('f1_you')

    print('\n==== CV RESULTS (5 folds, ensemble 3 seeds) ====')
    print(f"AUC: {auc_m:.4f} ± {auc_s:.4f}")
    print(f"ACC@Youden: {accy_m:.4f} ± {accy_s:.4f}")
    print(f"F1@Youden:  {f1y_m:.4f} ± {f1y_s:.4f}")

if __name__ == '__main__':
    main()
