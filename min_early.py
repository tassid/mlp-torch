# nome: tassiane anzolin
# MLP mínima em Torch + adaptação
# dataset: câncer de mama (sklearn)

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_loaders(batch_size: int = 32):
    X, y = load_breast_cancer(return_X_y=True)

    # split em train/val/test (80/20 e depois 80/20 do train => 64/16/20)
    X_trv, X_te, y_trv, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trv, y_trv, test_size=0.20, stratify=y_trv, random_state=42
    )

    # padroniza SEM vazamento (fit no train apenas)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr).astype("float32")
    X_va = sc.transform(X_va).astype("float32")
    X_te = sc.transform(X_te).astype("float32")

    # tensores
    X_tr = torch.from_numpy(X_tr)
    y_tr = torch.from_numpy(y_tr).long()
    X_va = torch.from_numpy(X_va)
    y_va = torch.from_numpy(y_va).long()
    X_te = torch.from_numpy(X_te)
    y_te = torch.from_numpy(y_te).long()

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)

    input_dim = X_tr.shape[1]
    return train_loader, val_loader, test_loader, input_dim


def make_model(input_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(64, 2),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += yb.size(0)
    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


def train(
    max_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    es_patience: int = 10,
    seed: int = 42,
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, input_dim = make_loaders(batch_size)

    model = make_model(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler baseado em validação (reduz LR quando val_loss estaciona)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_state = None
    best_val_loss = float("inf")
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0)
            seen += yb.size(0)

        train_loss = running / seen
        val_loss, val_acc = evaluate(model, val_loader, device)

        # step no scheduler com base na perda de validação
        scheduler.step(val_loss)

        # early stopping
        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or improved:
            lr_now = opt.param_groups[0]["lr"]
            print(
                f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc*100:.2f}% | lr {lr_now:.2e}"
            )

        if stale >= es_patience:
            print(f"early stopping at epoch {epoch} (no improvement for {es_patience} epochs)")
            break

    # restaura o melhor estado e avalia no teste
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"test_loss {test_loss:.4f} | test_acc {test_acc*100:.2f}%")
    return model


if __name__ == "__main__":
    train()
