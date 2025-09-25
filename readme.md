# mlp mínima em pytorch

implementação simples de uma rede **mlp** em [pytorch](https://pytorch.org/) usando o dataset **breast cancer wisconsin** do scikit-learn.  
o objetivo é mostrar um exemplo claro, curto e fácil de adaptar para outros datasets.

---

## arquivos

- `max.py` → versão principal (full-batch, ~150 épocas, 98–99% de acurácia).  
- `min.py` → versão mínima (poucas linhas, sem validação).  
- `min_early.py` → versão com validação, scheduler e early stopping.

todo: tem early stopping mesmo??
---

## requisitos

- python 3.10+  
- bibliotecas:
  ```bash
  pip install torch scikit-learn numpy
