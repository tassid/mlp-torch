# mlp mínima em pytorch

implementação simples de uma rede **mlp** em [pytorch](https://pytorch.org/) usando o dataset **breast cancer wisconsin** do scikit-learn.  
o objetivo é mostrar um exemplo claro, curto e fácil de adaptar para outros datasets.

---

## arquivos

- `bc_mlp_fullbatch_best.py` → versão principal (full-batch, ~150 épocas, 98–99% de acurácia).  
- `bc_mlp_min.py` → versão mínima (poucas linhas, sem validação).  
- `bc_mlp_min_tuned.py` → versão com validação, scheduler e early stopping.

---

## requisitos

- python 3.10+  
- bibliotecas:
  ```bash
  pip install torch scikit-learn numpy
