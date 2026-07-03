"""
TextCNN (Kim, 2014) para classificação de acórdãos TCU sobre o VOTO_LIMPO.

Modelo compacto (roda em CPU), treinado do zero, com CrossEntropy ponderada
por classe. Pensado para a disciplina de Deep Learning: demonstra que, sem
vazamento, profundidade não vence o baseline TF-IDF em corpus desbalanceado.
"""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

from .pesos import NOMES_CLASSES, pesos_tensor

RANDOM_STATE = 42


def _tokens(t: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ]{2,}", t.lower())


def montar_vocabulario(textos: list[str], max_vocab: int = 20_000) -> dict[str, int]:
    cnt = Counter(tok for t in textos for tok in _tokens(t))
    vocab = ["<pad>", "<unk>"] + [w for w, _ in cnt.most_common(max_vocab)]
    return {w: i for i, w in enumerate(vocab)}


def encode(textos: list[str], w2i: dict[str, int], max_len: int = 300) -> np.ndarray:
    unk = w2i.get("<unk>", 1)

    def _enc(t: str) -> list[int]:
        ids = [w2i.get(w, unk) for w in _tokens(t)][:max_len]
        return ids + [0] * (max_len - len(ids))

    return np.asarray([_enc(t) for t in textos], dtype=np.int64)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_classes: int = 3,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in filter_sizes]
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x).transpose(1, 2)                              # [B, E, L]
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs] # [B, nf]*K
        return self.fc(self.drop(torch.cat(h, dim=1)))


def _treinar_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    vocab_size: int,
    class_weights: np.ndarray,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> np.ndarray:
    torch.manual_seed(RANDOM_STATE)
    net = TextCNN(vocab_size).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    Xt = torch.tensor(X_tr, device=device)
    yt = torch.tensor(y_tr, device=device)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(net(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        preds = net(torch.tensor(X_te, device=device)).argmax(1).cpu().numpy()
    return preds


def treinar_kfold(
    df: pd.DataFrame,
    campo: str = "VOTO_LIMPO",
    n_splits: int = 5,
    max_vocab: int = 20_000,
    max_len: int = 300,
    epochs: int = 8,
    device: str = "cpu",
) -> dict:
    """K-Fold estratificado com pesos de classe por fold (recomputados no train fold)."""
    from .pesos import LABEL2ID

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    textos = df[campo].astype(str).tolist()
    y = np.asarray([LABEL2ID[l] for l in df["LABEL"].tolist()])

    w2i = montar_vocabulario(textos, max_vocab=max_vocab)
    X = encode(textos, w2i, max_len=max_len)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros_like(y)

    for k, (tr, te) in enumerate(skf.split(X, y), start=1):
        w = pesos_tensor(y[tr])
        oof[te] = _treinar_fold(
            X[tr], y[tr], X[te],
            vocab_size=len(w2i),
            class_weights=w,
            epochs=epochs,
            device=device,
        )
        print(f"  fold {k}/{n_splits} concluído")

    f1 = float(f1_score(y, oof, average="macro", zero_division=0))
    acc = float(accuracy_score(y, oof))
    cm = confusion_matrix(y, oof).tolist()

    per_class = f1_score(y, oof, average=None, labels=list(range(len(NOMES_CLASSES))), zero_division=0)
    per_class_dict = {NOMES_CLASSES[i]: float(v) for i, v in enumerate(per_class)}

    return {
        "campo": campo,
        "f1_macro": f1,
        "accuracy": acc,
        "per_class_f1": per_class_dict,
        "confusion_matrix": cm,
        "n_splits": n_splits,
        "vocab_size": len(w2i),
        "max_len": max_len,
        "epochs": epochs,
        "classification_report": classification_report(
            y, oof, target_names=NOMES_CLASSES, zero_division=0, digits=3
        ),
    }
