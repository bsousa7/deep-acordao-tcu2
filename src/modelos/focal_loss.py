"""
Focal Loss ponderada — para uso com transformers HuggingFace ou treino manual.

Fórmula (Lin et al., 2017):
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)

Aqui `alpha` é o vetor de pesos por classe (frequência inversa normalizada)
e `gamma` é o foco (2.0 é o padrão da literatura).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        idx = targets.view(-1, 1)
        log_pt = log_probs.gather(1, idx).squeeze(1)
        pt = probs.gather(1, idx).squeeze(1)

        focal = (1.0 - pt).clamp(min=1e-8) ** self.gamma
        loss = -focal * log_pt

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha.gather(0, targets) * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def alpha_por_frequencia_inversa(y_ids: np.ndarray, num_classes: int) -> torch.Tensor:
    """Vetor `alpha` com peso inversamente proporcional à frequência (normalizado)."""
    counts = np.bincount(y_ids, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    inv = counts.sum() / counts
    inv = inv / inv.mean()
    return torch.tensor(inv, dtype=torch.float32)
