"""
Fine-tuning LegalBert-pt (com LoRA opcional) sobre VOTO_LIMPO, com pesos de classe.

Duas variantes de loss expostas:
  * WeightedCE   — cross-entropy ponderada (padrão).
  * FocalLoss    — para desbalanceamento severo (usa src.modelos.focal_loss).
"""

from __future__ import annotations

import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from .focal_loss import FocalLoss, alpha_por_frequencia_inversa
from .pesos import ID2LABEL, LABEL2ID, NOMES_CLASSES, pesos_tensor

logger = logging.getLogger(__name__)

MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"
MODEL_FALLBACK = "neuralmind/bert-base-portuguese-cased"

RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PARAMS = dict(
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_ratio=0.10,
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    seed=RANDOM_STATE,
    fp16=True,
    logging_steps=50,
    report_to="none",
)

LORA_CONFIG_PARAMS = dict(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    bias="none",
)


def carregar_modelo(num_labels: int = 3) -> tuple:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    for nome in [MODEL_NAME, MODEL_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(nome)
            model = AutoModelForSequenceClassification.from_pretrained(
                nome,
                num_labels=num_labels,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            # Reinit do pooler (checkpoint LegalBert-pt não traz pooler treinado).
            std = getattr(model.config, "initializer_range", 0.02)
            try:
                dense = model.bert.pooler.dense
                torch.nn.init.normal_(dense.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(dense.bias)
            except AttributeError:
                pass
            logger.info("Modelo carregado: %s", nome)
            return tokenizer, model
        except Exception as exc:
            logger.warning("Falha ao carregar %s: %s. Tentando fallback...", nome, exc)
    raise RuntimeError("Nenhum modelo base disponível.")


def aplicar_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(task_type=TaskType.SEQ_CLS, **LORA_CONFIG_PARAMS)
    model = get_peft_model(model, config)

    # Descongelar o pooler para acompanhar os adapters.
    for name, param in model.named_parameters():
        if "pooler" in name:
            param.requires_grad = True

    model.print_trainable_parameters()
    return model


def _criar_dataset(textos: list[str], labels: list[int], tokenizer, max_length: int = 512):
    from datasets import Dataset

    enc = tokenizer(textos, truncation=True, padding="max_length", max_length=max_length)
    enc["labels"] = labels
    return Dataset.from_dict(enc)


def _compute_metrics():
    def _fn(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "f1_macro": float(f1_score(label_ids, preds, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(label_ids, preds)),
        }

    return _fn


def _weighted_trainer(class_weights):
    """Trainer com cross-entropy ponderada por classe."""
    import torch
    import torch.nn.functional as F
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits, labels, weight=torch.as_tensor(class_weights).to(logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _focal_trainer(alpha, gamma: float):
    """Trainer com Focal Loss (Lin et al., 2017), alpha = class weights."""
    import torch
    from transformers import Trainer

    class FocalTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._focal = FocalLoss(alpha=torch.as_tensor(alpha), gamma=gamma)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self._focal(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return FocalTrainer


def kfold(
    df: pd.DataFrame,
    campo: str = "VOTO_LIMPO",
    n_splits: int = N_SPLITS,
    loss: str = "weighted_ce",
    focal_gamma: float = 2.0,
    usar_lora: bool = True,
    output_base: str | Path = "resultados/modelos",
) -> dict:
    """
    LoRA K-Fold — VOTO_LIMPO, com pesos de classe por fold.

    loss: 'weighted_ce' (padrão) ou 'focal'.
    """
    import torch
    from transformers import EarlyStoppingCallback, TrainingArguments

    labels_arr = df["LABEL"].values
    textos_arr = df[campo].fillna("").values

    from collections import Counter

    for cls, cnt in Counter(labels_arr).items():
        if cnt < n_splits:
            raise ValueError(
                f"Classe '{cls}' tem {cnt} amostras (< {n_splits} folds). Reduza n_splits."
            )

    yid = np.array([LABEL2ID[l] for l in labels_arr])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_f1: list[float] = []
    fold_acc: list[float] = []

    for k, (tr, va) in enumerate(skf.split(textos_arr, yid), start=1):
        out = Path(output_base) / f"fold_{k}"
        out.mkdir(parents=True, exist_ok=True)

        tokenizer, model = carregar_modelo()
        if usar_lora:
            model = aplicar_lora(model)

        ytr = yid[tr].tolist()
        yva = yid[va].tolist()

        weights = pesos_tensor(np.asarray(ytr))
        ds_tr = _criar_dataset(textos_arr[tr].tolist(), ytr, tokenizer)
        ds_va = _criar_dataset(textos_arr[va].tolist(), yva, tokenizer)

        args = TrainingArguments(**{**TRAIN_PARAMS, "output_dir": str(out)})

        if loss == "focal":
            alpha = alpha_por_frequencia_inversa(np.asarray(ytr), len(NOMES_CLASSES))
            TrainerCls = _focal_trainer(alpha, gamma=focal_gamma)
        else:
            TrainerCls = _weighted_trainer(weights)

        trainer = TrainerCls(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            compute_metrics=_compute_metrics(),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        m = trainer.evaluate()
        fold_f1.append(float(m.get("eval_f1_macro", 0.0)))
        fold_acc.append(float(m.get("eval_accuracy", 0.0)))
        logger.info(
            "Fold %d/%d — F1-macro=%.4f | Acurácia=%.4f",
            k, n_splits, fold_f1[-1], fold_acc[-1],
        )

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1, ddof=1))
    se = std_f1 / sqrt(n_splits)
    try:
        from scipy.stats import t as t_dist

        lo, hi = t_dist.interval(0.95, df=n_splits - 1, loc=mean_f1, scale=se)
    except ImportError:
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n_splits - 1, 2.0)
        lo, hi = mean_f1 - t_crit * se, mean_f1 + t_crit * se

    return {
        "campo": campo,
        "loss": loss,
        "fold_scores": fold_f1,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_95": (float(lo), float(hi)),
        "mean_acc": float(np.mean(fold_acc)),
        "usar_lora": usar_lora,
    }
