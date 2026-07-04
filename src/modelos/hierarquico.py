"""
Modelo hierárquico para documentos longos (votos TCU).

Duas estratégias:
  1. Head+Tail — concatena início e fim do documento numa única sequência BERT.
  2. HierarchicalBert — encoder por sentença + attention sobre sentenças.

Ambas usam pesos de classe (cross-entropy ponderada).
"""

from __future__ import annotations

import logging
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from .pesos import ID2LABEL, LABEL2ID, NOMES_CLASSES, pesos_tensor

logger = logging.getLogger(__name__)

MODEL_NAME = "dominguesm/legal-bert-base-cased-ptbr"
MODEL_FALLBACK = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42


def _carregar_encoder(model_name: str | None = None):
    from transformers import AutoModel, AutoTokenizer

    for nome in [model_name or MODEL_NAME, MODEL_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(nome)
            encoder = AutoModel.from_pretrained(nome)
            logger.info("Encoder carregado: %s", nome)
            return tokenizer, encoder
        except Exception as exc:
            logger.warning("Falha ao carregar %s: %s", nome, exc)
    raise RuntimeError("Nenhum encoder disponível.")


# ---------------------------------------------------------------------------
# 1. Head+Tail tokenization + standard BERT classifier
# ---------------------------------------------------------------------------


def tokenizar_head_tail(
    textos: list[str], tokenizer, head_tokens: int = 256, tail_tokens: int = 254
) -> dict:
    """
    Tokeniza com estratégia head+tail: [CLS] head... [SEP] ...tail [SEP].

    Total max = head_tokens + tail_tokens + 2 (special tokens) = 512.
    Documentos curtos são tratados normalmente (padding).
    """
    max_length = head_tokens + tail_tokens + 2
    all_input_ids = []
    all_attention_mask = []

    for texto in textos:
        encoded = tokenizer.encode(texto, add_special_tokens=False)
        if len(encoded) <= max_length - 2:
            tokens = [tokenizer.cls_token_id] + encoded + [tokenizer.sep_token_id]
        else:
            head = encoded[:head_tokens]
            tail = encoded[-tail_tokens:]
            tokens = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id] + tail + [tokenizer.sep_token_id]
            tokens = tokens[:max_length]

        attn = [1] * len(tokens)
        pad_len = max_length - len(tokens)
        tokens += [tokenizer.pad_token_id] * pad_len
        attn += [0] * pad_len

        all_input_ids.append(tokens)
        all_attention_mask.append(attn)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
    }


def kfold_head_tail(
    df: pd.DataFrame,
    campo: str = "VOTO_LIMPO",
    n_splits: int = 5,
    head_tokens: int = 256,
    tail_tokens: int = 254,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    usar_lora: bool = True,
) -> dict:
    """
    K-Fold com LegalBert + LoRA usando tokenização head+tail.

    Cada documento é representado por [CLS] primeiros_N_tokens [SEP] últimos_M_tokens [SEP],
    garantindo que informação do final do voto (pré-dispositivo) seja capturada.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    labels_arr = df["LABEL"].values
    textos_arr = df[campo].fillna("").values
    yid = np.array([LABEL2ID[l] for l in labels_arr])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_f1, fold_acc = [], []
    max_length = head_tokens + tail_tokens + 2

    for k, (tr, va) in enumerate(skf.split(textos_arr, yid), start=1):
        logger.info("=== Head+Tail Fold %d/%d ===", k, n_splits)
        out = Path(f"/tmp/ht_fold_{k}")
        out.mkdir(parents=True, exist_ok=True)

        tokenizer, model = _carregar_classificador(max_length)
        if usar_lora:
            model = _aplicar_lora(model)

        ytr = yid[tr].tolist()
        yva = yid[va].tolist()
        weights = pesos_tensor(np.asarray(ytr))

        enc_tr = tokenizar_head_tail(textos_arr[tr].tolist(), tokenizer, head_tokens, tail_tokens)
        enc_tr["labels"] = ytr
        ds_tr = Dataset.from_dict(enc_tr)

        enc_va = tokenizar_head_tail(textos_arr[va].tolist(), tokenizer, head_tokens, tail_tokens)
        enc_va["labels"] = yva
        ds_va = Dataset.from_dict(enc_va)

        args = TrainingArguments(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(batch_size, 16),
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.10,
            lr_scheduler_type="linear",
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=not usar_lora,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            seed=RANDOM_STATE,
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            report_to="none",
        )

        WeightedTrainer = _weighted_trainer_cls(weights)
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            compute_metrics=_compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        m = trainer.evaluate()
        fold_f1.append(float(m.get("eval_f1_macro", 0.0)))
        fold_acc.append(float(m.get("eval_accuracy", 0.0)))
        logger.info("Fold %d — F1=%.4f | Acc=%.4f", k, fold_f1[-1], fold_acc[-1])

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _resumo_kfold(fold_f1, fold_acc, n_splits, campo, "head_tail", usar_lora)


# ---------------------------------------------------------------------------
# 2. Hierarchical BERT — encoder por sentença + attention sobre sentenças
# ---------------------------------------------------------------------------


class HierarchicalBert(nn.Module):
    """
    Modelo hierárquico: encoder BERT por sentença + attention ponderada
    sobre representações de sentenças + classificador linear.
    """

    def __init__(self, encoder, hidden_size: int, num_labels: int = 3):
        super().__init__()
        self.encoder = encoder
        self.sent_attn_w = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sent_input_ids, sent_attention_mask, sent_counts):
        """
        sent_input_ids: (batch, max_sents, max_sent_len)
        sent_attention_mask: (batch, max_sents, max_sent_len)
        sent_counts: (batch,) — número real de sentenças por documento
        """
        B, S, L = sent_input_ids.shape
        device = sent_input_ids.device

        flat_ids = sent_input_ids.view(B * S, L)
        flat_mask = sent_attention_mask.view(B * S, L)

        out = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls_emb = out.last_hidden_state[:, 0, :]  # (B*S, H)
        cls_emb = cls_emb.view(B, S, -1)  # (B, S, H)

        scores = self.sent_attn_w(cls_emb).squeeze(-1)  # (B, S)
        sent_mask = torch.arange(S, device=device).unsqueeze(0) < sent_counts.unsqueeze(1)
        scores = scores.masked_fill(~sent_mask, -1e9)
        weights = torch.softmax(scores, dim=1)  # (B, S)

        doc_repr = (weights.unsqueeze(-1) * cls_emb).sum(dim=1)  # (B, H)
        doc_repr = self.dropout(doc_repr)
        return self.classifier(doc_repr)


def segmentar_sentencas(texto: str, max_sents: int = 48) -> list[str]:
    """Segmenta texto em sentenças usando heurísticas para texto jurídico."""
    import re

    if not texto or not texto.strip():
        return [""]

    sents = re.split(r'(?<=[.;!?])\s+|\n+', texto)
    sents = [s.strip() for s in sents if s.strip()]

    if not sents:
        return [texto[:500]]

    if len(sents) > max_sents:
        step = len(sents) / max_sents
        sents = [sents[int(i * step)] for i in range(max_sents)]

    return sents


def _criar_dataset_hierarquico(
    textos: list[str],
    labels: list[int],
    tokenizer,
    max_sents: int = 48,
    max_sent_len: int = 128,
) -> dict:
    """Tokeniza em duas dimensões: (n_docs, max_sents, max_sent_len)."""
    all_input_ids = []
    all_attention_mask = []
    all_sent_counts = []

    for texto in textos:
        sents = segmentar_sentencas(texto, max_sents)
        n_sents = len(sents)
        all_sent_counts.append(n_sents)

        doc_ids = []
        doc_mask = []
        for sent in sents:
            enc = tokenizer(
                sent, truncation=True, padding="max_length",
                max_length=max_sent_len, return_tensors="np",
            )
            doc_ids.append(enc["input_ids"][0].tolist())
            doc_mask.append(enc["attention_mask"][0].tolist())

        pad_sent = [tokenizer.pad_token_id] + [tokenizer.pad_token_id] * (max_sent_len - 1)
        pad_mask = [0] * max_sent_len
        while len(doc_ids) < max_sents:
            doc_ids.append(pad_sent)
            doc_mask.append(pad_mask)

        all_input_ids.append(doc_ids)
        all_attention_mask.append(doc_mask)

    return {
        "sent_input_ids": all_input_ids,
        "sent_attention_mask": all_attention_mask,
        "sent_counts": all_sent_counts,
        "labels": labels,
    }


def kfold_hierarquico(
    df: pd.DataFrame,
    campo: str = "VOTO_LIMPO",
    n_splits: int = 5,
    max_sents: int = 48,
    max_sent_len: int = 128,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    freeze_encoder_epochs: int = 1,
) -> dict:
    """
    K-Fold com modelo hierárquico (encoder por sentença + attention).

    Estratégia de treino:
      - Primeiras `freeze_encoder_epochs` épocas: encoder congelado, só treina
        attention + classifier (warm-up das camadas novas).
      - Épocas restantes: fine-tune completo com lr reduzido para o encoder.
    """
    from torch.utils.data import DataLoader, TensorDataset

    labels_arr = df["LABEL"].values
    textos_arr = df[campo].fillna("").values
    yid = np.array([LABEL2ID[l] for l in labels_arr])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_f1, fold_acc = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for k, (tr, va) in enumerate(skf.split(textos_arr, yid), start=1):
        logger.info("=== Hierárquico Fold %d/%d ===", k, n_splits)

        tokenizer, encoder = _carregar_encoder()
        hidden_size = encoder.config.hidden_size
        model = HierarchicalBert(encoder, hidden_size, num_labels=len(NOMES_CLASSES))
        model.to(device)

        ytr = yid[tr].tolist()
        yva = yid[va].tolist()
        weights = torch.tensor(pesos_tensor(np.asarray(ytr)), device=device)

        logger.info("Tokenizando treino (%d docs)...", len(tr))
        data_tr = _criar_dataset_hierarquico(
            textos_arr[tr].tolist(), ytr, tokenizer, max_sents, max_sent_len
        )
        logger.info("Tokenizando validação (%d docs)...", len(va))
        data_va = _criar_dataset_hierarquico(
            textos_arr[va].tolist(), yva, tokenizer, max_sents, max_sent_len
        )

        ds_tr = TensorDataset(
            torch.tensor(data_tr["sent_input_ids"], dtype=torch.long),
            torch.tensor(data_tr["sent_attention_mask"], dtype=torch.long),
            torch.tensor(data_tr["sent_counts"], dtype=torch.long),
            torch.tensor(data_tr["labels"], dtype=torch.long),
        )
        ds_va = TensorDataset(
            torch.tensor(data_va["sent_input_ids"], dtype=torch.long),
            torch.tensor(data_va["sent_attention_mask"], dtype=torch.long),
            torch.tensor(data_va["sent_counts"], dtype=torch.long),
            torch.tensor(data_va["labels"], dtype=torch.long),
        )

        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        dl_va = DataLoader(ds_va, batch_size=batch_size * 2)

        best_f1 = 0.0
        patience, patience_limit = 0, 2

        for epoch in range(1, epochs + 1):
            if epoch <= freeze_encoder_epochs:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=learning_rate * 5,
                )
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {"params": model.encoder.parameters(), "lr": learning_rate * 0.1},
                    {"params": model.sent_attn_w.parameters(), "lr": learning_rate},
                    {"params": model.classifier.parameters(), "lr": learning_rate},
                    {"params": model.dropout.parameters(), "lr": learning_rate},
                ], weight_decay=0.01)

            model.train()
            total_loss = 0.0
            for batch in dl_tr:
                ids, mask, counts, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()
                logits = model(ids, mask, counts)
                loss = F.cross_entropy(logits, labels, weight=weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in dl_va:
                    ids, mask, counts, labels = [b.to(device) for b in batch]
                    logits = model(ids, mask, counts)
                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            acc = accuracy_score(all_labels, all_preds)
            logger.info(
                "  Epoch %d/%d — loss=%.4f | val_F1=%.4f | val_acc=%.4f",
                epoch, epochs, total_loss / len(dl_tr), f1, acc,
            )

            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit and epoch > freeze_encoder_epochs:
                    logger.info("  Early stopping na época %d", epoch)
                    break

        fold_f1.append(best_f1)
        fold_acc.append(best_acc)
        logger.info("Fold %d — best F1=%.4f | Acc=%.4f", k, best_f1, best_acc)

        del model, encoder, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _resumo_kfold(fold_f1, fold_acc, n_splits, campo, "hierarquico", False)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _carregar_classificador(max_length: int = 512):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for nome in [MODEL_NAME, MODEL_FALLBACK]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(nome)
            model = AutoModelForSequenceClassification.from_pretrained(
                nome,
                num_labels=len(NOMES_CLASSES),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            std = getattr(model.config, "initializer_range", 0.02)
            try:
                dense = model.bert.pooler.dense
                torch.nn.init.normal_(dense.weight, mean=0.0, std=std)
                torch.nn.init.zeros_(dense.bias)
            except AttributeError:
                pass
            logger.info("Classificador carregado: %s (max_length=%d)", nome, max_length)
            return tokenizer, model
        except Exception as exc:
            logger.warning("Falha: %s: %s", nome, exc)
    raise RuntimeError("Nenhum modelo base disponível.")


def _aplicar_lora(model):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, config)
    for name, param in model.named_parameters():
        if "pooler" in name:
            param.requires_grad = True
    model.print_trainable_parameters()
    return model


def _compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "f1_macro": float(f1_score(label_ids, preds, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(label_ids, preds)),
    }


def _weighted_trainer_cls(class_weights):
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.cross_entropy(
                logits, labels,
                weight=torch.as_tensor(class_weights).to(logits.device),
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _resumo_kfold(fold_f1, fold_acc, n_splits, campo, estrategia, usar_lora):
    mean_f1 = float(np.mean(fold_f1))
    std_f1 = float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0
    se = std_f1 / sqrt(n_splits) if n_splits > 1 else 0.0
    try:
        from scipy.stats import t as t_dist
        lo, hi = t_dist.interval(0.95, df=n_splits - 1, loc=mean_f1, scale=se)
    except (ImportError, ValueError):
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(n_splits - 1, 2.0)
        lo, hi = mean_f1 - t_crit * se, mean_f1 + t_crit * se

    return {
        "campo": campo,
        "estrategia": estrategia,
        "fold_scores": fold_f1,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "ci_95": (float(lo), float(hi)),
        "mean_acc": float(np.mean(fold_acc)),
        "usar_lora": usar_lora,
    }
