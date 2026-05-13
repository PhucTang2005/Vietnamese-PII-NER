"""
utils.py — Shared evaluation utilities for Vietnamese PII Detection.

Contains:
    - compute_metrics_standard: NER + classification metrics via argmax
      (used by BiLSTM, PhoBERT, XLM-RoBERTa)
    - compute_metrics_crf: NER + classification metrics via CRF Viterbi decode
      (used by BiLSTM-CRF — bind with functools.partial before passing to Trainer)
"""

from typing import Dict, Tuple

# pyrefly: ignore [missing-import]
import numpy as np
import torch
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
)


# ==============================================================================
# Internal Helpers
# ==============================================================================

def _extract_ner_tags(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[list, list]:
    """Convert prediction and label arrays into lists of BIO tag sequences.

    Filters out positions where the gold label is -100 (special tokens / padding).

    Args:
        predictions: Array of predicted label ids, shape (batch, seq_len).
        labels:      Array of gold label ids, shape (batch, seq_len).
        id2label:    Mapping from integer id to BIO label string.

    Returns:
        true_labels:      List of lists of gold BIO tag strings.
        true_predictions: List of lists of predicted BIO tag strings.
    """
    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        pred_tags = []
        true_tags = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            pred_tags.append(id2label[int(pred_id)])
            true_tags.append(id2label[int(label_id)])
        true_predictions.append(pred_tags)
        true_labels.append(true_tags)

    return true_labels, true_predictions


def _compute_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    o_label_id: int,
) -> Dict[str, float]:
    """Compute sentence-level binary classification metrics.

    A sentence is considered "has entity" if any non-O, non-padding token exists.

    Args:
        predictions: Array of predicted label ids, shape (batch, seq_len).
        labels:      Array of gold label ids, shape (batch, seq_len).
        o_label_id:  Integer id of the "O" (outside) label.

    Returns:
        Dictionary with cls_accuracy, cls_precision, cls_recall, cls_f1.
    """
    cls_true = []
    cls_pred = []

    for pred_seq, label_seq in zip(predictions, labels):
        has_entity_true = any(
            l not in (-100, o_label_id) for l in label_seq
        )
        has_entity_pred = any(
            p != o_label_id
            for p, l in zip(pred_seq, label_seq)
            if l != -100
        )
        cls_true.append(int(has_entity_true))
        cls_pred.append(int(has_entity_pred))

    cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
        cls_true, cls_pred, average="binary", zero_division=0
    )
    cls_accuracy = accuracy_score(cls_true, cls_pred)

    return {
        "cls_accuracy": cls_accuracy,
        "cls_precision": cls_precision,
        "cls_recall": cls_recall,
        "cls_f1": cls_f1,
    }


def _compute_ner_metrics(
    true_labels: list,
    true_predictions: list,
) -> Dict[str, float]:
    """Compute entity-level NER metrics using seqeval (micro average).

    Args:
        true_labels:      List of lists of gold BIO tag strings.
        true_predictions: List of lists of predicted BIO tag strings.

    Returns:
        Dictionary with precision, recall, f1.
    """
    return {
        "precision": precision_score(true_labels, true_predictions, average="micro"),
        "recall": recall_score(true_labels, true_predictions, average="micro"),
        "f1": f1_score(true_labels, true_predictions, average="micro"),
    }


# ==============================================================================
# Public API — Standard Metrics (BiLSTM, PhoBERT, XLM-R)
# ==============================================================================

def compute_metrics_standard(
    eval_preds: Tuple[np.ndarray, np.ndarray],
    id2label: Dict[int, str],
    label2id: Dict[str, int],
) -> Dict[str, float]:
    """Compute NER entity-level and sentence-level classification metrics.

    Uses argmax over logits to obtain predicted labels. Suitable for models
    that output raw logits (BiLSTM, PhoBERT, XLM-RoBERTa).

    This function should be wrapped with functools.partial before passing to
    the HuggingFace Trainer::

        from functools import partial
        compute_fn = partial(compute_metrics_standard, id2label=id2label, label2id=label2id)
        trainer = Trainer(..., compute_metrics=compute_fn)

    Args:
        eval_preds: Tuple of (logits, labels) as numpy arrays.
                    logits shape: (batch, seq_len, num_labels)
                    labels shape: (batch, seq_len)
        id2label:   Mapping from integer id to BIO label string.
        label2id:   Mapping from BIO label string to integer id.

    Returns:
        Dictionary containing:
            - precision, recall, f1 (NER entity-level, micro)
            - cls_accuracy, cls_precision, cls_recall, cls_f1 (sentence-level)
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # NER entity-level metrics
    true_labels, true_predictions = _extract_ner_tags(predictions, labels, id2label)
    ner_metrics = _compute_ner_metrics(true_labels, true_predictions)

    # Sentence-level classification metrics
    o_label_id = label2id["O"]
    cls_metrics = _compute_classification_metrics(predictions, labels, o_label_id)

    return {**ner_metrics, **cls_metrics}


# ==============================================================================
# Public API — CRF Metrics (BiLSTM-CRF)
# ==============================================================================

def compute_metrics_crf(
    eval_preds: Tuple[np.ndarray, np.ndarray],
    model: torch.nn.Module,
    id2label: Dict[int, str],
    device: torch.device,
) -> Dict[str, float]:
    """Compute NER and classification metrics using CRF Viterbi decoding.

    Instead of argmax, this function uses the CRF layer's decode method to
    obtain the optimal label sequence. The CRF layer is accessed via
    ``model.crf.decode()``.

    This function should be wrapped with functools.partial before passing to
    the HuggingFace Trainer::

        from functools import partial
        compute_fn = partial(
            compute_metrics_crf,
            model=model,
            id2label=id2label,
            device=device,
        )
        trainer = Trainer(..., compute_metrics=compute_fn)

    Args:
        eval_preds: Tuple of (emissions, labels) as numpy arrays.
                    emissions shape: (batch, seq_len, num_labels)
                    labels shape:    (batch, seq_len)
        model:      The BiLSTM-CRF model instance (must have a .crf attribute).
        id2label:   Mapping from integer id to BIO label string.
        device:     Torch device to run CRF decoding on.

    Returns:
        Dictionary containing:
            - precision, recall, f1 (NER entity-level, micro)
            - cls_accuracy, cls_precision, cls_recall, cls_f1 (sentence-level)
    """
    logits, labels = eval_preds

    # CRF Viterbi decode
    emissions_tensor = torch.tensor(logits, dtype=torch.float32).to(device)
    # Use a full-ones mask to decode all positions (avoid timestep-0 CRF error)
    mask_tensor = torch.tensor(
        labels != -100,
        dtype=torch.bool,
        device=device,
    )

    with torch.no_grad():
        crf_predictions = model.crf.decode(emissions_tensor, mask=mask_tensor)

    # Build NER tag sequences (filter -100 positions using gold labels)
    true_labels = []
    true_predictions = []
    cls_true = []
    cls_pred = []

    for i in range(len(labels)):
        pred_tags = []
        true_tags = []

        pred_idx = 0
        for j in range(len(labels[i])):
            if labels[i][j] != -100:
                pred_tags.append(id2label[crf_predictions[i][pred_idx]])
                true_tags.append(id2label[int(labels[i][j])])
                pred_idx += 1

        true_labels.append(true_tags)
        true_predictions.append(pred_tags)

        # Sentence-level classification
        has_entity_true = any(tag != "O" for tag in true_tags)
        has_entity_pred = any(tag != "O" for tag in pred_tags)
        cls_true.append(int(has_entity_true))
        cls_pred.append(int(has_entity_pred))

    # NER entity-level metrics
    ner_metrics = _compute_ner_metrics(true_labels, true_predictions)

    # Sentence-level classification metrics
    cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
        cls_true, cls_pred, average="binary", zero_division=0
    )
    cls_accuracy = accuracy_score(cls_true, cls_pred)

    cls_metrics = {
        "cls_accuracy": cls_accuracy,
        "cls_precision": cls_precision,
        "cls_recall": cls_recall,
        "cls_f1": cls_f1,
    }

    return {**ner_metrics, **cls_metrics}


# ==============================================================================
# Detailed Report (for final evaluation)
# ==============================================================================

def print_detailed_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
) -> str:
    """Generate and print a detailed per-entity-type classification report.

    Args:
        predictions: Array of predicted label ids, shape (batch, seq_len).
        labels:      Array of gold label ids, shape (batch, seq_len).
        id2label:    Mapping from integer id to BIO label string.

    Returns:
        The classification report string (also printed to stdout).
    """
    true_labels, true_predictions = _extract_ner_tags(predictions, labels, id2label)
    report = classification_report(true_labels, true_predictions, digits=4)
    print("Detailed NER Classification Report (per entity type):")
    print(report)
    return report
