"""
models.py — Custom model architectures for Vietnamese PII NER.

Contains:
    - BiLSTMForNER:     Embedding → BiLSTM → Dropout → Linear + CrossEntropyLoss
    - BiLSTMCRFForNER:  Embedding → BiLSTM → Dropout → Linear + CRF layer

Both classes follow HuggingFace Trainer conventions:
    - forward() returns a dict with 'loss' and 'logits' keys
    - A DummyConfig with id2label/label2id is attached for compatibility
    - save_pretrained() / from_pretrained() for checkpoint persistence

Transformer models (PhoBERT, XLM-R) use AutoModelForTokenClassification directly
and do NOT need custom classes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torchcrf import CRF
from src.config import (
    BILSTM_DROPOUT,
    BILSTM_EMB_DIM,
    BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS,
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
)

logger = logging.getLogger(__name__)


def _resolve_checkpoint_dir(checkpoint_dir: str) -> Path:
    """Return a local checkpoint directory, downloading HF Hub repos if needed."""
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.exists():
        return ckpt_path

    if "/" in checkpoint_dir:
        downloaded_path = snapshot_download(repo_id=checkpoint_dir)
        return Path(downloaded_path)

    return ckpt_path


# ==============================================================================
# Config Wrapper
# ==============================================================================

class DummyConfig:
    """Minimal config object to satisfy HuggingFace Trainer's expectation
    of ``model.config.id2label`` and ``model.config.label2id``.
    """

    def __init__(
        self,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        **kwargs,
    ):
        self.id2label = id2label
        self.label2id = label2id
        # Store any extra kwargs for serialization
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        """Serialize config to a JSON-compatible dictionary."""
        return {
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "label2id": self.label2id,
        }


# ==============================================================================
# BiLSTM for NER
# ==============================================================================

class BiLSTMForNER(nn.Module):
    """Bidirectional LSTM model for token classification (NER).

    Architecture:
        nn.Embedding → nn.LSTM (bidirectional, 2 layers) → nn.Dropout → nn.Linear

    The model uses CrossEntropyLoss with ``ignore_index=-100`` to handle
    padding and special tokens.

    Args:
        vocab_size:  Tokenizer vocabulary size.
        num_labels:  Number of BIO labels.
        id2label:    Mapping from label id to label string.
        label2id:    Mapping from label string to label id.
        padding_idx: Token id used for padding (default: 1 for PhoBERT).
        emb_dim:     Embedding dimension (default: 300).
        hidden_dim:  LSTM hidden dimension (default: 256).
        num_layers:  Number of LSTM layers (default: 2).
        dropout:     Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        padding_idx: int = 1,
        emb_dim: int = BILSTM_EMB_DIM,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers_lstm = num_layers
        self.dropout_rate = dropout

        # Attach config for HuggingFace Trainer compatibility
        self.config = DummyConfig(id2label, label2id)

        # Layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # Loss ignores padding tokens labeled as -100
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids:      Token ids, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len). Unused by
                            LSTM but accepted for Trainer compatibility.
            labels:         Gold label ids, shape (batch, seq_len). If provided,
                            loss is computed.

        Returns:
            Dict with 'loss' (if labels given) and 'logits' keys.
        """
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embeds)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)

        loss = None
        if labels is not None:
            loss = self.loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if loss is not None:
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and configuration to a directory.

        Creates:
            - ``config.json``:   Model architecture hyperparameters + label maps.
            - ``bilstm_weights.pth``: PyTorch state dict.

        Args:
            save_dir: Directory to save into (created if needed).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_dict = {
            "model_type": "bilstm",
            "vocab_size": self.vocab_size,
            "num_labels": self.num_labels,
            "padding_idx": self.padding_idx,
            "emb_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers_lstm,
            "dropout": self.dropout_rate,
            **self.config.to_dict(),
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # Save weights
        torch.save(self.state_dict(), save_path / "bilstm_weights.pth")
        logger.info("BiLSTMForNER saved to %s", save_path)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: Optional[str] = None) -> "BiLSTMForNER":
        """Load a BiLSTMForNER model from a checkpoint directory.

        Expects:
            - ``config.json``:   Saved by save_pretrained().
            - ``bilstm_weights.pth``: Saved by save_pretrained().

        Args:
            checkpoint_dir: Directory containing config.json and weights.
            device:         Device to load model onto (e.g. 'cuda', 'cpu').
                            If None, loads to CPU.

        Returns:
            Loaded BiLSTMForNER instance.
        """
        ckpt_path = _resolve_checkpoint_dir(checkpoint_dir)

        with open(ckpt_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Reconstruct id2label with integer keys
        id2label = {int(k): v for k, v in config["id2label"].items()}
        label2id = config["label2id"]

        model = cls(
            vocab_size=config["vocab_size"],
            num_labels=config["num_labels"],
            id2label=id2label,
            label2id=label2id,
            padding_idx=config.get("padding_idx", 1),
            emb_dim=config.get("emb_dim", BILSTM_EMB_DIM),
            hidden_dim=config.get("hidden_dim", BILSTM_HIDDEN_DIM),
            num_layers=config.get("num_layers", BILSTM_NUM_LAYERS),
            dropout=config.get("dropout", BILSTM_DROPOUT),
        )

        weights_path = ckpt_path / "bilstm_weights.pth"
        map_location = device if device else "cpu"
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

        if device:
            model.to(device)

        logger.info("BiLSTMForNER loaded from %s", ckpt_path)
        return model


# ==============================================================================
# BiLSTM-CRF for NER
# ==============================================================================

class BiLSTMCRFForNER(nn.Module):
    """Bidirectional LSTM with CRF layer for token classification (NER).

    Architecture:
        nn.Embedding → nn.LSTM (bidirectional, 2 layers) → nn.Dropout
        → nn.Linear (emissions) → CRF (pytorch-crf)

    During training, the CRF computes negative log-likelihood loss.
    During inference, the CRF performs Viterbi decoding.

    Args:
        vocab_size:  Tokenizer vocabulary size.
        num_labels:  Number of BIO labels.
        id2label:    Mapping from label id to label string.
        label2id:    Mapping from label string to label id.
        padding_idx: Token id used for padding (default: 1 for PhoBERT).
        emb_dim:     Embedding dimension (default: 300).
        hidden_dim:  LSTM hidden dimension (default: 256).
        num_layers:  Number of LSTM layers (default: 2).
        dropout:     Dropout probability (default: 0.3).
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        padding_idx: int = 1,
        emb_dim: int = BILSTM_EMB_DIM,
        hidden_dim: int = BILSTM_HIDDEN_DIM,
        num_layers: int = BILSTM_NUM_LAYERS,
        dropout: float = BILSTM_DROPOUT,
    ):
        super().__init__()


        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers_lstm = num_layers
        self.dropout_rate = dropout

        # Attach config for HuggingFace Trainer compatibility
        self.config = DummyConfig(id2label, label2id)

        # Shared layers (identical to BiLSTMForNER)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # CRF layer (replaces CrossEntropyLoss)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids:      Token ids, shape (batch, seq_len).
            attention_mask: Attention mask, shape (batch, seq_len).
                            Used as the CRF mask during training.
            labels:         Gold label ids, shape (batch, seq_len). If provided,
                            CRF negative log-likelihood loss is computed.
                            Values of -100 are replaced with 0 (O label) before
                            passing to CRF, since CRF does not support -100.

        Returns:
            Dict with 'loss' (if labels given) and 'logits' (emissions) keys.
        """
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.bilstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)

        loss = None
        if labels is not None:
            if attention_mask is not None:
                valid_label_mask = (labels != -100)
                crf_mask = attention_mask.bool().clone()
                crf_mask[:, 1:] = crf_mask[:, 1:] & valid_label_mask[:, 1:]
            else:

                valid_label_mask = (labels != -100)
                crf_mask = valid_label_mask.clone()
                crf_mask[:, 0] = True

 
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0

            loss = -self.crf(emissions, tags=safe_labels, mask=crf_mask, reduction="mean")

        if loss is not None:
            return {"loss": loss, "logits": emissions}
        return {"logits": emissions}

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and configuration to a directory.

        Creates:
            - ``config.json``:          Model architecture hyperparameters + label maps.
            - ``bilstmcrf_weights.pth``: PyTorch state dict (includes CRF params).

        Args:
            save_dir: Directory to save into (created if needed).
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "model_type": "bilstm-crf",
            "vocab_size": self.vocab_size,
            "num_labels": self.num_labels,
            "padding_idx": self.padding_idx,
            "emb_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers_lstm,
            "dropout": self.dropout_rate,
            **self.config.to_dict(),
        }
        with open(save_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        torch.save(self.state_dict(), save_path / "bilstmcrf_weights.pth")
        logger.info("BiLSTMCRFForNER saved to %s", save_path)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: Optional[str] = None) -> "BiLSTMCRFForNER":
        """Load a BiLSTMCRFForNER model from a checkpoint directory.

        Expects:
            - ``config.json``:          Saved by save_pretrained().
            - ``bilstmcrf_weights.pth``: Saved by save_pretrained().

        Args:
            checkpoint_dir: Directory containing config.json and weights.
            device:         Device to load model onto (e.g. 'cuda', 'cpu').
                            If None, loads to CPU.

        Returns:
            Loaded BiLSTMCRFForNER instance.
            
        Note:
            The model is returned in training mode. Call ``model.eval()``
            before running inference to disable dropout.
        """
        ckpt_path = _resolve_checkpoint_dir(checkpoint_dir)

        with open(ckpt_path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        id2label = {int(k): v for k, v in config["id2label"].items()}
        label2id = config["label2id"]

        model = cls(
            vocab_size=config["vocab_size"],
            num_labels=config["num_labels"],
            id2label=id2label,
            label2id=label2id,
            padding_idx=config.get("padding_idx", 1),
            emb_dim=config.get("emb_dim", BILSTM_EMB_DIM),
            hidden_dim=config.get("hidden_dim", BILSTM_HIDDEN_DIM),
            num_layers=config.get("num_layers", BILSTM_NUM_LAYERS),
            dropout=config.get("dropout", BILSTM_DROPOUT),
        )

        weights_path = ckpt_path / "bilstmcrf_weights.pth"
        map_location = device if device else "cpu"
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

        if device:
            model.to(device)

        logger.info("BiLSTMCRFForNER loaded from %s", ckpt_path)
        return model
