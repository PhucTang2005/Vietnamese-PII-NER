"""
train.py — Unified training script for all 4 NER model types.

Usage examples::

    # Train PhoBERT (load pre-tokenized data)
    python -m src.train --model_type phobert --data_dir data/tokenized_phobert

    # Train XLM-R (tokenize from scratch)
    python -m src.train --model_type xlmr --tokenize_from_scratch

    # Train BiLSTM-CRF with custom hyperparameters
    python -m src.train --model_type bilstm-crf --epochs 20 --lr 1e-3

Supported model types: bilstm, bilstm-crf, phobert, xlmr
"""

from src.utils import print_detailed_report
import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DefaultDataCollator,
    Trainer,
    TrainingArguments,
)

from src.config import (
    BILSTM_CHECKPOINT_DIR,
    BILSTM_CRF_CHECKPOINT_DIR,
    HF_DATASET_NAME,
    ID2LABEL,
    LABEL2ID,
    MODEL_TYPES,
    NUM_LABELS,
    PHOBERT_CHECKPOINT_DIR,
    PHOBERT_MODEL_NAME,
    TRAINING_CONFIG,
    VNCORENLP_DIR,
    XLMR_CHECKPOINT_DIR,
    XLMR_MODEL_NAME,
    get_tokenized_data_dir,
)
from src.data_loader import (
    init_segmenter,
    load_raw_dataset,
    load_tokenized_dataset,
    save_tokenized_dataset,
    tokenize_and_align_phobert,
    tokenize_and_align_xlmr,
)
from src.models import BiLSTMCRFForNER, BiLSTMForNER
from src.utils import compute_metrics_crf, compute_metrics_standard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model / Tokenizer Factory
# ==============================================================================

def _build_tokenizer(model_type: str):
    """Create the appropriate tokenizer for the given model type.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".

    Returns:
        A HuggingFace tokenizer instance.
    """
    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        return AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
    else:
        return AutoTokenizer.from_pretrained(XLMR_MODEL_NAME)


def _build_model(model_type: str, tokenizer, device: torch.device):
    """Create the appropriate model for the given model type.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        tokenizer:  Tokenizer instance (needed for vocab_size and padding_idx).
        device:     Torch device to place model on.

    Returns:
        A model instance moved to the specified device.
    """
    if model_type == "bilstm":
        model = BiLSTMForNER(
            vocab_size=tokenizer.vocab_size,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            padding_idx=tokenizer.pad_token_id,
        )
    elif model_type == "bilstm-crf":
        model = BiLSTMCRFForNER(
            vocab_size=tokenizer.vocab_size,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            padding_idx=tokenizer.pad_token_id,
        )
    elif model_type == "phobert":
        model = AutoModelForTokenClassification.from_pretrained(
            PHOBERT_MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    elif model_type == "xlmr":
        model = AutoModelForTokenClassification.from_pretrained(
            XLMR_MODEL_NAME,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "%s model initialized — %s parameters",
        model_type.upper(), f"{param_count:,}",
    )
    return model


# ==============================================================================
# Data Collator Factory
# ==============================================================================

def _build_data_collator(model_type: str, tokenizer):
    """Create the appropriate data collator.

    BiLSTM / BiLSTM-CRF use DefaultDataCollator (custom nn.Module).
    Transformer models use DataCollatorForTokenClassification.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        tokenizer:  Tokenizer instance.

    Returns:
        A data collator instance.
    """
    if model_type in ("bilstm", "bilstm-crf"):
        return DefaultDataCollator()
    else:
        return DataCollatorForTokenClassification(tokenizer=tokenizer)


# ==============================================================================
# Tokenization from Scratch
# ==============================================================================

def _tokenize_dataset(model_type: str, tokenizer, args):
    """Tokenize the raw HuggingFace dataset from scratch.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        tokenizer:  Tokenizer instance.
        args:       Parsed CLI arguments.

    Returns:
        Tokenized DatasetDict.
    """
    raw_dataset = load_raw_dataset(args.dataset_name)
    max_length = TRAINING_CONFIG[model_type]["max_length"]

    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        vncorenlp_dir = args.vncorenlp_dir or str(VNCORENLP_DIR)
        segmenter = init_segmenter(vncorenlp_dir)

        tokenize_fn = partial(
            tokenize_and_align_phobert,
            tokenizer=tokenizer,
            segmenter=segmenter,
            label2id=LABEL2ID,
            max_length=max_length,
        )
        desc = f"Tokenizing ({model_type}, PhoBERT tokenizer)"
    else:
        tokenize_fn = partial(
            tokenize_and_align_xlmr,
            tokenizer=tokenizer,
            label2id=LABEL2ID,
            max_length=max_length,
        )
        desc = "Tokenizing (XLM-R)"

    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=32,
        remove_columns=raw_dataset["train"].column_names,
        desc=desc,
    )

    # Save for future reuse
    if args.save_tokenized:
        save_tokenized_dataset(tokenized_dataset, model_type)

    return tokenized_dataset


# ==============================================================================
# Training Arguments Builder
# ==============================================================================

def _build_training_args(model_type: str, args) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config and CLI overrides.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        args:       Parsed CLI arguments.

    Returns:
        TrainingArguments instance.
    """
    cfg = TRAINING_CONFIG[model_type]

    output_dir = args.output_dir or str(Path(f"./{model_type}-ner-pii"))

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr if args.lr is not None else cfg["learning_rate"],
        per_device_train_batch_size=args.batch_size if args.batch_size is not None else cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        num_train_epochs=args.epochs if args.epochs is not None else cfg["num_train_epochs"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        logging_steps=cfg["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        report_to="none",
    )

    return training_args


# ==============================================================================
# Compute Metrics Factory
# ==============================================================================

def _build_compute_metrics(model_type: str, model, device: torch.device):
    """Build the appropriate compute_metrics function.

    For BiLSTM-CRF, uses functools.partial to bind the model, id2label,
    and device into compute_metrics_crf.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        model:      The model instance (needed for CRF decode).
        device:     Torch device.

    Returns:
        A callable with signature ``compute_metrics(eval_preds) -> dict``.
    """
    if model_type == "bilstm-crf":
        return partial(
            compute_metrics_crf,
            model=model,
            id2label=ID2LABEL,
            device=device,
        )
    else:
        return partial(
            compute_metrics_standard,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )


# ==============================================================================
# Save Best Model
# ==============================================================================

def _save_best_model(model_type: str, model, tokenizer, args):
    """Save the best model after training.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        model:      The trained model.
        tokenizer:  Tokenizer instance.
        args:       Parsed CLI arguments.
    """
    if args.save_dir:
        save_path = Path(args.save_dir)
    else:
        save_paths = {
            "bilstm": BILSTM_CHECKPOINT_DIR,
            "bilstm-crf": BILSTM_CRF_CHECKPOINT_DIR,
            "phobert": PHOBERT_CHECKPOINT_DIR,
            "xlmr": XLMR_CHECKPOINT_DIR,
        }
        save_path = save_paths[model_type]

    save_path.mkdir(parents=True, exist_ok=True)

    if model_type in ("bilstm", "bilstm-crf"):
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))
    else:
        trainer.save_model(str(save_path))
        tokenizer.save_pretrained(str(save_path))

    logger.info("Best model saved to %s", save_path)


# ==============================================================================
# Main Training Function
# ==============================================================================

def train(args):
    """Main training entry point.

    Args:
        args: Parsed CLI arguments from argparse.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Model type: %s", args.model_type)

    # 1. Tokenizer
    tokenizer = _build_tokenizer(args.model_type)
    logger.info("Tokenizer loaded — vocab size: %d", tokenizer.vocab_size)

    # 2. Dataset
    if args.tokenize_from_scratch:
        logger.info("Tokenizing dataset from scratch...")
        tokenized_dataset = _tokenize_dataset(args.model_type, tokenizer, args)
    else:
        tokenized_dataset = load_tokenized_dataset(
            model_type=args.model_type,
            data_dir=args.data_dir,
        )
    logger.info("Dataset: %s", tokenized_dataset)

    # 3. Model
    model = _build_model(args.model_type, tokenizer, device)

    # 4. Training components
    training_args = _build_training_args(args.model_type, args)
    data_collator = _build_data_collator(args.model_type, tokenizer)
    compute_metrics_fn = _build_compute_metrics(args.model_type, model, device)

    # 5. Trainer
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics_fn,
    }

    # Transformer models benefit from having processing_class set
    if args.model_type in ("phobert", "xlmr"):
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    # 6. Train
    logger.info("Starting training %s...", args.model_type.upper())
    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # 7. Final evaluation
    eval_results = trainer.evaluate()
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("NER Entity-Level (Micro):")
    logger.info("  Precision: %.4f", eval_results.get("eval_precision", 0))
    logger.info("  Recall:    %.4f", eval_results.get("eval_recall", 0))
    logger.info("  F1:        %.4f", eval_results.get("eval_f1", 0))
    logger.info("Classification (Has Entity):")
    logger.info("  Accuracy:  %.4f", eval_results.get("eval_cls_accuracy", 0))
    logger.info("  Precision: %.4f", eval_results.get("eval_cls_precision", 0))
    logger.info("  Recall:    %.4f", eval_results.get("eval_cls_recall", 0))
    logger.info("  F1:        %.4f", eval_results.get("eval_cls_f1", 0))
    logger.info("=" * 60)

    # 8. Detailed report
    if args.detailed_report:
        eval_output = trainer.predict(tokenized_dataset["validation"])
        if args.model_type == "bilstm-crf":
            # Reuse compute_metrics_crf logic để decode
            emissions = torch.tensor(eval_output.predictions, dtype=torch.float32)
            mask = torch.tensor(eval_output.label_ids != -100, dtype=torch.bool)
            mask[:, 0] = True
            with torch.no_grad():
                crf_preds = model.crf.decode(emissions.to(device), mask=mask.to(device))
            # Flatten thành array giống argmax output
            max_len = eval_output.predictions.shape[1]
            predictions = np.zeros((len(crf_preds), max_len), dtype=int)
            for i, seq in enumerate(crf_preds):
                predictions[i, :len(seq)] = seq
        else:
            predictions = np.argmax(eval_output.predictions, axis=-1)
        
        print_detailed_report(predictions, eval_output.label_ids, ID2LABEL)

    # 9. Save best model
    _save_best_model(args.model_type, model, tokenizer, args)

    return eval_results


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Vietnamese PII NER models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PhoBERT with pre-tokenized data
  python -m src.train --model_type phobert

  # Train XLM-R, tokenize from scratch
  python -m src.train --model_type xlmr --tokenize_from_scratch

  # Train BiLSTM-CRF with custom epochs
  python -m src.train --model_type bilstm-crf --epochs 20
        """,
    )

    # Required
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
        help="Model architecture to train.",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to pre-tokenized dataset directory. "
             "If not specified, uses default from config.",
    )
    parser.add_argument(
        "--tokenize_from_scratch",
        action="store_true",
        help="Tokenize raw dataset from HuggingFace instead of loading "
             "pre-tokenized data.",
    )
    parser.add_argument(
        "--save_tokenized",
        action="store_true",
        help="Save tokenized dataset to disk for future reuse.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=HF_DATASET_NAME,
        help=f"HuggingFace dataset name (default: {HF_DATASET_NAME}).",
    )

    # Training hyperparameters (overrides config defaults)
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Train batch size.")

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for training checkpoints and logs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the best model. If not specified, uses "
             "default from config (e.g. best_model_phobert/).",
    )

    # Misc
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--vncorenlp_dir",
        type=str,
        default=None,
        help=f"VnCoreNLP model directory (default: {VNCORENLP_DIR}).",
    )
    parser.add_argument(
        "--detailed_report",
        action="store_true",
        help="Print detailed per-entity-type classification report after training.",
    )

    return parser.parse_args()


def main():
    """Entry point for ``python -m src.train``."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
