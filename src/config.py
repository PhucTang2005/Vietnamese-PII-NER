"""
config.py — Central configuration for the Vietnamese PII Detection project.

Contains:
    - BIO label schema (54 entity types → 109 labels)
    - Model-specific hyperparameters (BiLSTM, BiLSTM-CRF, PhoBERT, XLM-R)
    - File paths and HuggingFace Hub repository IDs
"""

from pathlib import Path
from typing import Dict, List, Tuple

# ==============================================================================
# Project Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TOKENIZED_PHOBERT_DIR = DATA_DIR / "tokenized_phobert"
TOKENIZED_XLMR_DIR = DATA_DIR / "tokenized_xlmr"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BILSTM_CHECKPOINT_DIR = PROJECT_ROOT / "best_model_bilstm"
BILSTM_CRF_CHECKPOINT_DIR = PROJECT_ROOT / "best_model_bilstm-crf"
PHOBERT_CHECKPOINT_DIR = PROJECT_ROOT / "best_model_phobert"
XLMR_CHECKPOINT_DIR = PROJECT_ROOT / "best_model_xlmr"

VNCORENLP_DIR = PROJECT_ROOT / "vncorenlp"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

# ==============================================================================
# HuggingFace Hub
# ==============================================================================

HF_DATASET_NAME = "quynong/cs419-data"
HF_PHOBERT_REPO = "Phuc2005/pii-phobert-base-ner"
HF_XLMR_REPO = "Phuc2005/pii-xlm-r-base-ner"

# ==============================================================================
# Pretrained Model Names
# ==============================================================================

PHOBERT_MODEL_NAME = "vinai/phobert-base"
XLMR_MODEL_NAME = "xlm-roberta-base"

# ==============================================================================
# BIO Label Schema
# ==============================================================================

ENTITY_TYPES: List[str] = [
    "ACCOUNTNAME",
    "ACCOUNTNUMBER",
    "AGE",
    "AMOUNT",
    "BIC",
    "BITCOINADDRESS",
    "BUILDINGNUMBER",
    "CCCD",
    "CITY",
    "COMPANYNAME",
    "COUNTY",
    "CREDITCARDCVV",
    "CREDITCARDISSUER",
    "CREDITCARDNUMBER",
    "CURRENCY",
    "CURRENCYCODE",
    "CURRENCYNAME",
    "CURRENCYSYMBOL",
    "DATE",
    "DOB",
    "EMAIL",
    "ETHEREUMADDRESS",
    "EYECOLOR",
    "FIRSTNAME",
    "GENDER",
    "HEIGHT",
    "IBAN",
    "IPADDRESS",
    "JOBAREA",
    "JOBTITLE",
    "JOBTYPE",
    "LASTNAME",
    "LITECOINADDRESS",
    "MAC",
    "MASKEDNUMBER",
    "MIDDLENAME",
    "NEARBYGPSCOORDINATE",
    "ORDINALDIRECTION",
    "PASSWORD",
    "PHONEIMEI",
    "PHONENUMBER",
    "PIN",
    "PREFIX",
    "SECONDARYADDRESS",
    "SEX",
    "STATE",
    "STREET",
    "TIME",
    "URL",
    "USERAGENT",
    "USERNAME",
    "VEHICLEVIN",
    "VEHICLEVRM",
    "ZIPCODE",
]


def build_bio_labels() -> Tuple[List[str], Dict[str, int], Dict[int, str], int]:
    """Build BIO label schema from the predefined entity types.

    Returns:
        label_list: Ordered list of BIO labels, e.g. ["O", "B-ACCOUNTNAME", "I-ACCOUNTNAME", ...].
        label2id:   Mapping from label string to integer id.
        id2label:   Mapping from integer id to label string.
        num_labels: Total number of labels (109 = 1 + 54*2).
    """
    label_list = ["O"]
    for entity_type in ENTITY_TYPES:
        label_list.append(f"B-{entity_type}")
        label_list.append(f"I-{entity_type}")

    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for idx, label in enumerate(label_list)}
    num_labels = len(label_list)

    return label_list, label2id, id2label, num_labels


# Pre-compute label mappings for convenience (importable constants)
LABEL_LIST, LABEL2ID, ID2LABEL, NUM_LABELS = build_bio_labels()

# ==============================================================================
# Model-Specific Hyperparameters
# ==============================================================================

# Supported model types (used as CLI argument values)
MODEL_TYPES = ("bilstm", "bilstm-crf", "phobert", "xlmr")


TRAINING_CONFIG = {
    "bilstm": {
        "max_length": 256,
        "learning_rate": 2e-3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 15,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.0,
        "logging_steps": 100,
        "fp16": True,
    },
    "bilstm-crf": {
        "max_length": 256,
        "learning_rate": 2e-3,
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "num_train_epochs": 15,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.0,
        "logging_steps": 100,
        "fp16": True,
    },
    "phobert": {
        "max_length": 256,
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "fp16": True,
    },
    "xlmr": {
        "max_length": 512,
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "logging_steps": 100,
        "fp16": True,
    },
}

# BiLSTM-specific architecture hyperparameters
BILSTM_EMB_DIM = 300
BILSTM_HIDDEN_DIM = 256
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.3


def get_tokenized_data_dir(model_type: str) -> Path:
    """Return the correct tokenized data directory for a given model type.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".

    Returns:
        Path to the tokenized data directory.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        return TOKENIZED_PHOBERT_DIR
    elif model_type == "xlmr":
        return TOKENIZED_XLMR_DIR
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be one of {MODEL_TYPES}."
        )


def get_checkpoint_dir(model_type: str) -> Path:
    """Return the default checkpoint directory for a given model type.

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".

    Returns:
        Path to the checkpoint directory.

    Raises:
        ValueError: If model_type is not recognized.
    """
    dirs = {
        "bilstm": BILSTM_CHECKPOINT_DIR,
        "bilstm-crf": BILSTM_CRF_CHECKPOINT_DIR,
        "phobert": PHOBERT_CHECKPOINT_DIR,
        "xlmr": XLMR_CHECKPOINT_DIR,
    }
    if model_type not in dirs:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be one of {MODEL_TYPES}."
        )
    return dirs[model_type]
