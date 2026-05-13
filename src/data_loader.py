"""
data_loader.py — Data loading and tokenization pipelines for Vietnamese PII NER.

Contains two distinct pipelines:
    1. PhoBERT pipeline (also used by BiLSTM / BiLSTM-CRF):
       - VnCoreNLP word segmentation → PhoBERT tokenizer → custom offset mapping
    2. XLM-R pipeline:
       - Direct tokenization with native ``return_offsets_mapping=True``

Both pipelines convert character-level entity spans (``privacy_mask``) into
token-level BIO labels suitable for NER training.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, load_dataset, load_from_disk

from src.config import HF_DATASET_NAME, LABEL2ID, get_tokenized_data_dir

logger = logging.getLogger(__name__)


# ==============================================================================
# VnCoreNLP Segmenter
# ==============================================================================

def init_segmenter(vncorenlp_dir: str):
    """Initialize VnCoreNLP word segmenter.

    Downloads the model if not already present.

    Args:
        vncorenlp_dir: Directory containing (or to download) VnCoreNLP models.

    Returns:
        A VnCoreNLP segmenter instance.
    """
    import py_vncorenlp

    vncorenlp_path = Path(vncorenlp_dir)
    vncorenlp_path.mkdir(parents=True, exist_ok=True)

    py_vncorenlp.download_model(save_dir=str(vncorenlp_path))
    segmenter = py_vncorenlp.VnCoreNLP(save_dir=str(vncorenlp_path))

    logger.info("VnCoreNLP segmenter loaded from %s", vncorenlp_path)
    return segmenter


def segment_text(text: str, segmenter) -> str:
    """Word-segment Vietnamese text using VnCoreNLP.

    Args:
        text:       Raw Vietnamese text.
        segmenter:  A VnCoreNLP instance.

    Returns:
        Segmented text where multi-syllable words are joined by underscores.
    """
    try:
        sentences = segmenter.word_segment(text)
        return " ".join(sentences)
    except Exception as e:
        logger.warning(
            "VnCoreNLP segmentation failed for text %r: %s. Falling back to raw text.",
            text[:50], e
        )
        return text


# ==============================================================================
# Character-to-Token Label Alignment
# ==============================================================================

def _build_char_labels(source_text: str, privacy_mask: List[Dict]) -> List[str]:
    """Build character-level BIO label array from entity spans.

    Args:
        source_text:  The original text.
        privacy_mask: List of entity dicts with 'start', 'end', 'label' keys.

    Returns:
        List of BIO label strings, one per character in source_text.
    """
    char_labels = ["O"] * len(source_text)
    for ent in privacy_mask:
        start, end, label = ent["start"], ent["end"], ent["label"]
        if start >= len(source_text) or end > len(source_text):
            continue
        char_labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            char_labels[i] = f"I-{label}"
    return char_labels


def _build_seg_to_orig_map(source_text: str, segmented_text: str) -> List[int]:
    """Build a mapping from segmented text positions to original text positions.

    VnCoreNLP replaces spaces within multi-syllable words with '_' and may
    insert extra spaces between words. This function aligns the two.

    Args:
        source_text:    The original (unsegmented) text.
        segmented_text: The word-segmented text from VnCoreNLP.

    Returns:
        List where seg_to_orig[seg_idx] = original char index, or -1 if
        the segmented char has no corresponding original position.
    """
    seg_to_orig = []
    orig_idx = 0

    for seg_char in segmented_text:
        if seg_char == "_" and orig_idx < len(source_text) and source_text[orig_idx] == " ":
            seg_to_orig.append(orig_idx)
            orig_idx += 1
        elif seg_char == " ":
            if orig_idx < len(source_text) and source_text[orig_idx] == " ":
                seg_to_orig.append(orig_idx)
                orig_idx += 1
            else:
                seg_to_orig.append(-1)
        else:
            if orig_idx < len(source_text):
                seg_to_orig.append(orig_idx)
                orig_idx += 1
            else:
                seg_to_orig.append(-1)

    return seg_to_orig


def char_to_token_labels(
    source_text: str,
    privacy_mask: List[Dict],
    offset_mapping: List[tuple],
    segmented_text: str,
    label2id: Dict[str, int],
) -> List[int]:
    """Convert character-level entity spans to token-level BIO label ids.

    Uses offset_mapping (positions in the *segmented* text) and a seg-to-orig
    mapping to trace each token back to the original text's char labels.

    Args:
        source_text:    The original (unsegmented) text.
        privacy_mask:   List of entity dicts with 'start', 'end', 'label'.
        offset_mapping: List of (start, end) tuples in the segmented text space.
        segmented_text: The word-segmented text from VnCoreNLP.
        label2id:       Mapping from BIO label string to integer id.

    Returns:
        List of integer label ids, one per token. Special tokens get -100.
    """
    char_labels = _build_char_labels(source_text, privacy_mask)
    seg_to_orig = _build_seg_to_orig_map(source_text, segmented_text)

    token_labels = []

    for tok_start, tok_end in offset_mapping:
        # Special tokens (<s>, </s>, <pad>)
        if tok_start == 0 and tok_end == 0:
            token_labels.append(-100)
            continue

        # Collect original char positions covered by this token
        orig_positions = []
        for seg_pos in range(tok_start, min(tok_end, len(seg_to_orig))):
            if seg_pos < len(seg_to_orig) and seg_to_orig[seg_pos] >= 0:
                orig_positions.append(seg_to_orig[seg_pos])

        if not orig_positions:
            token_labels.append(label2id["O"])
            continue

        # Use the label of the first original character of this token
        first_orig_pos = orig_positions[0]
        if first_orig_pos < len(char_labels):
            lbl = char_labels[first_orig_pos]
        else:
            lbl = "O"

        token_labels.append(label2id.get(lbl, label2id["O"]))

    return token_labels


# ==============================================================================
# PhoBERT Offset Mapping Builder
# ==============================================================================

def _build_phobert_offset_mapping(
    input_ids: List[int],
    segmented_text: str,
    tokenizer,
) -> List[tuple]:
    """Build offset_mapping manually for PhoBERT tokenizer.

    PhoBERT's tokenizer does not support ``return_offsets_mapping=True``,
    so we reconstruct it by walking through the segmented text and matching
    each token's clean form (with '@@' subword markers stripped).

    Args:
        input_ids:      Token ids from the tokenizer.
        segmented_text: The word-segmented text.
        tokenizer:      The PhoBERT tokenizer instance.

    Returns:
        List of (start, end) tuples in the segmented text coordinate space.
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    offset_mapping = []
    curr_pos = 0

    for token in tokens:
        # Special tokens → (0, 0)
        if token in tokenizer.all_special_tokens:
            offset_mapping.append((0, 0))
            continue

        # Strip PhoBERT subword marker '@@'
        clean_token = token.replace("@@", "")

        # Skip spaces
        while curr_pos < len(segmented_text) and segmented_text[curr_pos] == " ":
            curr_pos += 1

        # Verify token matches at curr_pos, fallback search if not
        if not segmented_text[curr_pos:].lower().startswith(clean_token.lower()):
            found = segmented_text.lower().find(clean_token.lower(), curr_pos)
            if found != -1:
                curr_pos = found
            else:
                # Cannot align — mark as special token, do not advance
                logger.warning("Token %r not found in segmented text at pos %d", clean_token, curr_pos)
                offset_mapping.append((0, 0))
                continue

        start = curr_pos
        end = curr_pos + len(clean_token)
        offset_mapping.append((start, end))
        curr_pos = end

    return offset_mapping


# ==============================================================================
# PhoBERT / BiLSTM Tokenization Pipeline
# ==============================================================================

def tokenize_and_align_phobert(
    examples: Dict[str, Any],
    tokenizer,
    segmenter,
    label2id: Dict[str, int],
    max_length: int = 256,
) -> Dict[str, List]:
    """Tokenize and align labels for PhoBERT / BiLSTM / BiLSTM-CRF.

    Pipeline:
        1. Word-segment with VnCoreNLP
        2. Tokenize segmented text with PhoBERT tokenizer (no native offset_mapping)
        3. Build offset_mapping manually
        4. Align char-level entity spans to token-level BIO labels

    This function is designed to be used with ``dataset.map(batched=True)``
    via ``functools.partial``::

        from functools import partial
        tokenize_fn = partial(
            tokenize_and_align_phobert,
            tokenizer=tokenizer, segmenter=segmenter,
            label2id=label2id, max_length=256,
        )
        tokenized = dataset.map(tokenize_fn, batched=True, ...)

    Args:
        examples:   Batch dict with 'source_text' and 'privacy_mask' columns.
        tokenizer:  PhoBERT tokenizer instance.
        segmenter:  VnCoreNLP segmenter instance.
        label2id:   Mapping from BIO label string to integer id.
        max_length: Maximum sequence length for tokenizer.

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels' lists.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    # Note: intentionally processes one example at a time because VnCoreNLP
    # segmentation does not support batching.
    for source_text, privacy_mask in zip(
        examples["source_text"], examples["privacy_mask"]
    ):
        # Step 1: Word segmentation
        segmented = segment_text(source_text, segmenter)

        # Step 2: Tokenize (no return_offsets_mapping for PhoBERT)
        encoding = tokenizer(
            segmented,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )

        # Step 3: Build offset mapping manually
        offset_mapping = _build_phobert_offset_mapping(
            encoding["input_ids"], segmented, tokenizer
        )

        # Step 4: Align labels
        token_labels = char_to_token_labels(
            source_text, privacy_mask, offset_mapping, segmented, label2id
        )

        # Pad/truncate labels to max_length
        while len(token_labels) < max_length:
            token_labels.append(-100)
        token_labels = token_labels[:max_length]

        all_input_ids.append(encoding["input_ids"])
        all_attention_mask.append(encoding["attention_mask"])
        all_labels.append(token_labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


# ==============================================================================
# XLM-R Tokenization Pipeline
# ==============================================================================

def tokenize_and_align_xlmr(
    examples: Dict[str, Any],
    tokenizer,
    label2id: Dict[str, int],
    max_length: int = 512,
) -> Dict[str, List]:
    """Tokenize and align labels for XLM-RoBERTa.

    XLM-R supports ``return_offsets_mapping=True`` natively, so we can directly
    use it to map character positions to token positions without VnCoreNLP.

    This function is designed to be used with ``dataset.map(batched=True)``
    via ``functools.partial``::

        from functools import partial
        tokenize_fn = partial(
            tokenize_and_align_xlmr,
            tokenizer=tokenizer, label2id=label2id, max_length=512,
        )
        tokenized = dataset.map(tokenize_fn, batched=True, ...)

    Args:
        examples:   Batch dict with 'source_text' and 'privacy_mask' columns.
        tokenizer:  XLM-R tokenizer instance.
        label2id:   Mapping from BIO label string to integer id.
        max_length: Maximum sequence length for tokenizer.

    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels' lists.
    """
    # XLM-R supports return_offsets_mapping natively
    encoding = tokenizer(
        examples["source_text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )

    all_labels = []
    for i, privacy_mask in enumerate(examples["privacy_mask"]):
        source_text = examples["source_text"][i]
        offset_mapping = encoding["offset_mapping"][i]

        # Build char-level labels
        char_labels = _build_char_labels(source_text, privacy_mask)

        # Map char → token
        token_labels = []
        for tok_start, tok_end in offset_mapping:
            if tok_start == 0 and tok_end == 0:
                token_labels.append(-100)
            else:
                lbl = char_labels[tok_start] if tok_start < len(char_labels) else "O"
                token_labels.append(label2id.get(lbl, label2id["O"]))

        all_labels.append(token_labels)

    encoding["labels"] = all_labels
    # Remove offset_mapping — model does not accept it as input
    encoding.pop("offset_mapping")

    return encoding


# ==============================================================================
# Dataset Loading Utilities
# ==============================================================================

def load_raw_dataset(dataset_name: str = HF_DATASET_NAME) -> DatasetDict:
    """Load the raw dataset from HuggingFace Hub.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        DatasetDict with 'train' and 'validation' splits containing
        'source_text', 'language', and 'privacy_mask' columns.
    """
    dataset = load_dataset(dataset_name)
    logger.info("Loaded dataset: %s", dataset)
    return dataset


def load_tokenized_dataset(
    model_type: str,
    data_dir: Optional[str] = None,
) -> DatasetDict:
    """Load pre-tokenized dataset from disk.

    Automatically resolves the correct directory based on model_type:
        - bilstm / bilstm-crf / phobert → data/tokenized_phobert/
        - xlmr → data/tokenized_xlmr/

    Args:
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        data_dir:   Optional override for the tokenized data directory.
                    If None, uses the default from config.

    Returns:
        DatasetDict with 'train' and 'validation' splits containing
        'input_ids', 'attention_mask', and 'labels' columns.

    Raises:
        FileNotFoundError: If the tokenized data directory does not exist.
    """
    if data_dir is not None:
        data_path = Path(data_dir)
    else:
        data_path = get_tokenized_data_dir(model_type)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Tokenized data not found at {data_path}. "
            f"Run tokenization first or specify a valid --data_dir."
        )

    dataset = load_from_disk(str(data_path))
    logger.info("Loaded tokenized dataset from %s: %s", data_path, dataset)
    return dataset


def save_tokenized_dataset(
    dataset: DatasetDict,
    model_type: str,
    data_dir: Optional[str] = None,
) -> Path:
    """Save tokenized dataset to disk.

    Args:
        dataset:    The tokenized DatasetDict to save.
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        data_dir:   Optional override for the save directory.

    Returns:
        Path where the dataset was saved.
    """
    if data_dir is not None:
        save_path = Path(data_dir)
    else:
        save_path = get_tokenized_data_dir(model_type)

    save_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(save_path))
    logger.info("Saved tokenized dataset to %s", save_path)
    return save_path
