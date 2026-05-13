"""
run_inference.py — Inference pipeline for Vietnamese PII NER models.

Supports BiLSTM, BiLSTM-CRF, PhoBERT, and XLM-RoBERTa architectures.
Handles preprocessing (VnCoreNLP segmentation if needed), tokenization,
model inference, CRF decoding (if applicable), and label alignment back
to character spans in the original text.

Usage example::

    python -m inference.run_inference \\
        --model_type xlmr \\
        --text "Tôi là Nguyễn Văn A, sdt 0912345678, ở Hà Nội"
"""

import argparse
import logging
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.config import (
    HF_PHOBERT_REPO,
    HF_XLMR_REPO,
    ID2LABEL,
    MODEL_TYPES,
    PHOBERT_MODEL_NAME,
    TRAINING_CONFIG,
    VNCORENLP_DIR,
    XLMR_MODEL_NAME,
    get_checkpoint_dir,
)
from src.data_loader import init_segmenter, segment_text
from src.models import BiLSTMCRFForNER, BiLSTMForNER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model & Tokenizer Loaders
# ==============================================================================

def load_inference_components(
    model_type: str,
    model_path: Optional[str] = None,
    vncorenlp_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple:
    """Load tokenizer, model, and segmenter for inference.

    Args:
        model_type:    One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        model_path:    Path to local checkpoint or HF Hub ID. If None,
                       uses default local paths or HF IDs based on model_type.
        vncorenlp_dir: Path to VnCoreNLP. None uses default from config.
        device:        'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        (tokenizer, model, segmenter, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    # 1. Load Segmenter
    segmenter = None
    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        v_dir = vncorenlp_dir or str(VNCORENLP_DIR)
        segmenter = init_segmenter(v_dir)

    # 2. Determine default paths
    if model_path is None:
        if model_type == "phobert":
            model_path = HF_PHOBERT_REPO
        elif model_type == "xlmr":
            model_path = HF_XLMR_REPO
        else:
            model_path = str(get_checkpoint_dir(model_type))

    logger.info("Loading tokenizer and model from %s...", model_path)

    # 3. Load Tokenizer
    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if model_type == "phobert" else PHOBERT_MODEL_NAME
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 4. Load Model
    if model_type == "bilstm":
        model = BiLSTMForNER.from_pretrained(model_path, device=device)
    elif model_type == "bilstm-crf":
        model = BiLSTMCRFForNER.from_pretrained(model_path, device=device)
    elif model_type in ("phobert", "xlmr"):
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model.to(device_obj)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    logger.info("Inference components loaded successfully on %s.", device)
    return tokenizer, model, segmenter, device_obj


# ==============================================================================
# Offset Mapping Builders (from segmented text)
# ==============================================================================

def _build_phobert_offset_mapping(tokens: List[str], segmented_text: str, special_tokens: set) -> List[Tuple[int, int]]:
    """Build offset mapping for PhoBERT tokenization on segmented text."""
    offset_mapping = []
    curr_pos = 0

    for token in tokens:
        if token in ("<s>", "</s>", "<pad>", "<unk>"):
            offset_mapping.append((0, 0))
            continue

        clean_token = token.replace("@@", "")
        while curr_pos < len(segmented_text) and segmented_text[curr_pos] == " ":
            curr_pos += 1

        if not segmented_text[curr_pos:].lower().startswith(clean_token.lower()):
            found = segmented_text.lower().find(clean_token.lower(), curr_pos)
            if found != -1:
                curr_pos = found
            else:
                offset_mapping.append((0, 0))
                continue

        start = curr_pos
        end = curr_pos + len(clean_token)
        offset_mapping.append((start, end))
        curr_pos = end

    return offset_mapping


def _align_segmented_to_original(
    offset_mapping: List[Tuple[int, int]],
    segmented_text: str,
    original_text: str,
) -> List[Tuple[int, int]]:
    """Map segmented text offsets back to original text offsets."""
    seg_to_orig = []
    orig_idx = 0

    for seg_char in segmented_text:
        if seg_char == "_" and orig_idx < len(original_text) and original_text[orig_idx] == " ":
            seg_to_orig.append(orig_idx)
            orig_idx += 1
        elif seg_char == " ":
            if orig_idx < len(original_text) and original_text[orig_idx] == " ":
                seg_to_orig.append(orig_idx)
                orig_idx += 1
            else:
                seg_to_orig.append(-1)
        else:
            if orig_idx < len(original_text):
                seg_to_orig.append(orig_idx)
                orig_idx += 1
            else:
                seg_to_orig.append(-1)

    orig_offsets = []
    for tok_start, tok_end in offset_mapping:
        if tok_start == 0 and tok_end == 0:
            orig_offsets.append((0, 0))
            continue

        valid_orig_idx = [
            seg_to_orig[i] for i in range(tok_start, min(tok_end, len(seg_to_orig)))
            if i < len(seg_to_orig) and seg_to_orig[i] >= 0
        ]

        if valid_orig_idx:
            orig_offsets.append((valid_orig_idx[0], valid_orig_idx[-1] + 1))
        else:
            orig_offsets.append((0, 0))

    return orig_offsets


# ==============================================================================
# Postprocessing: Merge BIO tags into entities
# ==============================================================================

def _merge_bio_entities(
    token_predictions: List[str],
    offset_mapping: List[Tuple[int, int]],
    original_text: str,
) -> List[Dict]:
    """Merge BIO predicted tokens into character-level entity spans.

    Args:
        token_predictions: List of BIO string tags.
        offset_mapping:    List of (start, end) tuples in original text.
        original_text:     The original raw text.

    Returns:
        List of dictionaries containing 'label', 'start', 'end', 'text'.
    """
    entities = []
    current_entity = None

    for i, (pred, (start, end)) in enumerate(zip(token_predictions, offset_mapping)):
        if start == end:  # Special token
            continue

        if pred == "O":
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
            continue

        prefix, label = pred[:1], pred[2:]

        if prefix == "B":
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {"label": label, "start": start, "end": end}
        elif prefix == "I":
            if current_entity is not None and current_entity["label"] == label:
                current_entity["end"] = end
            else:
                # I- tag without B- tag → treat as B-
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {"label": label, "start": start, "end": end}

    if current_entity is not None:
        entities.append(current_entity)

    # Extract text from spans and trim whitespaces
    results = []
    for ent in entities:
        text_span = original_text[ent["start"]:ent["end"]]
        
        # Trim leading/trailing whitespace
        l_strip = len(text_span) - len(text_span.lstrip())
        r_strip = len(text_span) - len(text_span.rstrip())
        
        start = ent["start"] + l_strip
        end = ent["end"] - r_strip
        if start < end:
            results.append({
                "label": ent["label"],
                "start": start,
                "end": end,
                "text": original_text[start:end],
            })

    return results


# ==============================================================================
# Inference Pipeline
# ==============================================================================

def predict_entities(
    text: str,
    model_type: str,
    model: torch.nn.Module,
    tokenizer,
    segmenter=None,
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """Run inference on a single text string.

    Args:
        text:       The raw input text.
        model_type: One of "bilstm", "bilstm-crf", "phobert", "xlmr".
        model:      Loaded model instance.
        tokenizer:  Loaded tokenizer.
        segmenter:  VnCoreNLP segmenter (required if not XLM-R).
        device:     Torch device.

    Returns:
        List of detected entity dictionaries.
    """
    if device is None:
        device = next(model.parameters()).device

    if not text.strip():
        return []

    # 1. Preprocess & Tokenize
    if model_type in ("bilstm", "bilstm-crf", "phobert"):
        segmented = segment_text(text, segmenter)
        max_length = TRAINING_CONFIG[model_type]["max_length"]
        encoding = tokenizer(segmented, return_tensors="pt", truncation=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Build offset mapping from segmented back to original
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        seg_offsets = _build_phobert_offset_mapping(
    tokens, segmented,
    special_tokens=set(tokenizer.all_special_tokens))
        orig_offsets = _align_segmented_to_original(seg_offsets, segmented, text)

    else:
        # XLM-R
        max_length = TRAINING_CONFIG[model_type]["max_length"]
        encoding = tokenizer(
            text, return_tensors="pt", truncation=True, return_offsets_mapping=True
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        orig_offsets = encoding["offset_mapping"][0].tolist()

    # 2. Forward Pass
    with torch.no_grad():
        if model_type == "bilstm-crf":
            outputs = model(input_ids, attention_mask=attention_mask)
            emissions = outputs["logits"]
            mask = attention_mask.bool()
            crf_preds = model.crf.decode(emissions, mask=mask)
            preds = crf_preds[0]
        else:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            preds = torch.argmax(logits, dim=-1)[0].tolist()

    # 3. Decode BIO Labels
    token_predictions = [ID2LABEL[p] for p in preds]

    # Handle sequence truncation mismatch
    if len(token_predictions) > len(orig_offsets):
        token_predictions = token_predictions[:len(orig_offsets)]
    elif len(token_predictions) < len(orig_offsets):
        orig_offsets = orig_offsets[:len(token_predictions)]

    # 4. Merge
    entities = _merge_bio_entities(token_predictions, orig_offsets, text)
    return entities


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vietnamese PII NER Inference")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=MODEL_TYPES,
        help="Model architecture."
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Input text to process."
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to checkpoint or HF Hub repo ID. Uses default if None."
    )
    parser.add_argument(
        "--vncorenlp_dir", type=str, default=None,
        help="Path to VnCoreNLP models directory."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')."
    )

    args = parser.parse_args()

    tokenizer, model, segmenter, device = load_inference_components(
        model_type=args.model_type,
        model_path=args.model_path,
        vncorenlp_dir=args.vncorenlp_dir,
        device=args.device,
    )

    entities = predict_entities(
        text=args.text,
        model_type=args.model_type,
        model=model,
        tokenizer=tokenizer,
        segmenter=segmenter,
        device=device,
    )

    print(f"\n--- INFERENCE RESULTS ({args.model_type}) ---")
    print(f"Input: {args.text}\n")
    if not entities:
        print("No entities found.")
    else:
        for ent in entities:
            print(f"[{ent['label']}] {ent['text']} (pos: {ent['start']}-{ent['end']})")


if __name__ == "__main__":
    main()
