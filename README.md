# 🔍 Vietnamese PII Detection — Named Entity Recognition

A modular NER system for detecting **54 types of Personally Identifiable Information (PII)** in Vietnamese text. The project supports four model architectures trained and evaluated on the [`quynong/cs419-data`](https://huggingface.co/datasets/quynong/cs419-data) dataset.

## Problem Statement

Personally Identifiable Information (PII) — such as names, phone numbers, email addresses, national IDs, and bank accounts — is embedded in unstructured Vietnamese text across documents, chat logs, and web content. Automatically detecting and classifying these entities is critical for data privacy compliance, anonymization pipelines, and secure data handling.

This project frames PII detection as a **token-level Named Entity Recognition (NER)** task using the **BIO tagging scheme** (109 labels: `O` + 54 entity types × `B-`/`I-` prefixes).

## Pipeline Overview

```
Raw Vietnamese Text
        │
        ▼
┌─────────────────────┐
│   Tokenization      │  PhoBERT tokenizer (BiLSTM, BiLSTM-CRF, PhoBERT)
│                     │  XLM-R tokenizer   (XLM-RoBERTa)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Token-Level NER    │  BiLSTM → Linear → argmax
│  Model Inference    │  BiLSTM → Linear → CRF Viterbi decode
│                     │  PhoBERT / XLM-R → Linear → argmax
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  BIO Tag Merging    │  Merge B-/I- spans → character-level entities
└─────────┬───────────┘
          ▼
  List of PII Entities
  [{"label": "PHONE", "text": "0912345678", "start": 35, "end": 45}]
```

## Model Performance

Evaluated on the validation split of `quynong/cs419-data`:

| Model | Precision | Recall | **F1** | Classification F1 |
|---|---|---|---|---|
| BiLSTM | 0.9161 | 0.9329 | 0.9244 | 0.9990 |
| BiLSTM-CRF | 0.9439 | 0.9475 | 0.9457 | 0.9991 |
| PhoBERT-base | 0.9551 | 0.9630 | 0.9590 | 1.0000 |
| **XLM-RoBERTa** | **0.9579** | **0.9645** | **0.9612** | **1.0000** |

## HuggingFace Models

| Model | HuggingFace Hub |
|---|---|
| PhoBERT-base | [`Phuc2005/pii-phobert-base-ner`](https://huggingface.co/Phuc2005/pii-phobert-base-ner) |
| XLM-RoBERTa | [`Phuc2005/pii-xlm-r-base-ner`](https://huggingface.co/Phuc2005/pii-xlm-r-base-ner) |
| BiLSTM | [`Phuc2005/pii-bilstm-ner`](https://huggingface.co/Phuc2005/pii-bilstm-ner) |
| BiLSTM-CRF | [`Phuc2005/pii-bilstm-crf-ner`](https://huggingface.co/Phuc2005/pii-bilstm-crf-ner) |

## Dataset

[`quynong/cs419-data`](https://huggingface.co/datasets/quynong/cs419-data) — A Vietnamese PII corpus with 54 entity types, annotated at the character level.

| Split | # Samples | % Has Entity |
|---|---|---|
| Train | 54,117 | 55.8% |
| Validation | 6,014 | 55.9% |

- **109 BIO labels**: `O` + `B-`/`I-` for 54 PII types (person names, national IDs, bank accounts, addresses, emails, IPs, etc.)
- **Average length**: ~40 tokens/sentence — truncation at `MAX_LENGTH=256` is virtually non-existent (< 0.02%)
- **~44% of samples contain no entity**, supporting evaluation at both the token level and the sentence level

---

## 🚀 Quick Start — Try Without Installing

Try the live demo on HuggingFace Spaces — no setup required:

👉 **[Open Demo](https://huggingface.co/spaces/Phuc2005/pii-demo-ner)**

1. Enter Vietnamese text in the input box
2. Click **"⚡ Trích xuất PII"**
3. Detected PII entities will be highlighted with labels

> If the Space is slow on first visit (~30s), the server is cold-starting — wait and retry.

---

## 📓 Training on Google Colab (Recommended)

Each notebook is self-contained: it clones the repo, installs dependencies, tokenizes data, trains the model, evaluates, and runs inference — all in one file.

Click the badge to open directly in Google Colab:

| Model | Notebook | Open in Colab |
|---|---|---|
| **XLM-RoBERTa** | `finetune_xlm_r_base_ner.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhucTang2005/Vietnamese-PII-NER/blob/main/notebooks/finetune_xlm_r_base_ner.ipynb) |
| **PhoBERT-base** | `finetune_phobert_ner.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhucTang2005/Vietnamese-PII-NER/blob/main/notebooks/finetune_phobert_ner.ipynb) |
| **BiLSTM** | `train_bilstm_ner.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhucTang2005/Vietnamese-PII-NER/blob/main/notebooks/train_bilstm_ner.ipynb) |
| **BiLSTM-CRF** | `train_bilstm_crf_ner.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhucTang2005/Vietnamese-PII-NER/blob/main/notebooks/train_bilstm_crf_ner.ipynb) |

### How to Use the Notebooks

Each notebook follows the same structure with numbered sections:

| Section | What It Does | Cells to Run |
|---|---|---|
| **§1 Setup** | Clones the repo, installs dependencies | Run once — Colab may restart |
| **§2 Load Dataset** | Loads `quynong/cs419-data` from HuggingFace | Run once |
| **§3 Tokenization** | Tokenizes raw text, aligns BIO labels | Run once |
| **§4 Model Definition** | Initializes the model architecture | Run once |
| **§5 Training** | Trains the model (GPU recommended) | Run once (takes 10–40 min) |
| **§6 Evaluation** | Computes NER F1, precision, recall | Run once |
| **§7 Inference** | Loads the pretrained model from HF Hub and runs on sample text | **Run this section alone for quick testing** |
| **§8 Save & Export** | Saves trained model to checkpoint directory | Optional |

> **💡 HuggingFace Token (Recommended):** [`quynong/cs419-data`](https://huggingface.co/datasets/quynong/cs419-data) is a public dataset but may occasionally prompt for authentication due to rate limits. To avoid this, add a token before running the notebooks on Google Colab:
> 1. Create a [HuggingFace access token](https://huggingface.co/settings/tokens)
> 2. In Colab, go to **🔑 Secrets** (left sidebar) → add a secret named `HF_TOKEN` with your token value
> 3. Toggle the **"Notebook access"** switch ON
> The notebook will automatically read this secret and log in before downloading the dataset.

> **💡 Quick Inference Only:** If you just want to test a pretrained model without training, run **§1 Setup** → **§2 Load Dataset** (to build `id2label`) → skip to **§7 Inference**. The inference cell downloads the model from HuggingFace Hub automatically.

---

## 🖥️ Local Development

### Installation

```bash
git clone https://github.com/PhucTang2005/Vietnamese-PII-NER.git
cd Vietnamese-PII-NER
pip install -r requirements.txt
```

### Inference via CLI

```bash
# XLM-RoBERTa (recommended — no Java required)
python -m inference.run_inference --model_type xlmr --text "Vui lòng chuyển 500.000 VNĐ vào STK 123456789 của Lê Văn B."
```

### Training via CLI

```bash
# Train XLM-RoBERTa (tokenize from scratch)
python -m src.train --model_type xlmr --tokenize_from_scratch --save_tokenized

# Train PhoBERT
python -m src.train --model_type phobert --epochs 5 --lr 5e-5

# Train BiLSTM-CRF
python -m src.train --model_type bilstm-crf --epochs 15 --batch_size 32
```

### Gradio Web Demo (Local)

```bash
python app.py
# Opens at http://127.0.0.1:7860
```

---

## ⚠️ Note on Word Segmentation (VnCoreNLP)

The released PhoBERT checkpoint was trained using a **VnCoreNLP-based Vietnamese word segmentation pipeline** to align with PhoBERT's pretraining conventions.

For deployment simplicity and improved portability, the current public training/inference scripts were later simplified to **remove the Java/VnCoreNLP dependency**. All models (BiLSTM, BiLSTM-CRF, PhoBERT) now tokenize raw text directly using the PhoBERT tokenizer without prior word segmentation.

While the tokenizer-only pipeline may introduce minor performance differences, it **significantly improves reproducibility and ease of use** across local environments and Google Colab.

> **XLM-RoBERTa** has never required word segmentation — it processes raw text natively via SentencePiece and consistently achieves the highest F1 score.

---

## Repository Structure

```
Vietnamese-PII-NER/
├── src/
│   ├── config.py           # Hyperparameters, BIO labels, paths
│   ├── data_loader.py      # Tokenization & label alignment
│   ├── models.py           # BiLSTM and BiLSTM-CRF architectures
│   ├── train.py            # Unified training script (HF Trainer API)
│   └── utils.py            # Metrics (seqeval, classification, CRF decode)
├── inference/
│   └── run_inference.py    # CLI inference pipeline
├── notebooks/
│   ├── finetune_xlm_r_base_ner.ipynb
│   ├── finetune_phobert_ner.ipynb
│   ├── train_bilstm_ner.ipynb
│   └── train_bilstm_crf_ner.ipynb
├── app.py                  # Gradio web demo (XLM-RoBERTa)
└── requirements.txt
```

---

## License

This project is for academic and research purposes.

## Acknowledgments

- Dataset: [`quynong/cs419-data`](https://huggingface.co/datasets/quynong/cs419-data)
- PhoBERT: [VinAI Research](https://github.com/VinAIResearch/PhoBERT)
- XLM-RoBERTa: [HuggingFace](https://huggingface.co/xlm-roberta-base)
