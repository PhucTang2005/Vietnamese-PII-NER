# Vietnamese PII Detection (Nhận diện Dữ liệu Cá nhân Tiếng Việt)

Dự án phát triển các mô hình nhận dạng thực thể có tên (NER) chuyên biệt để phát hiện 54 loại dữ liệu cá nhân (PII) trong văn bản tiếng Việt. Dự án hỗ trợ 4 kiến trúc mô hình khác nhau: **BiLSTM**, **BiLSTM-CRF**, **PhoBERT-base**, và **XLM-RoBERTa**.

## Hiệu năng mô hình

Kết quả thử nghiệm trên tập test (`quynong/cs419-data`):

| Model | NER Precision | NER Recall | NER F1 | Classification F1 |
|---|---|---|---|---|
| **BiLSTM** | 0.9161 | 0.9329 | 0.9244 | 0.9990 |
| **BiLSTM-CRF** | 0.9439 | 0.9475 | 0.9457 | 0.9991 |
| **PhoBERT-base** | 0.9551 | 0.9630 | 0.9590 | 1.0000 |
| **XLM-RoBERTa** | 0.9579 | 0.9645 | **0.9612** | **1.0000** |

---

## Section 1: Quick Start (End User)

Truy cập **HuggingFace Space** bên dưới để thử nghiệm trực tiếp trên trình duyệt,
không cần cài đặt bất cứ thứ gì:

👉 **[Mở Demo tại đây](https://huggingface.co/spaces/Phuc2005/pii-demo-ner)**

**Hướng dẫn sử dụng:**
1. Nhập đoạn văn bản tiếng Việt vào ô **"Văn bản đầu vào"**
2. Nhấn nút **"Trích xuất PII"**
3. Kết quả sẽ hiển thị các thực thể được tô màu kèm nhãn phân loại

> Nếu lần đầu truy cập Space bị chậm (~30 giây), đây là do server đang khởi động lại — chờ thêm một chút rồi thử lại.

---

## Section 2: Developer Guide

Phần này dành cho các nhà phát triển muốn clone repository, huấn luyện lại mô hình, hoặc tinh chỉnh code.

### 2.1 Cài đặt

1. Clone repository:
```bash
git clone https://github.com/your-username/pii-ner-vietnamese.git
cd pii-ner-vietnamese
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 2.2 Huấn luyện mô hình (Training)

Repository hỗ trợ huấn luyện 4 mô hình thông qua một script duy nhất `src/train.py`.

**Cách 1: Train XLM-RoBERTa (Tokenize dữ liệu từ đầu)**
```bash
python -m src.train --model_type xlmr --tokenize_from_scratch --save_tokenized
```

**Cách 2: Train PhoBERT (Ghi đè siêu tham số)**
```bash
python -m src.train --model_type phobert --epochs 10 --lr 3e-5
```

**Cách 3: Train BiLSTM-CRF**
```bash
python -m src.train --model_type bilstm-crf --batch_size 32
```

### 2.3 Chạy Suy luận (Inference)

Sử dụng script `inference/run_inference.py` để chạy dự đoán trên văn bản thô. Pipeline này sẽ tự động xử lý các bước như word segmentation (nếu cần), offset mapping, và CRF decoding (nếu dùng mô hình CRF).
> ⚠️ PhoBERT, BiLSTM, BiLSTM-CRF yêu cầu Java JDK 11+.
> Khuyến nghị chạy trên Google Colab (xem mục 2.4).

```bash
# Dùng XLM-R (tải trực tiếp từ HuggingFace Hub)
python -m inference.run_inference \
    --model_type xlmr \
    --text "Vui lòng chuyển 500.000 VNĐ vào STK 123456789 của Lê Văn B."

# Dùng BiLSTM-CRF (từ local checkpoint)
python -m inference.run_inference \
    --model_type bilstm-crf \
    --model_path ./checkpoints/best_model_bilstm-crf \
    --text "Tài khoản email của tôi là example@gmail.com"
```

### 2.4 Hướng dẫn chạy trên Google Colab (Khuyên dùng)

Google Colab là môi trường lý tưởng nhất để chạy và huấn luyện các mô hình này vì đã được cài đặt sẵn Linux và Java.

1. **Mở một notebook mới trên Google Colab** (chọn runtime GPU nếu muốn train).
2. **Chạy các lệnh sau trong một cell:**

```bash
# Clone repository
!git clone https://github.com/your-username/pii-ner-vietnamese.git
%cd pii-ner-vietnamese

# Cài đặt thư viện
!pip install -r requirements.txt
```

3. **Chạy thử mô hình PhoBERT (hoặc bất kỳ mô hình nào):**

```bash
!python -m inference.run_inference \
    --model_type phobert \
    --text "Xin chào, tôi là Nguyễn Văn An, số điện thoại 0912345678."
```

### 2.5 Các vấn đề gặp phải trên Windows local

Lưu ý (Dành cho mô hình PhoBERT, BiLSTM và BiLSTM-CRF): Cả 3 mô hình này đều dùng chung pipeline tiền xử lý và yêu cầu thư viện `py-vncorenlp` để phân mảnh từ. Bạn cần cài đặt Java (JDK 11+) và thiết lập biến môi trường `JAVA_HOME` nếu muốn train/inference trên máy tính cá nhân (Windows). Mô hình XLM-RoBERTa là mô hình duy nhất không yêu cầu bước này.
---

## Cấu trúc Repository

```text
pii-ner-vietnamese/
├── src/
│   ├── config.py         # Siêu tham số, nhãn BIO, đường dẫn
│   ├── data_loader.py    # Xử lý VnCoreNLP, Tokenization, Label Alignment
│   ├── models.py         # Custom nn.Module cho BiLSTM và BiLSTM-CRF
│   ├── train.py          # Script huấn luyện (Trainer API)
│   └── utils.py          # Tính toán metric (Seqeval, Classification, CRF decode)
├── inference/
│   └── run_inference.py  # Pipeline dự đoán cho văn bản thô
├── notebooks/            # (Archive) Các file Jupyter Notebook gốc
├── data/
│   ├── tokenized_phobert/# Dữ liệu đã tokenize cho PhoBERT/BiLSTM
│   └── tokenized_xlmr/   # Dữ liệu đã tokenize cho XLM-R
├── app.py                # Gradio UI Demo
└── requirements.txt      # Dependencies
```
