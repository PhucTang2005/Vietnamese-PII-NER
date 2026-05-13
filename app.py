import gradio as gr
import json
import os
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading
import torch
from transformers import pipeline

# ==========================================
# 1. Custom CSS
# ==========================================
custom_css = """
.pii-entity {
    font-weight: bold;
    background-color: rgba(250, 204, 21, 0.3);
    border-bottom: 2px solid #eab308;
    border-radius: 4px;
    padding: 2px 4px;
    position: relative;
    cursor: help;
    transition: background-color 0.2s;
}
.pii-entity:hover {
    background-color: rgba(250, 204, 21, 0.6);
}
.pii-entity .tooltip-label {
    visibility: hidden;
    background-color: #1f2937;
    color: #f9fafb;
    text-align: center;
    border-radius: 6px;
    padding: 4px 8px;
    position: absolute;
    z-index: 50;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.2s, bottom 0.2s;
    font-size: 0.75rem;
    font-family: monospace;
    font-weight: normal;
    white-space: nowrap;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
}
.pii-entity .tooltip-label::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #1f2937 transparent transparent transparent;
}
.pii-entity:hover .tooltip-label {
    visibility: visible;
    opacity: 1;
    bottom: 135%;
}
.text-container {
    font-size: 1rem;
    line-height: 1.8;
    padding: 1.5rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    min-height: 120px;
}
.dark .text-container {
    background: #1f2937;
    border-color: #374151;
    color: #e5e7eb;
}
.model-header-xlmr {
    background: linear-gradient(135deg, #8b5cf6, #5b21b6);
    color: white;
    padding: 8px 16px;
    border-radius: 8px 8px 0 0;
    font-weight: bold;
    font-size: 0.9rem;
    margin-bottom: 0;
}
.batch-sample {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    margin-bottom: 24px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.dark .batch-sample {
    border-color: #374151;
}
.batch-sample-header {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    padding: 10px 16px;
    font-weight: bold;
    font-size: 0.95rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.batch-sample-input {
    padding: 12px 16px;
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.85rem;
    color: #475569;
    white-space: pre-wrap;
    max-height: 120px;
    overflow-y: auto;
}
.dark .batch-sample-input {
    background: #111827;
    border-color: #374151;
    color: #94a3b8;
}
.batch-model-col {
    padding: 12px 16px;
}
.batch-model-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-block;
}
.batch-model-label.xlmr {
    background: #ede9fe;
    color: #5b21b6;
}
.dark .batch-model-label.xlmr {
    background: #2e1065;
    color: #c4b5fd;
}
.batch-entity-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 6px;
}
.batch-entity-tag {
    font-size: 0.72rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: #fef3c7;
    color: #92400e;
    font-family: monospace;
    border: 1px solid #fde68a;
}
.dark .batch-entity-tag {
    background: #422006;
    color: #fde68a;
    border-color: #78350f;
}
.batch-summary {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 20px;
    font-size: 0.9rem;
}
.dark .batch-summary {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border-color: #166534;
    color: #bbf7d0;
}
.batch-progress {
    text-align: center;
    padding: 40px;
    color: #6b7280;
    font-size: 1.1rem;
}
"""


# ==========================================
# 2. Model Wrapper
# ==========================================
class TransformerNERWrapper:
    """Unified wrapper for HuggingFace NER pipeline models."""

    def __init__(self, hf_repo: str, threshold: float = 0.5):
        self.pipe = pipeline(
            "ner",
            model=hf_repo,
            tokenizer=hf_repo,
            aggregation_strategy="simple",
            device=-1,
            torch_dtype=torch.float32,
        )
        self.threshold = threshold

    def predict(self, text: str) -> List[Dict]:
        raw = self.pipe(text)
        results = []
        for e in raw:
            label = e.get("entity_group") or e.get("entity") or ""
            if not label:
                continue
            score = float(e.get("score", 0.0))
            if score >= self.threshold:
                results.append({
                    "text": e.get("word", ""),
                    "label": label,
                    "start": e.get("start", 0),
                    "end": e.get("end", 0),
                    "score": score,
                })
        return results


# ==========================================
# 3. Lazy model loaders (singleton)
# ==========================================
_MODEL_INSTANCES: Dict[str, object] = {}
_MODEL_FACTORIES = {
    "xlmr": lambda: TransformerNERWrapper(
        "Phuc2005/pii-xlm-r-base-ner", 
        threshold=0.5
    ),
}

MODEL_DISPLAY_NAMES = {
    "xlmr": "XLM-RoBERTa",
}

# Pre-warm:
def _preload_models():
    print("🔄 Pre-loading models in background...")
    for key in _MODEL_FACTORIES:
        get_model(key)
    print("✅ All models loaded.")



def get_model(key: str):
    if key not in _MODEL_FACTORIES:
        return None
    if key not in _MODEL_INSTANCES:
        print(f"🚀 Khởi tạo model: {key}")
        _MODEL_INSTANCES[key] = _MODEL_FACTORIES[key]()
        print(f"✅ Model '{key}' sẵn sàng.")
    return _MODEL_INSTANCES[key]


# ==========================================
# 4. Hàm tạo HTML
# ==========================================
def build_html_with_tooltips(text: str, spans: List[Dict]) -> str:
    spans = sorted(spans, key=lambda x: x["start"])
    html_content = ""
    last_idx = 0
    for span in spans:
        start, end, label = span["start"], span["end"], span["entity"]
        if start >= end or start < last_idx:
            continue
        html_content += text[last_idx:start].replace('\n', '<br>')
        pii_text = text[start:end].replace('\n', '<br>')
        html_content += (
            f'<span class="pii-entity">{pii_text}'
            f'<span class="tooltip-label">{label}</span></span>'
        )
        last_idx = end
    html_content += text[last_idx:].replace('\n', '<br>')
    return f'<div class="text-container">{html_content}</div>'


def _build_entity_tags_html(preds: list) -> str:
    if not preds or (isinstance(preds, dict) and "error" in preds):
        return '<span style="color:#ef4444;">Lỗi</span>'
    if not preds:
        return '<span style="color:#9ca3af;">Không phát hiện PII</span>'
    tags = []
    for p in preds:
        label = p.get("label", "?")
        text_val = p.get("text", "")
        display = text_val if len(text_val) <= 30 else text_val[:27] + "..."
        display = display.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        tags.append(f'<span class="batch-entity-tag" title="{display}">{label}: {display}</span>')
    return "".join(tags)


# ==========================================
# 5. Core: chạy 1 model, trả về (html, json)
# ==========================================
def run_single_model(text: str, model_key: str):
    try:
        wrapper = get_model(model_key)
        preds = wrapper.predict(text)
        spans_for_html = [
            {"entity": p["label"], "start": p["start"], "end": p["end"]}
            for p in preds
        ]
        html = build_html_with_tooltips(text, spans_for_html)
        return html, preds

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        return (
            f"<div style='color:red;'>Lỗi model {model_key}: {exc}</div>",
            {"error": str(exc), "traceback": tb}
        )

# ==========================================
# 6. Wrapper cho Gradio (1 model)
# ==========================================
def process_text(text: str):
    html_xlmr, json_xlmr = run_single_model(text, "xlmr")
    return html_xlmr, json_xlmr


# ==========================================
# 6b. Batch processing — test file upload
# ==========================================
def process_test_file(file):
    if file is None:
        return "<div class='batch-progress'>⬆️ Vui lòng upload file JSON.</div>", []

    try:
        file_path = file if isinstance(file, str) else file.name
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return f"<div style='color:red;padding:20px;'>Lỗi đọc file: {exc}</div>", []

    if not isinstance(data, list):
        return "<div style='color:red;padding:20px;'>File JSON phải là một mảng (array) các object có trường \"text\".</div>", []

    total = len(data)
    all_results = []
    html_parts = []
    total_pii_xlmr = 0

    for idx, item in enumerate(data):
        text = item.get("text", "")
        if not text:
            continue

        sample_num = idx + 1
        short_text = text[:150].replace("<", "&lt;").replace(">", "&gt;").replace("\n", " ")
        if len(text) > 150:
            short_text += "..."

        result_entry = {"sample": sample_num, "text_preview": text[:100]}

        # Run model
        html_xlmr, preds_xlmr = run_single_model(text, "xlmr")
        result_entry["xlmr"] = preds_xlmr
        if isinstance(preds_xlmr, list):
            total_pii_xlmr += len(preds_xlmr)

        all_results.append(result_entry)

        # Build sample HTML
        html_parts.append(f'<div class="batch-sample">')
        html_parts.append(
            f'<div class="batch-sample-header">'
            f'<span>📄 Mẫu #{sample_num} / {total}</span>'
            f'<span style="font-weight:normal;font-size:0.8rem;">{len(text)} ký tự</span>'
            f'</div>'
        )
        html_parts.append(f'<div class="batch-sample-input">{short_text}</div>')

        html_parts.append(f'<div style="padding:12px 16px;">')
        html_parts.append(f'<div class="batch-model-label xlmr">XLM-RoBERTa</div>')
        if html_xlmr:
            html_parts.append(html_xlmr)
        n = len(preds_xlmr) if isinstance(preds_xlmr, list) else 0
        html_parts.append(f'<div class="batch-entity-tags"><strong style="font-size:0.75rem;margin-right:6px;">{n} entities:</strong>')
        html_parts.append(_build_entity_tags_html(preds_xlmr))
        html_parts.append('</div></div>')

        html_parts.append('</div>')  # end batch-sample

    # Build summary
    summary_lines = [
        f"<div class='batch-summary'>",
        f"<strong>📊 Tổng kết:</strong> {total} mẫu đã xử lý<br>",
        f"🟣 <strong>XLM-RoBERTa:</strong> {total_pii_xlmr} PII entities phát hiện",
        "</div>"
    ]

    final_html = "".join(summary_lines) + "".join(html_parts)
    return final_html, all_results


# ==========================================
# 7. Giao diện Gradio UI — XLM-RoBERTa
# ==========================================
DEFAULT_TEXT = (
    "Xin chào, tôi là Nguyễn Văn An, số điện thoại 0912345678, địa chỉ 123 Lê Lợi, Quận 1. Ví của tôi là 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
)

with gr.Blocks(title="PII Extractor — XLM-RoBERTa", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🔍 Hệ thống Trích xuất PII — XLM-RoBERTa")

    with gr.Tabs():
        # ── Tab 1: Nhập văn bản ──────────────────────────────────────
        with gr.TabItem("📝 Nhập văn bản"):
            gr.Markdown(
                "Nhập văn bản bên dưới và nhấn **Trích xuất PII**. "
                "Kết quả của **XLM-RoBERTa** sẽ hiển thị."
            )

            with gr.Row():
                input_text = gr.Textbox(
                    lines=12,
                    label="📝 Văn bản đầu vào",
                    value=DEFAULT_TEXT,
                    scale=1,
                )

            submit_btn = gr.Button("⚡ Trích xuất PII", variant="primary", size="lg")

            gr.Markdown("## 📊 Kết quả nhận diện")
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.HTML(
                        '<div class="model-header-xlmr">🟣 Model — XLM-RoBERTa (Phuc2005/pii-xlm-r-base-ner, threshold=0.5)</div>'
                    )
                    out_html_xlmr = gr.HTML(label="XLM-RoBERTa Output")

            gr.Markdown("## 🗂️ Dữ liệu JSON thô")
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("**XLM-RoBERTa JSON**")
                    out_json_xlmr = gr.JSON(label="XLM-RoBERTa Raw JSON")

            submit_btn.click(
                fn=process_text,
                inputs=[input_text],
                outputs=[out_html_xlmr, out_json_xlmr],
            )

        # ── Tab 2: Test File ─────────────────────────────────────────
        with gr.TabItem("📂 Test File"):
            gr.Markdown(
                "Upload file JSON chứa các mẫu test (format: `[{\"text\": \"...\"}, ...]`). "
                "Hệ thống sẽ chạy model trên từng mẫu và hiển thị kết quả trực quan."
            )

            with gr.Row():
                with gr.Column(scale=2):
                    test_file_input = gr.File(
                        label="📁 Upload file JSON test",
                        file_types=[".json"],
                        type="filepath",
                    )

            test_btn = gr.Button("🚀 Chạy Test Batch", variant="primary", size="lg")

            gr.Markdown("## 📊 Kết quả Test")
            test_output_html = gr.HTML(label="Kết quả trực quan")
            with gr.Accordion("🗂️ JSON chi tiết", open=False):
                test_output_json = gr.JSON(label="Kết quả JSON")

            test_btn.click(
                fn=process_test_file,
                inputs=[test_file_input],
                outputs=[test_output_html, test_output_json],
            )

if __name__ == "__main__":
    threading.Thread(target=_preload_models, daemon=True).start()
    demo.launch(server_name="127.0.0.1", server_port=7860)
