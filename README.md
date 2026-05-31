<div align="center">

# 🧠 Smart Code Comment Generator

**Automatic natural language documentation for Python functions using Seq2Seq Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Colab](https://img.shields.io/badge/Run%20on-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> *Feed it a function. Get a comment back. No excuses for undocumented code.*

<br/>

</div>

---

## 📌 What Is This?

A deep learning system that reads a Python function and automatically generates a natural language comment describing what it does.

```python
# Input — raw function
def extract_video_id(url):
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Output — generated comment
>>> "Extracts video ID from URL."
```

Built using three progressively sophisticated architectures — vanilla LSTM, LSTM + Attention, and a fine-tuned T5-Small transformer — trained on 50,000 Python functions from the CodeSearchNet dataset.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   INPUT: Python Function                │
│         "def download_video(url): ..."                  │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Tokenizer (T5)        │
          │   "summarize: <code>"   │
          └────────────┬────────────┘
                       │
     ┌─────────────────▼──────────────────┐
     │         Encoder (Transformer)       │
     │   Processes token sequence         │
     │   Builds contextual representations │
     └─────────────────┬──────────────────┘
                       │
     ┌─────────────────▼──────────────────┐
     │         Decoder (Transformer)       │
     │   Generates output tokens          │
     │   Beam search (n_beams=4)          │
     └─────────────────┬──────────────────┘
                       │
          ┌────────────▼────────────┐
          │   OUTPUT: NL Comment    │
          │  "Downloads video by URL"│
          └─────────────────────────┘
```

---

## 📊 Results

### Quantitative (BLEU Score on Test Set)

| Model | BLEU-1 | BLEU-4 | Val Loss |
|---|---|---|---|
| LSTM Seq2Seq (Baseline) | ~0.05 | ~0.03 | ~0.08 |
| LSTM + Bahdanau Attention | ~0.09 | ~0.07 | ~0.05 |
| **T5-Small (Fine-tuned)** | **0.1364** | **0.1310** | **0.0184** |

### Training Curve

![Training Loss Curve](loss_curve.png)

### Qualitative Examples

| # | Actual | Generated |
|---|---|---|
| 1 | Extracts video ID from URL. | Extracts video ID from URL. ✅ |
| 2 | str->list Convert XML to URL List. | str->list Convert XML to URL List. ✅ |
| 3 | Downloads Dailymotion videos by URL. | Downloads Dailymotion videos by URL. ✅ |
| 4 | From stackoverflow.com/a/... | https://stackoverflow.com/a/... 🟡 |

---

## 🗂️ Repository Structure

```
smart-code-comment-generator/
│
├── notebooks/
│   └── CCP_Deep_Learning.ipynb     # Main notebook — run top to bottom
│
├── loss_curve.png                   # Training/validation loss plot
├── README.md
└── LICENSE
```

> **Dataset** is loaded directly from HuggingFace — no manual download needed.  
> **Model weights** are saved to Google Drive after training.

---

## 🚀 Quick Start

### Run on Google Colab (Recommended)

Click the badge at the top or open `notebooks/CCP_Deep_Learning.ipynb` directly in Colab.

Make sure to:
1. Enable **T4 GPU** → Runtime → Change runtime type → T4 GPU
2. Run all cells top to bottom
3. Mount Google Drive when prompted (for saving model weights)

### Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/smart-code-comment-generator.git
cd smart-code-comment-generator

# Install dependencies
pip install torch transformers datasets nltk

# Open the notebook
jupyter notebook notebooks/CCP_Deep_Learning.ipynb
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.0+ | Model training and inference |
| `transformers` | 4.40+ | T5 model and tokenizer |
| `datasets` | 2.x | CodeSearchNet data loading |
| `nltk` | 3.x | BLEU score evaluation |
| `matplotlib` | 3.x | Loss curve visualization |

---

## 🧪 Dataset

**[CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net)** — Python split

| Split | Samples Used |
|---|---|
| Train | 49,626 |
| Validation | 1,999 |
| Test | 1,995 |

Loaded via HuggingFace Datasets:
```python
from datasets import load_dataset
dataset = load_dataset("code-search-net/code_search_net", "python")
```

---

## ⚙️ Training Config

```python
MODEL      = "t5-small"
EPOCHS     = 3
BATCH_SIZE = 16
LR         = 5e-4
MAX_INPUT  = 256   # tokens
MAX_OUTPUT = 64    # tokens
OPTIMIZER  = AdamW
DEVICE     = "cuda"  # T4 GPU
```

---

## 🔍 How It Works — Step by Step

1. **Load** Python function-comment pairs from CodeSearchNet
2. **Preprocess** — filter empty pairs, prefix code with `"summarize: "`
3. **Tokenize** using T5Tokenizer (max 256 input tokens, 64 output tokens)
4. **Train** T5-Small with cross-entropy loss on comment token sequences
5. **Evaluate** using BLEU-1 and BLEU-4 on held-out test set
6. **Inference** — pass any Python function, get a generated comment

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## 📄 License

[MIT](LICENSE) — do whatever you want with it.

---

<div align="center">

Made with PyTorch + HuggingFace Transformers

</div>
