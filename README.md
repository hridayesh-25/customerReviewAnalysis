# Customer Emotion Intelligence
### AI-Powered Customer Review Analysis using Deep Learning
**B.Tech Final Year Project**

---

## Overview

A polished Streamlit web application that classifies customer reviews as **Positive** or **Negative** using three pre-trained deep learning models: LSTM, BiLSTM, and CNN. Models are loaded directly from saved `.keras` files — no retraining or fine-tuning occurs at any point.

---

## Features

- Live single review analysis with all three models simultaneously
- Individual prediction cards with confidence scores and probability bars
- Majority voting ensemble for a robust final prediction
- Real evaluation metrics from a 40,000-review held-out test set
- Interactive performance charts, radar comparison, and confusion matrices
- Batch CSV upload — analyze thousands of reviews at once
- Downloadable results CSV
- Dedicated Methodology page covering the full pipeline

---

## Navigation

| Page | Description |
|------|-------------|
| Dashboard | Hero section — model performance cards, best model highlight, quick analyzer, dataset snapshot |
| Review Prediction | Dedicated full-page review analysis with per-model cards and confidence chart |
| Model Comparison | Real metrics table, bar chart, radar chart, confusion matrices |
| Batch Analysis | Upload CSV, run all 3 models on every row, view charts, download results |
| Methodology | Dataset, preprocessing pipeline, architectures, sigmoid classification, majority voting (collapsible sections) |

---

## Models Used

| Model | Architecture | Parameters | Output |
|-------|-------------|------------|--------|
| LSTM | Embedding → LSTM(64) → BN → Dense(32) → Dense(1) | 3,891,777 | Sigmoid |
| BiLSTM | Embedding → BiLSTM(128) → BN → Dense(32) → Dense(1) | 3,943,489 | Sigmoid |
| CNN | Embedding → Conv1D(128) → GlobalMaxPool → BN → Dense(64) → Dense(1) | 3,930,881 | Sigmoid |

**Class mapping (verified empirically):**
- Sigmoid output > 0.5 → **Positive**
- Sigmoid output ≤ 0.5 → **Negative**

---

## Verified Evaluation Metrics

Evaluated on a held-out Amazon test set: **40,000 reviews (20,000 positive / 20,000 negative)**

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| LSTM | 91.32% | 93.98% | 88.30% | 91.05% |
| BiLSTM | **92.06%** | 91.87% | **92.29%** | **92.08%** |
| CNN | 91.80% | 92.37% | 91.14% | 91.75% |

**Best model: BiLSTM** (highest accuracy and F1 score)

> These metrics are real — computed by running each model on the held-out test set. No values were fabricated.

---

## Tokenizers

Each tokenizer is paired exclusively with its corresponding model and is never shared.

| Tokenizer | Model | num_words | oov_token | lower |
|-----------|-------|-----------|-----------|-------|
| lstm_tokenizer.pkl | LSTM | 30,000 | `<OOV>` | True |
| bilstm_tokenizer.pkl | BiLSTM | 30,000 | `<OOV>` | True |
| cnn_tokenizer.pkl | CNN | 30,000 | `<OOV>` | True |

---

## Project Structure

```
customer_emotion_app/
│
├── app.py                            # Main Streamlit application
│
├── models/
│   ├── lstm_customer_emotion.keras
│   ├── bilstm_customer_emotion.keras
│   └── cnn_customer_emotion.keras
│
├── tokenizers/
│   ├── lstm_tokenizer.pkl
│   ├── bilstm_tokenizer.pkl
│   └── cnn_tokenizer.pkl
│
├── data/
│   ├── metrics.json                  # Verified evaluation metrics
│   └── sample_reviews.csv            # 20-row test CSV for batch analysis
│
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.9–3.11 (project uses 3.10)

```bash
# 1. Navigate to the project folder
cd customer_emotion_app

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Preprocessing Pipeline

```
Customer Review
      ↓
Lowercasing + punctuation filtering (via tokenizer)
      ↓
texts_to_sequences()  [model-specific tokenizer]
      ↓
pad_sequences(maxlen=200, padding='post', truncating='post')
      ↓
Model inference  (LSTM / BiLSTM / CNN)
      ↓
Sigmoid output → Positive (>0.5) or Negative (≤0.5)
      ↓
Majority Voting
      ↓
Final Prediction
```

---

## Batch CSV Format

```csv
Review
"The product quality is excellent and delivery was fast."
"Terrible quality, completely broken on arrival."
"Works as described, good value for money."
```

Supported column names: `Review`, `review`, `text`, `Text`, `comment`
If your column has a different name, select it from the dropdown after upload.

A ready-to-use sample file is available at `data/sample_reviews.csv`.

---

## Future Improvements

- BERT / transformer-based model for comparison
- Multi-class emotion classification (joy, anger, neutral, etc.)
- LIME / SHAP token attribution for explainability
- REST API wrapper using FastAPI
- Confidence calibration visualization
- Live product review scraping integration
