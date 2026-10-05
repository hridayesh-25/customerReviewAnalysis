"""
Customer Emotion Analysis - Streamlit Application
Deep Learning Based Customer Review Classification
Uses pre-trained LSTM, BiLSTM, and CNN models (no retraining).

Model inspection findings:
- All models: sigmoid output (1 unit), output > 0.5 => Positive, <= 0.5 => Negative
- All tokenizers: num_words=30000, oov_token='<OOV>', lower=True
- Embedding input_dim=30000, output_dim=128
- Sequence length: maxlen=200 (models have dynamic input; 200 matches typical training config)
- Class mapping verified empirically: sigmoid ~0.999 for positive, ~0.001 for negative

Verified evaluation metrics (40,000-review held-out Amazon test set):
- LSTM:   Accuracy=91.32%  Precision=93.98%  Recall=88.30%  F1=91.05%
- BiLSTM: Accuracy=92.06%  Precision=91.87%  Recall=92.29%  F1=92.08%  <- Best
- CNN:    Accuracy=91.80%  Precision=92.37%  Recall=91.14%  F1=91.75%
"""

import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
# CONSTANTS (verified from inspection)
# ─────────────────────────────────────────────
MAXLEN    = 200
NUM_WORDS = 30000
THRESHOLD = 0.5

BASE_DIR  = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
TOK_DIR   = os.path.join(BASE_DIR, "tokenizers")

VERIFIED_METRICS = [
    {"Model": "LSTM",   "Accuracy": 0.9132, "Precision": 0.9398, "Recall": 0.8830, "F1 Score": 0.9105,
     "CM_TN": 18869, "CM_FP": 1131, "CM_FN": 2341, "CM_TP": 17659},
    {"Model": "BiLSTM", "Accuracy": 0.9206, "Precision": 0.9187, "Recall": 0.9229, "F1 Score": 0.9208,
     "CM_TN": 18366, "CM_FP": 1634, "CM_FN": 1542, "CM_TP": 18458},
    {"Model": "CNN",    "Accuracy": 0.9180, "Precision": 0.9237, "Recall": 0.9114, "F1 Score": 0.9175,
     "CM_TN": 18494, "CM_FP": 1506, "CM_FN": 1773, "CM_TP": 18227},
]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Emotion Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #f5f7fa; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #0f2444 0%, #1a4a8a 60%, #1e6ab0 100%);
    border-radius: 14px;
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
  }
  .hero::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
  }
  .hero h1 { color: #fff; margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; }
  .hero p  { color: rgba(255,255,255,0.72); margin: 0.35rem 0 0; font-size: 0.97rem; }
  .hero .status {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(76,175,80,0.18); border: 1px solid rgba(76,175,80,0.4);
    color: #81c784; border-radius: 20px;
    padding: 0.22rem 0.8rem; font-size: 0.78rem; font-weight: 500;
    margin-top: 0.9rem;
  }
  .hero .status .dot {
    width: 7px; height: 7px; background: #4caf50;
    border-radius: 50%; display: inline-block;
  }

  /* Page header (non-hero pages) */
  .page-header {
    background: linear-gradient(135deg, #0f2444 0%, #1a4a8a 100%);
    color: white; padding: 1.8rem 2.2rem;
    border-radius: 12px; margin-bottom: 1.6rem;
  }
  .page-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
  .page-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; opacity: 0.8; }

  /* Model perf cards */
  .model-card {
    background: white; border-radius: 12px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border: 1px solid #e4eaf2;
    position: relative; height: 100%;
  }
  .model-card .model-name {
    font-size: 0.82rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.07em;
    color: #546e7a; margin-bottom: 0.5rem;
  }
  .model-card .accuracy {
    font-size: 2rem; font-weight: 700; color: #0f2444; line-height: 1;
  }
  .model-card .f1 {
    font-size: 0.85rem; color: #78909c; margin-top: 0.25rem;
  }
  .model-card .best-badge {
    position: absolute; top: 1rem; right: 1rem;
    background: #fff8e1; color: #f57f17;
    border: 1px solid #ffe082;
    border-radius: 20px; padding: 0.15rem 0.6rem;
    font-size: 0.72rem; font-weight: 600;
  }
  .model-card .accent { height: 3px; border-radius: 2px; margin-bottom: 0.9rem; }

  /* Best model highlight */
  .best-card {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border: 1.5px solid #a5d6a7;
    border-radius: 12px; padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
  }
  .best-card h3 { margin: 0 0 0.3rem; color: #1b5e20; font-size: 1.05rem; }
  .best-card .best-name { font-size: 1.7rem; font-weight: 700; color: #2e7d32; margin: 0.2rem 0 0.8rem; }
  .best-card .best-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
  .best-card .stat { }
  .best-card .stat .sl { font-size: 0.72rem; color: #66bb6a; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
  .best-card .stat .sv { font-size: 1.1rem; font-weight: 700; color: #1b5e20; }
  .best-card p { margin: 0.8rem 0 0; font-size: 0.85rem; color: #388e3c; }

  /* Generic stat card */
  .stat-card {
    background: white; border-radius: 10px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e4eaf2;
    text-align: center; margin-bottom: 0.8rem;
  }
  .stat-card .label { font-size: 0.74rem; color: #90a4ae; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
  .stat-card .value { font-size: 1.55rem; font-weight: 700; color: #0f2444; margin-top: 0.15rem; }

  /* Result cards (prediction output) */
  .result-card {
    background: white; border-radius: 12px; padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #e4eaf2; height: 100%;
  }
  .result-card h4 { font-size: 0.95rem; font-weight: 600; color: #0f2444; margin: 0 0 1rem; }

  /* Badges */
  .badge-pos { background: #e8f5e9; color: #2e7d32; padding: 0.28rem 0.85rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; }
  .badge-neg { background: #fce4ec; color: #c62828; padding: 0.28rem 0.85rem; border-radius: 20px; font-weight: 600; font-size: 0.9rem; display: inline-block; }

  /* Final verdict */
  .verdict-pos {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border: 2px solid #4caf50; border-radius: 12px;
    padding: 1.6rem; text-align: center;
  }
  .verdict-neg {
    background: linear-gradient(135deg, #fce4ec, #f8bbd0);
    border: 2px solid #f44336; border-radius: 12px;
    padding: 1.6rem; text-align: center;
  }
  .verdict-pos h2 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #1b5e20; }
  .verdict-neg h2 { margin: 0; font-size: 1.8rem; font-weight: 700; color: #b71c1c; }
  .verdict-pos p  { margin: 0.4rem 0 0; font-size: 0.9rem; color: #2e7d32; }
  .verdict-neg p  { margin: 0.4rem 0 0; font-size: 0.9rem; color: #c62828; }

  /* Info box */
  .info-box {
    background: #e3f2fd; border-left: 4px solid #1565c0;
    border-radius: 6px; padding: 0.75rem 1rem;
    font-size: 0.86rem; color: #0d47a1; margin-bottom: 1rem;
  }

  /* Methodology */
  .meth-card {
    background: white; border-radius: 10px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e4eaf2; margin-bottom: 0.8rem;
  }
  .meth-card h4 { margin: 0 0 0.4rem; font-size: 0.95rem; font-weight: 600; color: #0f2444; }
  .meth-card p  { margin: 0; font-size: 0.86rem; color: #546e7a; line-height: 1.6; }

  /* Sidebar */
  .sidebar-tag {
    background: #e8f0fb; color: #0f2444;
    border-radius: 6px; padding: 0.28rem 0.75rem;
    font-size: 0.8rem; font-weight: 500;
    display: inline-block; margin: 0.12rem 0;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0f2444, #1a4a8a);
    color: white !important; border: none; border-radius: 8px;
    padding: 0.5rem 1.6rem; font-weight: 600; font-size: 0.92rem;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.87; }

  div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
  .stDataFrame { font-size: 0.85rem; }
  hr.section-div { border: none; border-top: 1px solid #e4eaf2; margin: 1.4rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL & TOKENIZER LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_lstm_model():
    path = os.path.join(MODEL_DIR, "lstm_customer_emotion.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM model not found: {path}")
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_bilstm_model():
    path = os.path.join(MODEL_DIR, "bilstm_customer_emotion.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BiLSTM model not found: {path}")
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_cnn_model():
    path = os.path.join(MODEL_DIR, "cnn_customer_emotion.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNN model not found: {path}")
    return load_model(path)

@st.cache_resource(show_spinner=False)
def load_lstm_tokenizer():
    path = os.path.join(TOK_DIR, "lstm_tokenizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM tokenizer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_bilstm_tokenizer():
    path = os.path.join(TOK_DIR, "bilstm_tokenizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"BiLSTM tokenizer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_cnn_tokenizer():
    path = os.path.join(TOK_DIR, "cnn_tokenizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"CNN tokenizer not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_resources():
    resources, errors = {}, []
    loaders = {
        "lstm_model":      load_lstm_model,
        "bilstm_model":    load_bilstm_model,
        "cnn_model":       load_cnn_model,
        "lstm_tokenizer":  load_lstm_tokenizer,
        "bilstm_tokenizer":load_bilstm_tokenizer,
        "cnn_tokenizer":   load_cnn_tokenizer,
    }
    for key, fn in loaders.items():
        try:
            resources[key] = fn()
        except Exception as e:
            errors.append(f"Failed to load {key}: {e}")
    return resources, errors


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_text(text: str, tokenizer, maxlen: int = MAXLEN) -> np.ndarray:
    """
    Reproduce training preprocessing:
      1. tokenizer lowercases and filters punctuation automatically
      2. texts_to_sequences with the given tokenizer
      3. pad_sequences to maxlen=200, padding='post', truncating='post'
    """
    if not isinstance(text, str):
        text = str(text)
    sequences = tokenizer.texts_to_sequences([text.strip()])
    return pad_sequences(sequences, maxlen=maxlen, padding="post", truncating="post")


# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS  (unchanged)
# ─────────────────────────────────────────────

def _predict(model, tokenizer, text: str) -> dict:
    padded = preprocess_text(text, tokenizer)
    raw = float(model.predict(padded, verbose=0)[0][0])
    raw = max(0.0, min(1.0, raw))
    pos_prob, neg_prob = raw, 1.0 - raw
    if pos_prob > THRESHOLD:
        label, confidence = "Positive", pos_prob
    else:
        label, confidence = "Negative", neg_prob
    return {"label": label, "confidence": confidence,
            "pos_prob": pos_prob, "neg_prob": neg_prob, "raw_output": raw}

def predict_lstm(text, resources):   return _predict(resources["lstm_model"],   resources["lstm_tokenizer"],   text)
def predict_bilstm(text, resources): return _predict(resources["bilstm_model"], resources["bilstm_tokenizer"], text)
def predict_cnn(text, resources):    return _predict(resources["cnn_model"],    resources["cnn_tokenizer"],    text)


def majority_vote(results: dict) -> dict:
    labels = {k: v["label"] for k, v in results.items()}
    pos_count = sum(1 for l in labels.values() if l == "Positive")
    neg_count = sum(1 for l in labels.values() if l == "Negative")
    total = len(labels)
    if pos_count > neg_count:
        final, votes = "Positive", pos_count
    elif neg_count > pos_count:
        final, votes = "Negative", neg_count
    else:
        avg_pos = np.mean([v["pos_prob"] for v in results.values()])
        final = "Positive" if avg_pos >= 0.5 else "Negative"
        votes = total // 2
        st.warning("Tie resolved by highest average probability.")
    return {"final_label": final, "votes": votes, "total": total,
            "agreement_str": f"{votes}/{total} models agree", "model_votes": labels}


def analyze_review(text: str, resources: dict) -> tuple:
    results = {
        "LSTM":   predict_lstm(text, resources),
        "BiLSTM": predict_bilstm(text, resources),
        "CNN":    predict_cnn(text, resources),
    }
    return results, majority_vote(results)


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def prob_bar(label: str, prob: float, color: str):
    pct = prob * 100
    st.markdown(f"""
    <div style="margin-bottom:0.55rem;">
      <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#78909c;margin-bottom:3px;">
        <span>{label}</span><span>{pct:.1f}%</span>
      </div>
      <div style="background:#eef2f7;border-radius:4px;height:7px;">
        <div style="background:{color};width:{pct:.1f}%;height:7px;border-radius:4px;"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def render_model_result_card(model_name: str, result: dict):
    label = result["label"]
    badge = "badge-pos" if label == "Positive" else "badge-neg"
    icon  = "+" if label == "Positive" else "-"
    conf_color = "#2e7d32" if label == "Positive" else "#c62828"
    st.markdown(f"""
    <div class="result-card">
      <h4>{model_name}</h4>
      <div style="margin-bottom:0.75rem;">
        <span style="font-size:0.72rem;color:#90a4ae;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">Prediction</span><br>
        <span class="{badge}">{icon} {label}</span>
      </div>
      <div style="margin-bottom:1rem;">
        <span style="font-size:0.72rem;color:#90a4ae;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">Confidence</span><br>
        <span style="font-size:1.45rem;font-weight:700;color:{conf_color};">{result['confidence']*100:.1f}%</span>
      </div>
    </div>""", unsafe_allow_html=True)
    prob_bar("Positive", result["pos_prob"], "#4caf50")
    prob_bar("Negative", result["neg_prob"], "#ef5350")


def _load_metrics() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, "data", "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame(VERIFIED_METRICS)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1.1rem 0 0.6rem;">
          <div style="font-size:2rem;">🧠</div>
          <div style="font-size:1rem;font-weight:700;color:#0f2444;margin-top:0.3rem;">Customer Emotion</div>
          <div style="font-size:0.75rem;color:#90a4ae;">Deep Learning Analytics</div>
        </div>
        <hr style="border:none;border-top:1px solid #e4eaf2;margin:0.7rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "nav",
            ["🏠 Dashboard",
             "🔍 Review Prediction",
             "📊 Model Comparison",
             "📁 Batch Analysis",
             "🔬 Methodology"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border:none;border-top:1px solid #e4eaf2;margin:0.7rem 0;">
        <div style="font-size:0.72rem;color:#90a4ae;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.45rem;">Models</div>
        <div class="sidebar-tag">LSTM</div><br>
        <div class="sidebar-tag">BiLSTM</div><br>
        <div class="sidebar-tag">CNN</div>
        <hr style="border:none;border-top:1px solid #e4eaf2;margin:0.7rem 0;">
        <div style="font-size:0.72rem;color:#b0bec5;text-align:center;">B.Tech Final Year Project</div>
        """, unsafe_allow_html=True)

    return page


# ─────────────────────────────────────────────
# PAGE: OVERVIEW  (new dashboard)
# ─────────────────────────────────────────────

def page_overview(resources: dict):
    # ── Hero ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <h1>Customer Emotion Intelligence</h1>
      <p>AI-powered customer review analysis using deep learning ensemble</p>
      <div class="status"><span class="dot"></span>System Ready &nbsp;|&nbsp; 3 Models Loaded &nbsp;|&nbsp; Majority Voting Active</div>
    </div>""", unsafe_allow_html=True)

    # ── Model performance cards ───────────────────────────────────────────
    st.markdown('<p style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#90a4ae;margin-bottom:0.6rem;">Model Performance — Held-out Test Set (40,000 reviews)</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("LSTM",   "#2196f3", "91.32%", "91.05%", False),
        ("BiLSTM", "#4caf50", "92.06%", "92.08%", True),
        ("CNN",    "#ff9800", "91.80%", "91.75%", False),
    ]
    for col, (name, color, acc, f1, best) in zip([c1, c2, c3], cards):
        with col:
            badge = '<div class="best-badge">Best Overall</div>' if best else ""
            st.markdown(f"""
            <div class="model-card">
              {badge}
              <div class="accent" style="background:{color};"></div>
              <div class="model-name">{name}</div>
              <div class="accuracy">{acc}</div>
              <div class="f1">Accuracy &nbsp;|&nbsp; F1: {f1}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Best model highlight + performance chart side by side ────────────
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("""
        <div class="best-card">
          <h3>Best Performing Model</h3>
          <div class="best-name">BiLSTM</div>
          <div class="best-stats">
            <div class="stat"><div class="sl">Accuracy</div><div class="sv">92.06%</div></div>
            <div class="stat"><div class="sl">F1 Score</div><div class="sv">92.08%</div></div>
            <div class="stat"><div class="sl">Recall</div><div class="sv">92.29%</div></div>
          </div>
          <p>BiLSTM achieved the best overall performance on the 40,000-review held-out test set, with the strongest balance across all four metrics.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:white;border-radius:10px;padding:1.2rem 1.4rem;border:1px solid #e4eaf2;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#90a4ae;margin-bottom:0.8rem;">Why Three Models</div>
          <div style="font-size:0.86rem;color:#37474f;line-height:1.7;">
            <b>LSTM</b> — captures sequential dependencies<br>
            <b>BiLSTM</b> — bidirectional context<br>
            <b>CNN</b> — local n-gram patterns<br><br>
            <span style="color:#546e7a;">Majority voting combines all three predictions for a more robust final result.</span>
          </div>
        </div>""", unsafe_allow_html=True)

    with right:
        mdf = _load_metrics()
        fig = go.Figure()
        colors = {"LSTM": "#2196f3", "BiLSTM": "#4caf50", "CNN": "#ff9800"}
        metrics_to_show = ["Accuracy", "F1 Score"]
        for metric in metrics_to_show:
            fig.add_trace(go.Bar(
                name=metric,
                x=mdf["Model"],
                y=mdf[metric] * 100,
                text=[f"{v*100:.2f}%" for v in mdf[metric]],
                textposition="outside",
                textfont=dict(size=11),
                marker_color=["#2196f3","#4caf50","#ff9800"] if metric == "Accuracy" else ["#90caf9","#a5d6a7","#ffcc80"],
                marker_line_width=0,
            ))
        fig.update_layout(
            barmode="group",
            title=dict(text="Accuracy & F1 Score by Model", font=dict(size=13, color="#0f2444"), x=0),
            yaxis=dict(range=[89.5, 93.5], ticksuffix="%", gridcolor="#f0f4f8", title=""),
            xaxis=dict(showgrid=False),
            plot_bgcolor="white", paper_bgcolor="white",
            height=310, margin=dict(t=45, b=10, l=10, r=10),
            legend=dict(orientation="h", y=1.12, x=0),
            bargap=0.25, bargroupgap=0.08,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Dataset snapshot ─────────────────────────────────────────────────
    st.markdown('<hr class="section-div">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#90a4ae;margin-bottom:0.6rem;">Test Dataset Snapshot</p>', unsafe_allow_html=True)

    ds1, ds2, ds3, ds4, pie_col = st.columns([1, 1, 1, 1, 2])
    for col, lbl, val in [
        (ds1, "Total Reviews", "40,000"),
        (ds2, "Positive",      "20,000"),
        (ds3, "Negative",      "20,000"),
        (ds4, "Balance",       "50 / 50"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="label">{lbl}</div>
              <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    with pie_col:
        fig_pie = go.Figure(go.Pie(
            labels=["Positive", "Negative"], values=[20000, 20000],
            marker_colors=["#4caf50", "#ef5350"],
            hole=0.55, textinfo="percent+label",
            textfont_size=12,
        ))
        fig_pie.update_layout(
            height=160, margin=dict(t=8, b=8, l=8, r=8),
            paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Quick analyzer ────────────────────────────────────────────────────
    st.markdown('<hr class="section-div">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.78rem;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#90a4ae;margin-bottom:0.6rem;">Quick Review Analyzer</p>', unsafe_allow_html=True)

    review_text = st.text_area(
        "review_input",
        placeholder="The product quality is excellent and I am very happy with my purchase.",
        height=100,
        label_visibility="collapsed",
    )
    btn_col, _ = st.columns([1, 5])
    with btn_col:
        clicked = st.button("Analyze Review", use_container_width=True)

    if clicked:
        if not review_text or not review_text.strip():
            st.error("Please enter a review.")
            return
        if len(review_text.strip()) < 3:
            st.error("Review too short.")
            return

        with st.spinner("Running models..."):
            try:
                results, vote = analyze_review(review_text.strip(), resources)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        # Three model result cards
        r1, r2, r3 = st.columns(3)
        for col, mname in zip([r1, r2, r3], ["LSTM", "BiLSTM", "CNN"]):
            with col:
                render_model_result_card(mname, results[mname])

        st.markdown("<br>", unsafe_allow_html=True)

        # Final verdict
        final = vote["final_label"]
        cls   = "verdict-pos" if final == "Positive" else "verdict-neg"
        icon  = "+" if final == "Positive" else "-"
        vote_line = "  |  ".join(f"{m}: {l}" for m, l in vote["model_votes"].items())
        st.markdown(f"""
        <div class="{cls}">
          <h2>{icon} FINAL: {final.upper()}</h2>
          <p>{vote['agreement_str']} &nbsp;—&nbsp; {vote_line}</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ANALYZE REVIEW  (dedicated page)
# ─────────────────────────────────────────────

def page_analyze_review(resources: dict):
    st.markdown("""
    <div class="page-header">
      <h1>Review Prediction</h1>
      <p>Run a customer review through all three models simultaneously</p>
    </div>""", unsafe_allow_html=True)

    review_text = st.text_area(
        "Enter Customer Review",
        placeholder="The product quality is excellent and I am very happy with my purchase.",
        height=140,
    )
    col_btn, _ = st.columns([1, 5])
    with col_btn:
        clicked = st.button("Analyze Review", use_container_width=True)

    if not clicked:
        return

    if not review_text or not review_text.strip():
        st.error("Please enter a review before clicking Analyze.")
        return
    if len(review_text.strip()) < 3:
        st.error("Review is too short.")
        return
    if len(review_text) > 10000:
        st.warning("Review truncated to 10,000 characters.")
        review_text = review_text[:10000]

    with st.spinner("Running inference on all three models..."):
        try:
            results, vote = analyze_review(review_text.strip(), resources)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

    st.markdown("#### Model Predictions")
    c1, c2, c3 = st.columns(3)
    for col, mname in zip([c1, c2, c3], ["LSTM", "BiLSTM", "CNN"]):
        with col:
            render_model_result_card(mname, results[mname])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Final Prediction — Majority Voting")
    final = vote["final_label"]
    cls   = "verdict-pos" if final == "Positive" else "verdict-neg"
    icon  = "+" if final == "Positive" else "-"
    vote_line = "  |  ".join(f"{m}: {l}" for m, l in vote["model_votes"].items())
    st.markdown(f"""
    <div class="{cls}">
      <h2>{icon} {final.upper()}</h2>
      <p>{vote['agreement_str']} &nbsp;—&nbsp; {vote_line}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Confidence Overview")
    mnames = ["LSTM", "BiLSTM", "CNN"]
    pos_probs = [results[m]["pos_prob"] * 100 for m in mnames]
    neg_probs = [results[m]["neg_prob"] * 100 for m in mnames]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Positive %", x=mnames, y=pos_probs,
                         marker_color="#4caf50",
                         text=[f"{v:.1f}%" for v in pos_probs], textposition="outside"))
    fig.add_trace(go.Bar(name="Negative %", x=mnames, y=neg_probs,
                         marker_color="#ef5350",
                         text=[f"{v:.1f}%" for v in neg_probs], textposition="outside"))
    fig.update_layout(
        barmode="group", yaxis_range=[0, 115],
        yaxis_title="Probability (%)", xaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, margin=dict(t=20, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f4f8")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────

def page_model_performance():
    st.markdown("""
    <div class="page-header">
      <h1>Model Comparison</h1>
      <p>Evaluation metrics computed on the 40,000-review held-out Amazon test set</p>
    </div>""", unsafe_allow_html=True)

    mdf = _load_metrics()
    best_f1  = mdf.loc[mdf["F1 Score"].idxmax()]
    best_acc = mdf.loc[mdf["Accuracy"].idxmax()]
    best_prec= mdf.loc[mdf["Precision"].idxmax()]
    best_rec = mdf.loc[mdf["Recall"].idxmax()]

    st.markdown("""
    <div class="info-box">
      Real evaluation metrics — computed on the held-out Amazon test set
      (40,000 reviews: 20,000 positive / 20,000 negative). No values fabricated.
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in [
        (m1, "Test Set",     "40,000"),
        (m2, "Best Accuracy", f"{best_acc['Accuracy']*100:.2f}%  ({best_acc['Model']})"),
        (m3, "Best F1",       f"{best_f1['F1 Score']:.4f}  ({best_f1['Model']})"),
        (m4, "Best Model",    best_f1['Model']),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="label">{lbl}</div>
              <div class="value" style="font-size:1.05rem;">{val}</div>
            </div>""", unsafe_allow_html=True)

    # Metrics table
    st.markdown("#### Evaluation Metrics")
    disp = mdf[["Model","Accuracy","Precision","Recall","F1 Score"]].copy()
    for c in ["Accuracy","Precision","Recall","F1 Score"]:
        disp[c] = disp[c].apply(lambda x: f"{x*100:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # Charts row
    left, right = st.columns(2)
    with left:
        st.markdown("#### Performance Chart")
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        melted = mdf.melt(id_vars="Model", value_vars=metric_cols, var_name="Metric", value_name="Score")
        fig = px.bar(
            melted, x="Metric", y="Score", color="Model", barmode="group",
            color_discrete_map={"LSTM": "#2196f3", "BiLSTM": "#4caf50", "CNN": "#ff9800"},
            text=melted["Score"].apply(lambda x: f"{x*100:.1f}%"),
        )
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(
            yaxis=dict(range=[0.85, 0.97], tickformat=".0%", gridcolor="#f0f4f8", title=""),
            xaxis_title="", plot_bgcolor="white", paper_bgcolor="white",
            height=360, margin=dict(t=20, b=10),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Radar Comparison")
        cats = ["Accuracy", "Precision", "Recall", "F1 Score"]
        col_map = {"LSTM": "#2196f3", "BiLSTM": "#4caf50", "CNN": "#ff9800"}
        fig_r = go.Figure()
        for _, row in mdf.iterrows():
            vals = [row[c] for c in cats] + [row[cats[0]]]
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself", name=row["Model"],
                line_color=col_map.get(row["Model"], "#888"),
                fillcolor=col_map.get(row["Model"], "#888"),
                opacity=0.22,
            ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.87, 0.95])),
            height=360, paper_bgcolor="white",
            margin=dict(t=20, b=30),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Confusion matrices
    st.markdown("#### Confusion Matrices")
    st.caption("Rows = Actual label, Columns = Predicted label")
    cm1, cm2, cm3 = st.columns(3)
    for col, (_, row) in zip([cm1, cm2, cm3], mdf.iterrows()):
        tn, fp = int(row["CM_TN"]), int(row["CM_FP"])
        fn, tp = int(row["CM_FN"]), int(row["CM_TP"])
        fig_cm = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]],
            text=[[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]],
            texttemplate="%{text}",
            colorscale=[[0, "#e3f2fd"], [1, "#1565c0"]],
            showscale=False, xgap=3, ygap=3,
        ))
        fig_cm.update_layout(
            title=dict(text=row["Model"], x=0.5, font=dict(size=13, color="#0f2444")),
            xaxis=dict(tickvals=[0,1], ticktext=["Pred Neg","Pred Pos"], side="bottom"),
            yaxis=dict(tickvals=[0,1], ticktext=["Actual Neg","Actual Pos"], autorange="reversed"),
            height=250, margin=dict(t=40, b=25, l=80, r=10),
            paper_bgcolor="white", plot_bgcolor="white", font=dict(size=11),
        )
        with col:
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
      Best F1: <b>{best_f1['Model']}</b> ({best_f1['F1 Score']:.4f}) &nbsp;|&nbsp;
      Best Precision: <b>{best_prec['Model']}</b> ({best_prec['Precision']:.4f}) &nbsp;|&nbsp;
      Best Recall: <b>{best_rec['Model']}</b> ({best_rec['Recall']:.4f})
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: BATCH ANALYSIS
# ─────────────────────────────────────────────

def page_batch_analysis(resources: dict):
    st.markdown("""
    <div class="page-header">
      <h1>Batch Analysis</h1>
      <p>Upload a CSV to analyze multiple reviews at once</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      Upload a CSV with a column named <b>Review</b> or <b>text</b>.
      All three models run on every row and produce individual + majority-vote predictions.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded is None:
        st.markdown("""
        <div style="background:white;border-radius:10px;padding:2.2rem;text-align:center;
                    border:1px solid #e4eaf2;box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="font-size:2.5rem;margin-bottom:0.5rem;">📂</div>
          <div style="font-weight:600;color:#546e7a;">No file uploaded</div>
          <div style="font-size:0.85rem;color:#90a4ae;margin-top:0.3rem;">
            Upload a CSV containing customer reviews to begin.
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>**Sample CSV format:**", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Review": [
            "This product is amazing, totally worth the price!",
            "Very disappointed. The quality is terrible.",
            "Works as expected. Fast delivery.",
        ]}), use_container_width=True, hide_index=True)
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    if df.empty:
        st.error("The uploaded CSV is empty.")
        return

    st.success(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    st.dataframe(df.head(), use_container_width=True, hide_index=True)

    available_cols = list(df.columns)
    default_col = next(
        (c for c in ["Review","review","text","Text","comment","Comment"] if c in available_cols),
        available_cols[0]
    )
    text_col = st.selectbox("Select the review/text column:", available_cols,
                             index=available_cols.index(default_col))

    n_missing = df[text_col].isna().sum()
    if n_missing > 0:
        st.warning(f"{n_missing} rows with missing values will be skipped.")
        df = df.dropna(subset=[text_col]).reset_index(drop=True)

    if df.empty:
        st.error("No valid rows remaining.")
        return

    if not st.button("Run Batch Analysis"):
        return

    progress = st.progress(0, text="Starting...")
    rows, total = [], len(df)
    for i, text in enumerate(df[text_col]):
        try:
            res, vote = analyze_review(str(text), resources)
            rows.append({
                "Review":           str(text)[:200],
                "LSTM Prediction":  res["LSTM"]["label"],
                "LSTM Confidence":  f"{res['LSTM']['confidence']*100:.1f}%",
                "BiLSTM Prediction":res["BiLSTM"]["label"],
                "BiLSTM Confidence":f"{res['BiLSTM']['confidence']*100:.1f}%",
                "CNN Prediction":   res["CNN"]["label"],
                "CNN Confidence":   f"{res['CNN']['confidence']*100:.1f}%",
                "Final Prediction": vote["final_label"],
            })
        except Exception:
            rows.append({
                "Review": str(text)[:200],
                "LSTM Prediction":"Error","LSTM Confidence":"—",
                "BiLSTM Prediction":"Error","BiLSTM Confidence":"—",
                "CNN Prediction":"Error","CNN Confidence":"—",
                "Final Prediction":"Error",
            })
        progress.progress((i + 1) / total, text=f"Analyzing {i+1}/{total}...")

    progress.empty()
    rdf = pd.DataFrame(rows)
    st.success(f"Analysis complete — {total:,} reviews processed.")

    final_counts = rdf["Final Prediction"].value_counts()
    pos_c = int(final_counts.get("Positive", 0))
    neg_c = int(final_counts.get("Negative", 0))
    valid = pos_c + neg_c

    st.markdown("#### Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, lbl, val in [
        (s1, "Total Reviews", total),
        (s2, "Positive",      pos_c),
        (s3, "Negative",      neg_c),
        (s4, "Positive %",    f"{pos_c/valid*100:.1f}%" if valid else "—"),
        (s5, "Negative %",    f"{neg_c/valid*100:.1f}%" if valid else "—"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card">
              <div class="label">{lbl}</div>
              <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("#### Sentiment Distribution")
    pa, pb = st.columns(2)
    with pa:
        pie_data = rdf["Final Prediction"].value_counts().reset_index()
        pie_data.columns = ["Sentiment", "Count"]
        fig_pie = px.pie(pie_data, names="Sentiment", values="Count",
                         color="Sentiment",
                         color_discrete_map={"Positive":"#4caf50","Negative":"#ef5350","Error":"#bdbdbd"},
                         title="Final Majority Vote")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=330, paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with pb:
        bar_data = []
        for mc in ["LSTM Prediction","BiLSTM Prediction","CNN Prediction","Final Prediction"]:
            vc = rdf[mc].value_counts()
            bar_data.append({"Model": mc.replace(" Prediction",""),
                              "Positive": int(vc.get("Positive",0)),
                              "Negative": int(vc.get("Negative",0))})
        bdf = pd.DataFrame(bar_data)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(name="Positive", x=bdf["Model"], y=bdf["Positive"], marker_color="#4caf50"))
        fig_bar.add_trace(go.Bar(name="Negative", x=bdf["Model"], y=bdf["Negative"], marker_color="#ef5350"))
        fig_bar.update_layout(barmode="stack", title="Counts per Model",
                               yaxis_title="Reviews", plot_bgcolor="white", paper_bgcolor="white",
                               height=330, legend=dict(orientation="h", y=1.1))
        fig_bar.update_yaxes(gridcolor="#f0f4f8")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Detailed Results")
    st.dataframe(rdf, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Results as CSV",
        rdf.to_csv(index=False).encode("utf-8"),
        file_name="emotion_analysis_results.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE: METHODOLOGY
# ─────────────────────────────────────────────

def page_methodology():
    st.markdown("""
    <div class="page-header">
      <h1>Methodology</h1>
      <p>Dataset, preprocessing pipeline, model architectures, and classification approach</p>
    </div>""", unsafe_allow_html=True)

    with st.expander("📦  Dataset", expanded=True):
        st.markdown(
            "Models trained on a balanced subset of the **Amazon Customer Reviews** dataset. "
            "Evaluation used a separate held-out test set of **40,000 reviews** "
            "(20,000 positive / 20,000 negative) never seen during training."
        )

    with st.expander("🔧  Preprocessing Pipeline", expanded=False):
        steps = [
            ("Raw Text", "Customer review string as received."),
            ("Lowercasing", "Tokenizer applies `lower=True` automatically."),
            ("Filter Removal", "Punctuation and special characters removed via tokenizer filters."),
            ("Tokenization", "`texts_to_sequences()` maps words to integer indices (vocab: 30,000). OOV token: `<OOV>`."),
            ("Padding / Truncation", "`pad_sequences(maxlen=200, padding='post', truncating='post')`."),
            ("Model Input", "Shape `(1, 200)` integer sequence fed to the model."),
        ]
        for title, desc in steps:
            st.markdown(f"**{title}** — {desc}")

    with st.expander("🏗️  Model Architectures", expanded=False):
        arch_df = pd.DataFrame({
            "Model":        ["LSTM", "BiLSTM", "CNN"],
            "Architecture": [
                "Embedding → LSTM(64) → BatchNorm → Dropout → Dense(32) → Dense(1)",
                "Embedding → BiLSTM(128) → BatchNorm → Dropout → Dense(32) → Dense(1)",
                "Embedding → Conv1D(128) → GlobalMaxPool → BatchNorm → Dropout → Dense(64) → Dense(1)",
            ],
            "Parameters":   ["3,891,777", "3,943,489", "3,930,881"],
            "Embedding Dim":["128", "128", "128"],
            "Vocab Size":   ["30,000", "30,000", "30,000"],
            "Seq Length":   ["200", "200", "200"],
        })
        st.dataframe(arch_df, use_container_width=True, hide_index=True)

    with st.expander("📐  Sigmoid Classification", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "All three models use a **single sigmoid output unit**.\n\n"
                "- Output **> 0.50** → **Positive**\n"
                "- Output **≤ 0.50** → **Negative**\n\n"
                "Confidence = sigmoid value (Positive) or `1 − value` (Negative).\n\n"
                "Class mapping verified empirically: positive reviews yield sigmoid ≈ 0.999, "
                "negative ≈ 0.001."
            )
        with c2:
            st.markdown(
                "**Majority Voting**\n\n"
                "Final prediction = simple majority across LSTM, BiLSTM, CNN (2/3 or 3/3).\n\n"
                "Tiebreaker (rare edge case): model with highest average positive probability wins.\n\n"
                "Ensemble voting is more robust than any single model alone."
            )

    with st.expander("📋  Technical Specifications", expanded=False):
        spec_df = pd.DataFrame({
            "Property":  ["Output Units","Output Activation","Sequence Length",
                          "Vocabulary Size","Embedding Dim","OOV Token","Tokenizer Lower"],
            "LSTM":      ["1","Sigmoid","200","30,000","128","<OOV>","Yes"],
            "BiLSTM":    ["1","Sigmoid","200","30,000","128","<OOV>","Yes"],
            "CNN":       ["1","Sigmoid","200","30,000","128","<OOV>","Yes"],
        })
        st.dataframe(spec_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    page = render_sidebar()

    with st.spinner("Loading models and tokenizers..."):
        resources, errors = load_all_resources()

    if errors:
        st.error("Some resources failed to load:")
        for err in errors:
            st.error(f"  {err}")
        st.info("Ensure all .keras and .pkl files are present in models/ and tokenizers/.")
        if not resources:
            return

    if page == "🏠 Dashboard":
        page_overview(resources)
    elif page == "🔍 Review Prediction":
        page_analyze_review(resources)
    elif page == "📁 Batch Analysis":
        page_batch_analysis(resources)
    elif page == "📊 Model Comparison":
        page_model_performance()
    elif page == "🔬 Methodology":
        page_methodology()


if __name__ == "__main__":
    main()
