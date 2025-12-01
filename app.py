import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import xgboost as xgb

# ============================================
# Streamlit basic config
# ============================================
st.set_page_config(
    page_title="Learning Analytics – HTBT & XGBoost Dashboard",
    layout="wide",
    page_icon="📊",
)

# ============================================
# Custom CSS for UI
# ============================================
CUSTOM_CSS = """
<style>
    .stApp {
        background: radial-gradient(circle at top, #0f172a 0, #020617 55%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
    .metric-card {
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.85));
        box-shadow: 0 16px 40px rgba(0,0,0,0.45);
        border: 1px solid rgba(148,163,184,0.35);
        margin-bottom: 0.75rem;
    }
    .section-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        background: rgba(15,23,42,0.92);
        border: 1px solid rgba(51,65,85,0.8);
        box-shadow: 0 18px 45px rgba(0,0,0,0.75);
        margin-bottom: 1.25rem;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f9fafb;
        margin-bottom: 0.25rem;
    }
    .section-subtitle {
        font-size: 0.86rem;
        color: #9ca3af;
        margin-bottom: 0.75rem;
    }
    .small-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
    }
    .risk-low {
        background: linear-gradient(135deg, #22c55e, #16a34a);
    }
    .risk-med {
        background: linear-gradient(135deg, #eab308, #ca8a04);
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
    }
    .risk-pill {
        padding: 0.3rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #0b1120;
        display: inline-block;
    }
    .footer-note {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 1.5rem;
        text-align: right;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================
# HTBT constants (training settings)
# ============================================
HTBT_N_STATIC = 33
HTBT_NUM_CLASSES = 4
SEQ_LEN = 30

# ============================================
# HTBT model definition (same as training)
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class HTBT(nn.Module):
    def __init__(
        self, n_static, seq_len, d_model=128, n_heads=4,
        n_layers=3, d_ff=256, dropout=0.1, num_classes=4
    ):
        super().__init__()
        self.seq_proj = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.LayerNorm(d_model)
        )
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model), nn.ReLU(), nn.LayerNorm(d_model)
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.LayerNorm(d_ff),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.ReLU(),
            nn.Linear(d_ff // 2, num_classes),
        )

        self.attn_weights = None

    def forward(self, static_x, seq_x):
        seq = self.seq_proj(seq_x.unsqueeze(-1))
        seq = self.pos_enc(seq)
        seq_enc = self.seq_encoder(seq)
        seq_pool = seq_enc.mean(dim=1)

        static_emb = self.static_proj(static_x)
        q = static_emb.unsqueeze(1)
        attn_out, attn = self.cross_attn(q, seq_enc, seq_enc, need_weights=True)
        attn_out = attn_out.squeeze(1)
        self.attn_weights = attn

        fused = self.fusion_mlp(torch.cat([static_emb, attn_out], dim=-1))
        logits = self.classifier(fused)
        return logits, seq_pool, static_emb

# ============================================
# Simplified 12-feature UI subset
# ============================================
SIMPLIFIED = OrderedDict([
    ("total_clicks", "Total VLE Activity"),
    ("avg_clicks_per_visit", "Average Clicks per Visit"),
    ("active_days", "Active Study Days"),
    ("weighted_score", "Weighted Assessment Score"),
    ("avg_score", "Average Score"),
    ("performance_trend", "Performance Trend"),
    ("study_duration", "Study Duration (Days)"),
    ("engagement_efficiency", "Engagement Efficiency"),
    ("cbii", "Cognitive–Behavioural Index (CBII)"),
    ("tpi", "Temporal Persistence Index"),
    ("dropout_risk_proxy", "Dropout Risk Indicator"),
    ("activity_entropy", "Study Entropy"),
])

# ============================================
# Default random values
# ============================================
def default_value(name):
    n = name.lower()
    if "click" in n: return np.random.randint(100, 3000)
    if "visit" in n: return np.random.uniform(3, 80)
    if "active_days" in n: return np.random.randint(3, 60)
    if "score" in n: return np.random.uniform(30, 90)
    if "trend" in n: return np.random.uniform(-5, 5)
    if "duration" in n: return np.random.uniform(20, 200)
    if "efficiency" in n: return np.random.uniform(0.2, 2.5)
    if "cbii" in n: return np.random.uniform(0.0, 1.0)
    if "tpi" in n: return np.random.uniform(0.0, 40.0)
    if "dropout" in n: return np.random.uniform(0.0, 1.5)
    if "entropy" in n: return np.random.uniform(0, 3.5)
    return np.random.uniform(0, 1)

# ============================================
# Cached model loading
# ============================================
@st.cache_resource
def load_xgb():
    model = xgb.XGBClassifier()
    model.load_model("xgb_final.json")
    return model


@st.cache_resource
def load_htbt():
    model = HTBT(HTBT_N_STATIC, SEQ_LEN, num_classes=HTBT_NUM_CLASSES)
    try:
        state = torch.load("htbt_best.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        print("HTBT load failed:", e)
        return None

@st.cache_resource
def build_xgb_metadata():
    model = load_xgb()
    # class labels
    classes = getattr(model, "classes_", np.arange(4))

    # feature names
    booster = model.get_booster()
    if booster and booster.feature_names:
        feats = booster.feature_names
    elif hasattr(model, "feature_names_in_"):
        feats = model.feature_names_in_
    else:
        feats = list(SIMPLIFIED.keys())

    return model, classes, feats

@st.cache_resource
def build_explainers():
    model, classes, feats = build_xgb_metadata()
    bg = np.zeros((400, len(feats)), dtype=np.float32)

    for i, f in enumerate(feats):
        if f in SIMPLIFIED:
            bg[:, i] = [default_value(f) for _ in range(400)]

    try: shap_exp = shap.TreeExplainer(model)
    except: shap_exp = None

    try:
        lime_exp = LimeTabularExplainer(
            bg, feature_names=feats, class_names=[str(c) for c in classes],
            discretize_continuous=True, random_state=42
        )
    except:
        lime_exp = None

    return shap_exp, lime_exp, bg, feats, classes

# ============================================
# Load everything
# ============================================
xgb_model, xgb_classes, FEATURES = build_xgb_metadata()
shap_exp, lime_exp, bg_data, FEATURES, xgb_classes = build_explainers()
htbt = load_htbt()

# ============================================
# Header
# ============================================
st.markdown(
    "<h1>📊 Hybrid Temporal–Behavioural Analytics Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='color:#9ca3af;'>Interactive predictive modelling with XGBoost & HTBT, including SHAP, LIME and Hybrid explanations.</p>",
    unsafe_allow_html=True
)

# ============================================
# UI Layout
# ============================================
col_left, col_right = st.columns([1.1, 1.9])

# ============================================
# LEFT: INPUT PANEL
# ============================================
with col_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Student Indicators</div>", unsafe_allow_html=True)

    static_vals = {}

    with st.form("input_form"):
        for f, label in SIMPLIFIED.items():
            static_vals[f] = st.number_input(
                label,
                value=float(default_value(f)),
                step=0.1,
                format="%.3f",
            )

        st.markdown("#### 30-Day Activity Pattern")
        seq_vals = []
        for i in range(SEQ_LEN):
            seq_vals.append(
                st.number_input(
                    f"Day {i+1}",
                    value=float(np.random.randint(0, 60)),
                    step=1.0,
                    key=f"seq{i}",
                )
            )

        run = st.form_submit_button("🔍 Check Results")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# RIGHT: OUTPUT PANEL
# ============================================
with col_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Predictions</div>", unsafe_allow_html=True)

    if not run:
        st.info("Fill the inputs and click **Check Results**.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Prepare XGBoost vector
        x_vec = np.zeros(len(FEATURES), dtype=np.float32)
        for i, f in enumerate(FEATURES):
            x_vec[i] = static_vals.get(f, 0.0)

        x_pred = xgb_model.predict_proba(x_vec.reshape(1, -1))[0]
        pred_idx = int(np.argmax(x_pred))
        conf = float(np.max(x_pred))

        risk = ["High Risk", "Moderate", "Low", "Very Low"]
        risk_css = ["risk-high", "risk-med", "risk-low", "risk-low"]

        st.markdown(
            f"<div class='metric-card'><span class='small-label'>XGBoost Prediction</span>"
            f"<h3>Class {xgb_classes[pred_idx]}</h3>"
            f"<span class='risk-pill {risk_css[pred_idx]}'>{risk[pred_idx]} · {conf*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- HTBT prediction ---
        if htbt is not None:
            vec_htbt = np.zeros(HTBT_N_STATIC, dtype=np.float32)
            for i, f in enumerate(SIMPLIFIED.keys()):
                if i < HTBT_N_STATIC:
                    vec_htbt[i] = static_vals[f]

            static_tensor = torch.tensor(vec_htbt.reshape(1, -1), dtype=torch.float32)
            seq_tensor = torch.tensor(seq_vals, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits, _, _ = htbt(static_tensor, seq_tensor)
                p = torch.softmax(logits, 1).numpy()[0]
                pi = int(np.argmax(p))

            st.markdown(
                f"<div class='metric-card'><span class='small-label'>HTBT Prediction</span>"
                f"<h3>Class {xgb_classes[pi]}</h3>"
                f"<span class='risk-pill risk-low'>Confidence · {np.max(p)*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("HTBT model could not be loaded in Streamlit environment.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Explanation Tabs
# ============================================
if run:

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔍 Explainability (SHAP, LIME, Hybrid)</div>", unsafe_allow_html=True)

    col_shap, col_lime = st.columns(2)

    # --- SHAP ---
    with col_shap:
        st.markdown("<div class='small-label'>SHAP Explanation</div>", unsafe_allow_html=True)
        if shap_exp:
            sv = shap_exp.shap_values(x_vec.reshape(1, -1))
            if isinstance(sv, list): sv = sv[pred_idx][0]
            else: sv = sv[0]

            df = pd.DataFrame({"feature": FEATURES, "value": sv})
            df = df[df.feature.isin(SIMPLIFIED.keys())]
            df["abs"] = df["value"].abs()
            df = df.sort_values("abs", ascending=False).head(10)

            fig, ax = plt.subplots(figsize=(5,3))
            sns.barplot(x="value", y="feature", data=df, ax=ax, palette="rocket")
            st.pyplot(fig)
        else:
            st.info("SHAP is unavailable.")

    # --- LIME ---
    with col_lime:
        st.markdown("<div class='small-label'>LIME Explanation</div>", unsafe_allow_html=True)
        if lime_exp:
            le = lime_exp.explain_instance(x_vec, xgb_model.predict_proba)
            fig = le.as_pyplot_figure()
            st.pyplot(fig)
        else:
            st.info("LIME unavailable.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Hybrid ---
    st.markdown("<div class='small-label'>Hybrid SHAP + LIME</div>", unsafe_allow_html=True)

    if shap_exp and lime_exp:
        sv = shap_exp.shap_values(x_vec.reshape(1, -1))
        if isinstance(sv, list): sv = sv[pred_idx][0]
        else: sv = sv[0]
        shap_imp = np.abs(sv)

        lime_dict = dict(le.local_exp[pred_idx])
        lime_imp = np.zeros_like(shap_imp)
        for idx, w in lime_dict.items(): lime_imp[idx] = abs(w)

        shap_norm = shap_imp / (shap_imp.sum()+1e-9)
        lime_norm = lime_imp / (lime_imp.sum()+1e-9)

        hybrid = 0.5 * shap_norm + 0.5 * lime_norm
        dfh = pd.DataFrame({"feature": FEATURES, "hybrid": hybrid})
        dfh = dfh[dfh.feature.isin(SIMPLIFIED.keys())]
        dfh = dfh.sort_values("hybrid", ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x="hybrid", y="feature", data=dfh, ax=ax, palette="mako")
        st.pyplot(fig)
    else:
        st.info("Hybrid explanation unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown(
    "<div class='footer-note'>Dashboard powered by XGBoost + Hybrid Temporal–Behavioural Transformer (HTBT).</div>",
    unsafe_allow_html=True
)
