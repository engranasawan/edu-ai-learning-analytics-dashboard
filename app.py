import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ============================================================
# Streamlit basic config
# ============================================================
st.set_page_config(
    page_title="Learning Analytics – HTBT & XGBoost Dashboard",
    layout="wide",
    page_icon="📊",
)

# ============================================================
# Custom CSS
# ============================================================
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

# ============================================================
# HTBT architecture (for inference)
# ============================================================
SEQ_LEN = 30  # 30-day activity window


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
        # x: (B, L, D)
        return x + self.pe[:, : x.size(1), :].to(x.dtype)


class HTBT(nn.Module):
    def __init__(
        self,
        n_static,
        seq_len,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        num_classes=4,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.seq_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_ff),
            nn.Linear(d_ff, d_model),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, num_classes),
        )

        self.attn_weights = None

    def forward(self, static_x, seq_x):
        # static_x: (B, n_static), seq_x: (B, L)
        seq = seq_x.unsqueeze(-1)  # (B, L, 1)
        seq = self.seq_proj(seq)
        seq = self.pos_enc(seq)
        seq_enc = self.seq_encoder(seq)  # (B, L, D)
        seq_pool = seq_enc.mean(dim=1)   # (B, D)

        static_emb = self.static_proj(static_x)  # (B, D)

        q = static_emb.unsqueeze(1)  # (B, 1, D)
        k = seq_enc
        v = seq_enc
        attn_out, attn_weights = self.cross_attn(
            query=q, key=k, value=v, need_weights=True, average_attn_weights=False
        )
        attn_out = attn_out.squeeze(1)
        self.attn_weights = attn_weights  # (B, heads, 1, L)

        fused = torch.cat([static_emb, attn_out], dim=-1)
        fused = self.fusion_mlp(fused)
        logits = self.classifier(fused)
        return logits, seq_pool, static_emb


# ============================================================
# Feature configuration (simplified UI ↔ model features)
# ============================================================

# Backend feature names (from preprocessing / training) mapped to simple labels
SIMPLIFIED = {
    "total_clicks": "Total VLE Activity",
    "avg_clicks_per_visit": "Average Clicks per Visit",
    "active_days": "Active Study Days",
    "weighted_score": "Weighted Assessment Score",
    "avg_score": "Average Score",
    "performance_trend": "Performance Trend",
    "study_duration": "Study Duration (Days)",
    "engagement_efficiency": "Engagement Efficiency",
    "cbii": "Cognitive–Behavioural Index (CBII)",
    "tpi": "Temporal Persistence Index",
    "dropout_risk_proxy": "Dropout Risk Indicator",
    "activity_entropy": "Study Entropy",
}


def random_default(name: str) -> float:
    n = name.lower()
    if "total_clicks" in n:
        return float(np.random.randint(100, 3000))
    if "avg_clicks" in n:
        return float(np.random.randint(10, 900))
    if "active_days" in n:
        return float(np.random.randint(1, 150))
    if "weighted_score" in n or "avg_score" in n:
        return float(np.random.uniform(30, 90))
    if "trend" in n:
        return float(np.random.uniform(-5, 5))
    if "duration" in n:
        return float(np.random.uniform(30, 240))
    if "engagement_efficiency" in n:
        return float(np.random.uniform(0.1, 3.0))
    if "cbii" in n:
        return float(np.random.uniform(0.0, 1.0))
    if "tpi" in n:
        return float(np.random.uniform(0.0, 50.0))
    if "dropout_risk_proxy" in n:
        return float(np.random.uniform(0.0, 2.0))
    if "entropy" in n:
        return float(np.random.uniform(0.0, 3.5))
    return 0.0


def sample_background(feature_names, n_samples=400):
    bg = np.zeros((n_samples, len(feature_names)), dtype=np.float32)
    for j, fname in enumerate(feature_names):
        if fname in SIMPLIFIED:
            vals = [random_default(fname) for _ in range(n_samples)]
            bg[:, j] = np.array(vals)
        else:
            bg[:, j] = 0.0
    return bg


# ============================================================
# Cached model loading
# ============================================================
@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_final.json")
    return model


@st.cache_resource
def load_htbt_model(n_static: int, num_classes: int):
    try:
        model = HTBT(
            n_static=n_static,
            seq_len=SEQ_LEN,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            dropout=0.1,
            num_classes=num_classes,
        )
        state = torch.load("htbt_best.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        # If shapes or file are incompatible/missing, return None
        return None


# ============================================================
# Load models
# ============================================================
xgb_model = load_xgb_model()

if hasattr(xgb_model, "classes_"):
    XGB_CLASSES = np.array(xgb_model.classes_)
else:
    XGB_CLASSES = np.array([0, 1, 2, 3])

# human labels for classes
CLASS_LABELS = {
    0: "Fail",
    1: "Withdrawn",
    2: "Pass",
    3: "Distinction",
}
CLASS_LABELS_STR = [CLASS_LABELS.get(int(c), f"Class {int(c)}") for c in XGB_CLASSES]

# Get full backend feature list from the model
if hasattr(xgb_model, "feature_names_in_"):
    FEATURES = list(xgb_model.feature_names_in_)
else:
    FEATURES = list(SIMPLIFIED.keys())

# Map simplified features to indices in FEATURES
SIMPLIFIED_IDX = {
    fname: FEATURES.index(fname) for fname in SIMPLIFIED.keys() if fname in FEATURES
}

NUM_STATIC = len(FEATURES)
NUM_CLASSES = len(XGB_CLASSES)

# Optional HTBT model
htbt_model = load_htbt_model(NUM_STATIC, NUM_CLASSES)

# SHAP and LIME explainers (no caching with model as argument to avoid unhashable issues)
try:
    shap_explainer = shap.TreeExplainer(xgb_model)
except Exception:
    shap_explainer = None

try:
    background_data = sample_background(FEATURES, n_samples=400)
    lime_explainer = LimeTabularExplainer(
        training_data=background_data,
        feature_names=FEATURES,
        class_names=CLASS_LABELS_STR,
        discretize_continuous=True,
        random_state=42,
    )
except Exception:
    lime_explainer = None
    background_data = None


# ============================================================
# Helper: risk label
# ============================================================
def risk_label_from_class(pred_class: int, conf: float):
    # Treat Fail/Withdrawn as higher risk, Pass/Distinction as lower risk
    if pred_class == 0:
        txt = f"High risk of Fail · {conf*100:.1f}%"
        css = "risk-pill risk-high"
    elif pred_class == 1:
        txt = f"High risk of Withdrawal · {conf*100:.1f}%"
        css = "risk-pill risk-high"
    elif pred_class == 2:
        txt = f"Likely Pass · {conf*100:.1f}%"
        css = "risk-pill risk-low"
    else:
        txt = f"Likely Distinction · {conf*100:.1f}%"
        css = "risk-pill risk-low"
    return txt, css


# ============================================================
# Header
# ============================================================
st.markdown(
    "<h1 style='margin-bottom:0.25rem;'>📊 Hybrid Temporal–Behavioural Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9ca3af;margin-top:0;'>Interactive predictive modelling with XGBoost & HTBT, including SHAP, LIME and Hybrid explanations.</p>",
    unsafe_allow_html=True,
)

# ============================================================
# Layout
# ============================================================
col_left, col_right = st.columns([1.1, 1.9])

# ------------------------- LEFT: Inputs ----------------------
with col_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Student Indicators</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Set a student profile using key behavioural and performance indicators. Defaults are randomly initialised on each run.</div>",
        unsafe_allow_html=True,
    )

    static_values = {}
    seq_values = []

    with st.form("input_form"):
        st.markdown("#### 🔧 Summary Indicators")

        for backend_name, label in SIMPLIFIED.items():
            default_val = random_default(backend_name)
            static_values[backend_name] = st.number_input(
                label,
                value=float(default_val),
                step=0.1,
                format="%.3f",
                key=f"static_{backend_name}",
            )

        st.markdown("#### 📊 30-Day Activity Pattern")
        st.caption("Daily virtual learning environment activity (e.g., clicks or interactions).")

        with st.expander("Edit 30-day activity sequence"):
            for i in range(SEQ_LEN):
                default_seq = float(np.random.randint(0, 60))
                v = st.number_input(
                    f"Day {i+1}",
                    value=default_seq,
                    min_value=0.0,
                    step=1.0,
                    key=f"seq_day_{i}",
                )
                seq_values.append(v)

        submit = st.form_submit_button("🔍 Run Prediction")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------- RIGHT: Predictions & Explanations ----------------------
with col_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Predictions</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Predicted outcome, risk characterisation, and confidence from XGBoost and (where available) HTBT.</div>",
        unsafe_allow_html=True,
    )

    if not submit:
        st.info("Enter or adjust the indicators on the left and click **Run Prediction** to see results and explanations.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Build full feature vector for XGBoost (length = len(FEATURES))
        x_vec = np.zeros(len(FEATURES), dtype=np.float32)
        for fname, idx in SIMPLIFIED_IDX.items():
            x_vec[idx] = float(static_values[fname])

        # ---------- XGBoost prediction ----------
        proba = xgb_model.predict_proba(x_vec.reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
        pred_class = int(XGB_CLASSES[pred_idx])
        pred_name = CLASS_LABELS.get(pred_class, f"Class {pred_class}")
        conf = float(np.max(proba))
        risk_text, risk_css = risk_label_from_class(pred_class, conf)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<span class='small-label'>XGBoost Predicted Outcome</span>", unsafe_allow_html=True)
            st.markdown(
                f"<h3 style='margin-top:0.25rem;'>{pred_name}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<span class='{risk_css}'>{risk_text}</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------- HTBT prediction ----------
        if htbt_model is None:
            with c2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<span class='small-label'>HTBT Prediction</span>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top:0.25rem;'>Unavailable</h3>", unsafe_allow_html=True)
                st.markdown(
                    "<span class='risk-pill risk-med'>HTBT model file could not be loaded in this environment.</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            attn_weights = None
        else:
            static_tensor = torch.tensor(x_vec.reshape(1, -1), dtype=torch.float32)
            seq_tensor = torch.tensor(seq_values, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits_htbt, _, _ = htbt_model(static_tensor, seq_tensor)
                probs_htbt = torch.softmax(logits_htbt, dim=1).cpu().numpy()[0]
                htbt_idx = int(np.argmax(probs_htbt))
                htbt_class = int(XGB_CLASSES[htbt_idx])
                htbt_name = CLASS_LABELS.get(htbt_class, f"Class {htbt_class}")
                htbt_conf = float(np.max(probs_htbt))
                attn_weights = htbt_model.attn_weights

            with c2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<span class='small-label'>HTBT Predicted Outcome</span>", unsafe_allow_html=True)
                st.markdown(
                    f"<h3 style='margin-top:0.25rem;'>{htbt_name}</h3>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<span class='risk-pill risk-low'>Confidence · {htbt_conf*100:.1f}%</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # Explainability (SHAP, LIME, Hybrid)
        # ============================================================
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔍 Explainability (SHAP, LIME, Hybrid)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Feature-level explanations for this profile using SHAP, LIME and a hybrid aggregation.</div>",
            unsafe_allow_html=True,
        )

        col_shap, col_lime = st.columns(2)

        shap_vals_full = None
        lime_res = None

        # ---------- SHAP ----------
        with col_shap:
            st.markdown("<div class='small-label'>SHAP Feature Contributions</div>", unsafe_allow_html=True)
            if shap_explainer is None:
                st.info("SHAP explainer is not available in this environment.")
            else:
                try:
                    sv_all = shap_explainer.shap_values(x_vec.reshape(1, -1))
                    if isinstance(sv_all, list):
                        shap_vals_full = sv_all[pred_idx][0]
                    else:
                        shap_vals_full = sv_all[0]

                    df_shap = pd.DataFrame(
                        {"feature": FEATURES, "shap_value": shap_vals_full}
                    )
                    df_shap = df_shap[df_shap["feature"].isin(SIMPLIFIED.keys())].copy()
                    df_shap["abs_val"] = df_shap["shap_value"].abs()
                    df_shap = df_shap.sort_values("abs_val", ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(
                        x="shap_value",
                        y="feature",
                        data=df_shap,
                        ax=ax,
                        palette="rocket",
                    )
                    ax.set_xlabel("SHAP value (impact on prediction)")
                    ax.set_ylabel("")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.info(f"SHAP explanation could not be generated: {e}")

        # ---------- LIME ----------
        with col_lime:
            st.markdown("<div class='small-label'>LIME Local Explanation</div>", unsafe_allow_html=True)
            if lime_explainer is None:
                st.info("LIME explainer is not available in this environment.")
            else:
                try:
                    lime_res = lime_explainer.explain_instance(
                        x_vec,
                        xgb_model.predict_proba,
                        top_labels=1,
                        num_features=len(FEATURES),
                    )
                    lime_fig = lime_res.as_pyplot_figure(label=pred_idx)
                    lime_fig.set_size_inches(5, 3)
                    plt.tight_layout()
                    st.pyplot(lime_fig)
                    plt.close(lime_fig)
                except Exception as e:
                    st.info(f"LIME explanation could not be generated: {e}")

        # ---------- Hybrid (SHAP + LIME) ----------
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown("<div class='small-label'>Hybrid SHAP + LIME Explanation</div>", unsafe_allow_html=True)

        if shap_vals_full is None or lime_res is None:
            st.info("Hybrid explanation unavailable because SHAP or LIME could not be computed.")
        else:
            try:
                shap_imp = np.abs(shap_vals_full)
                shap_norm = shap_imp / (shap_imp.sum() + 1e-9)

                lime_local = dict(lime_res.local_exp[pred_idx])
                lime_imp = np.zeros(len(FEATURES), dtype=float)
                for idx, w in lime_local.items():
                    if 0 <= idx < len(lime_imp):
                        lime_imp[idx] = abs(w)
                lime_norm = lime_imp / (lime_imp.sum() + 1e-9)

                hybrid = 0.5 * shap_norm + 0.5 * lime_norm

                df_hybrid = pd.DataFrame(
                    {"feature": FEATURES, "hybrid_score": hybrid}
                )
                df_hybrid = df_hybrid[df_hybrid["feature"].isin(SIMPLIFIED.keys())]
                df_hybrid = df_hybrid.sort_values(
                    "hybrid_score", ascending=False
                ).head(10)

                fig_h, ax_h = plt.subplots(figsize=(6, 3))
                sns.barplot(
                    x="hybrid_score",
                    y="feature",
                    data=df_hybrid,
                    ax=ax_h,
                    palette="mako",
                )
                ax_h.set_xlabel("Hybrid importance (normalised SHAP + LIME)")
                ax_h.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig_h)
                plt.close(fig_h)

                st.caption(
                    "The hybrid score averages normalised SHAP and LIME contributions, highlighting features consistently important across both explanation methods."
                )
            except Exception as e:
                st.info(f"Hybrid explanation could not be generated: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================================
        # HTBT temporal attention
        # ============================================================
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>⏱️ HTBT Temporal Attention (30-Day Activity)</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-subtitle'>Relative attention assigned to each day in the 30-day activity window (if HTBT is available).</div>",
            unsafe_allow_html=True,
        )

        if htbt_model is None or attn_weights is None:
            st.info("HTBT temporal attention is not available in this environment.")
        else:
            try:
                # attn_weights: (B, heads, 1, L)
                attn_np = (
                    attn_weights.mean(dim=1)
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                fig_a, ax_a = plt.subplots(figsize=(8, 2.8))
                ax_a.plot(
                    np.arange(1, SEQ_LEN + 1),
                    attn_np,
                    marker="o",
                    linewidth=1.5,
                )
                ax_a.set_xlabel("Day in 30-day window")
                ax_a.set_ylabel("Attention weight")
                ax_a.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_a)
                plt.close(fig_a)

                st.caption(
                    "Higher attention weights indicate days whose activity patterns contributed more strongly to the HTBT prediction."
                )
            except Exception as e:
                st.info(f"Could not visualise HTBT attention: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown(
    "<div class='footer-note'>Dashboard powered by XGBoost and the Hybrid Temporal–Behavioural Transformer (HTBT) with layered SHAP, LIME, and Hybrid explanations.</div>",
    unsafe_allow_html=True,
)
