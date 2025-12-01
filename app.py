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
import xgboost as xgb
from collections import OrderedDict

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
    .risk-pill {
        padding: 0.3rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #0b1120;
        display: inline-block;
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
# Simplified feature set (user-facing names)
# ============================================
SIMPLIFIED_FEATURES = OrderedDict([
    ("total_clicks",           "Total VLE Activity"),
    ("avg_clicks_per_visit",   "Average Clicks per Visit"),
    ("active_days",            "Active Study Days"),
    ("weighted_score",         "Weighted Assessment Score"),
    ("avg_score",              "Average Score"),
    ("performance_trend",      "Performance Trend"),
    ("study_duration",         "Study Duration (Days)"),
    ("engagement_efficiency",  "Engagement Efficiency"),
    ("cbii",                   "Cognitive–Behavioural Index (CBII)"),
    ("tpi",                    "Temporal Persistence Index (TPI)"),
    ("dropout_risk_proxy",     "Dropout Risk Indicator"),
    ("activity_entropy",       "Study Entropy"),
])

SEQ_LEN = 30  # HTBT sequence length used in training (last 30 days)

# ============================================
# HTBT Architecture (same as training)
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, : x.size(1), :].to(x.dtype)
        return x


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

        # Daily clicks -> embedding
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

        # Static features -> embedding
        self.static_proj = nn.Sequential(
            nn.Linear(n_static, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Cross-attention (static → sequence)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Fusion + classifier
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
        seq = self.seq_proj(seq)   # (B, L, D)
        seq = self.pos_enc(seq)
        seq_enc = self.seq_encoder(seq)  # (B, L, D)
        seq_pool = seq_enc.mean(dim=1)   # (B, D)

        static_emb = self.static_proj(static_x)  # (B, D)

        q = static_emb.unsqueeze(1)  # (B, 1, D)
        k = seq_enc                  # (B, L, D)
        v = seq_enc
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_out = attn_out.squeeze(1)  # (B, D)
        self.attn_weights = attn_weights  # (B, heads, 1, L)

        fused = torch.cat([static_emb, attn_out], dim=-1)
        fused = self.fusion_mlp(fused)
        logits = self.classifier(fused)
        return logits, seq_pool, static_emb


# ============================================
# Helper functions
# ============================================
def random_default(feature_name: str) -> float:
    """Generate a plausible random default for each feature type.
    Called during UI build so defaults change across reruns."""
    name = feature_name.lower()
    rng = np.random.default_rng()

    if "click" in name or "days" in name:
        return float(rng.integers(20, 800))
    if "score" in name:
        return float(rng.uniform(30, 90))
    if "trend" in name:
        return float(rng.uniform(-5, 5))
    if "engagement_efficiency" in name:
        return float(rng.uniform(0.1, 2.5))
    if "cbii" in name:
        return float(rng.uniform(0.0, 1.0))
    if "tpi" in name:
        return float(rng.uniform(0.0, 40.0))
    if "dropout_risk_proxy" in name:
        return float(rng.uniform(0.0, 1.5))
    if "entropy" in name:
        return float(rng.uniform(0.0, 3.0))
    if "duration" in name:
        return float(rng.uniform(20, 250))
    return float(rng.uniform(0.0, 1.0))


def sample_background(all_feature_names, n_samples=300):
    """Synthetic background for SHAP/LIME (no training data loaded here)."""
    rng = np.random.default_rng()
    bg = np.zeros((n_samples, len(all_feature_names)), dtype=np.float32)
    for j, fname in enumerate(all_feature_names):
        if fname in SIMPLIFIED_FEATURES:
            vals = [random_default(fname) for _ in range(n_samples)]
            bg[:, j] = np.array(vals)
        else:
            bg[:, j] = rng.normal(0.0, 1.0, size=n_samples)
    return bg


def class_label_text(class_index: int) -> str:
    """Human-readable label for encoded class index (generic, model-agnostic)."""
    mapping = {
        0: "Category 0 – Very strong outcome (e.g., high achievement)",
        1: "Category 1 – Medium outcome (e.g., borderline / pass)",
        2: "Category 2 – Satisfactory outcome (e.g., pass / good standing)",
        3: "Category 3 – At-risk outcome (e.g., fail or withdrawal)",
    }
    return mapping.get(class_index, f"Category {class_index} (encoded label)")


def risk_pill_from_conf(conf: float) -> str:
    """Simple visual band based on confidence of prediction."""
    if conf >= 0.85:
        cls = "risk-pill risk-low"
        text = "High confidence"
    elif conf >= 0.6:
        cls = "risk-pill risk-med"
        text = "Moderate confidence"
    else:
        cls = "risk-pill risk-high"
        text = "Low confidence"
    return cls, text


# ============================================
# Cached model loaders
# ============================================
@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_final.json")
    return model


@st.cache_resource
def load_htbt_model(n_static: int, seq_len: int, num_classes: int):
    model = HTBT(
        n_static=n_static,
        seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        dropout=0.1,
        num_classes=num_classes,
    )
    state_dict = torch.load("htbt_best.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================================
# Load XGBoost and set up feature space
# ============================================
try:
    xgb_model = load_xgb_model()
    XGB_FEATURES = list(getattr(xgb_model, "feature_names_in_", []))
    if not XGB_FEATURES:
        XGB_FEATURES = list(SIMPLIFIED_FEATURES.keys())
    xgb_classes = getattr(xgb_model, "classes_", np.arange(4))
    num_classes = len(xgb_classes)
    xgb_available = True
except Exception as e:
    xgb_model = None
    XGB_FEATURES = list(SIMPLIFIED_FEATURES.keys())
    xgb_classes = np.arange(4)
    num_classes = len(xgb_classes)
    xgb_available = False
    st.error(f"XGBoost model could not be loaded: {e}")

# HTBT was trained on X_static with 33 features = (XGB features) + final_result
N_STATIC_HTBT = len(XGB_FEATURES) + 1

# ============================================
# Load HTBT model (if possible)
# ============================================
try:
    htbt_model = load_htbt_model(
        n_static=N_STATIC_HTBT,
        seq_len=SEQ_LEN,
        num_classes=num_classes,
    )
    htbt_available = True
except Exception as e:
    htbt_model = None
    htbt_available = False
    htbt_error_msg = str(e)

# ============================================
# Explainability objects (SHAP & LIME) – no caching to avoid hash issues
# ============================================
if xgb_available:
    background_data = sample_background(XGB_FEATURES, n_samples=300)
    shap_explainer = shap.TreeExplainer(xgb_model)
    lime_explainer = LimeTabularExplainer(
        training_data=background_data,
        feature_names=XGB_FEATURES,
        class_names=[str(c) for c in xgb_classes],
        discretize_continuous=True,
        random_state=42,
    )
else:
    shap_explainer = None
    lime_explainer = None
    background_data = None

# ============================================
# Header
# ============================================
st.markdown(
    "<h1 style='margin-bottom:0.25rem;'>📊 Hybrid Temporal–Behavioural Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9ca3af;margin-top:0;'>Interactive predictive modelling with XGBoost & HTBT, including SHAP, LIME and Hybrid explanations.</p>",
    unsafe_allow_html=True,
)

# ============================================
# Layout: Left = Inputs, Right = Results & Explanations
# ============================================
col_left, col_right = st.columns([1.1, 1.9])

# ------------------------------------------------
# LEFT: Feature inputs
# ------------------------------------------------
with col_left:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Student Indicators</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Enter key indicators for a student. Default values are randomly initialised on each run so the profile changes on refresh.</div>",
        unsafe_allow_html=True,
    )

    static_inputs = {}
    seq_inputs = []

    with st.form("feature_input_form"):
        st.markdown("#### Key Engagement & Performance Indicators")

        # Static feature inputs
        for f_name, label in SIMPLIFIED_FEATURES.items():
            default_val = random_default(f_name)
            static_inputs[f_name] = st.number_input(
                label,
                value=float(default_val),
                step=0.1,
                format="%.3f",
            )

        st.markdown("#### 30-Day Activity Pattern")
        st.caption("Approximate daily activity in the virtual learning environment over the last 30 days.")

        with st.expander("View / Edit 30-Day Activity Sequence"):
            for i in range(SEQ_LEN):
                default_seq_val = float(np.random.default_rng().integers(0, 60))
                seq_val = st.number_input(
                    f"Day {i+1}",
                    value=default_seq_val,
                    min_value=0.0,
                    step=1.0,
                    key=f"seq_day_{i}",
                )
                seq_inputs.append(seq_val)

        run_inference = st.form_submit_button("🔍 Run Prediction")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# RIGHT: Predictions & Explanations
# ------------------------------------------------
with col_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🎯 Predictions</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Predicted encoded outcome category and confidence from XGBoost and HTBT.</div>",
        unsafe_allow_html=True,
    )

    if not run_inference:
        st.info("Adjust the indicators on the left and click **“Run Prediction”** to generate predictions and explanations.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # -----------------------------------------
        # Build XGBoost feature vector
        # -----------------------------------------
        xgb_vec = np.zeros(len(XGB_FEATURES), dtype=np.float32)
        for j, fname in enumerate(XGB_FEATURES):
            if fname in static_inputs:
                xgb_vec[j] = float(static_inputs[fname])
            else:
                xgb_vec[j] = 0.0
        xgb_input = xgb_vec.reshape(1, -1)

        # -----------------------------------------
        # XGBoost prediction
        # -----------------------------------------
        if xgb_available:
            xgb_probs = xgb_model.predict_proba(xgb_input)[0]
            xgb_pred_idx = int(np.argmax(xgb_probs))
            xgb_conf = float(np.max(xgb_probs))
            pill_cls, pill_text = risk_pill_from_conf(xgb_conf)

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(
                "<span class='small-label'>XGBoost – Encoded Outcome</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='margin-top:0.25rem;'>Class {int(xgb_classes[xgb_pred_idx])}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:0.9rem;margin-top:0.1rem;'>{class_label_text(xgb_pred_idx)}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='{pill_cls}'>{pill_text} · {xgb_conf*100:.1f}% probability</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("XGBoost model is not available, so predictions cannot be computed.")

        # -----------------------------------------
        # HTBT prediction & attention
        # -----------------------------------------
        if htbt_available and htbt_model is not None:
            # Construct static vector for HTBT: [XGB_FEATURES..., extra_slot]
            htbt_vec = np.zeros(N_STATIC_HTBT, dtype=np.float32)
            for j, fname in enumerate(XGB_FEATURES):
                if fname in static_inputs:
                    htbt_vec[j] = float(static_inputs[fname])
                else:
                    htbt_vec[j] = 0.0
            # last index corresponds to encoded final_result during training; here set to 0

            static_tensor = torch.tensor(htbt_vec.reshape(1, -1), dtype=torch.float32)
            seq_tensor = torch.tensor(seq_inputs, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                logits_htbt, _, _ = htbt_model(static_tensor, seq_tensor)
                probs_htbt = torch.softmax(logits_htbt, dim=1).cpu().numpy()[0]
                htbt_pred_idx = int(np.argmax(probs_htbt))
                htbt_conf = float(np.max(probs_htbt))
                attn = htbt_model.attn_weights

            pill_cls_h, pill_text_h = risk_pill_from_conf(htbt_conf)

            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(
                "<span class='small-label'>HTBT – Encoded Outcome</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='margin-top:0.25rem;'>Class {int(xgb_classes[htbt_pred_idx])}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:0.9rem;margin-top:0.1rem;'>{class_label_text(htbt_pred_idx)}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='{pill_cls_h}'>{pill_text_h} · {htbt_conf*100:.1f}% probability</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("HTBT model file could not be loaded in this environment. Only XGBoost predictions are shown.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # Explainability: SHAP, LIME, Hybrid (XGBoost)
        # ============================================
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>🔍 Explainability (XGBoost)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Local feature explanations for this profile using SHAP, LIME, and a hybrid aggregation.</div>",
            unsafe_allow_html=True,
        )

        if xgb_available and shap_explainer is not None and lime_explainer is not None:
            c_shap, c_lime = st.columns(2)

            # -----------------------------
            # SHAP local explanation
            # -----------------------------
            with c_shap:
                st.markdown(
                    "<div class='small-label'>SHAP Feature Contributions</div>",
                    unsafe_allow_html=True,
                )
                try:
                    shap_values_instance = shap_explainer.shap_values(xgb_input)
                    # Handle multiclass / different return shapes
                    if isinstance(shap_values_instance, list):
                        sv = shap_values_instance[xgb_pred_idx][0]
                    else:
                        arr = np.array(shap_values_instance)
                        if arr.ndim == 3:
                            sv = arr[0, :, xgb_pred_idx]
                        else:
                            sv = arr[0]

                    sv = np.asarray(sv).flatten()
                    df_shap = pd.DataFrame(
                        {
                            "feature": XGB_FEATURES,
                            "value": sv,
                        }
                    )
                    # Focus on simplified features
                    df_shap = df_shap[df_shap["feature"].isin(SIMPLIFIED_FEATURES.keys())]
                    df_shap["abs_val"] = df_shap["value"].abs()
                    df_shap = df_shap.sort_values("abs_val", ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.barplot(
                        x="value",
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
                    st.info(f"SHAP explanation not available: {e}")

            # -----------------------------
            # LIME local explanation
            # -----------------------------
            with c_lime:
                st.markdown(
                    "<div class='small-label'>LIME Local Explanation</div>",
                    unsafe_allow_html=True,
                )
                try:
                    lime_exp = lime_explainer.explain_instance(
                        xgb_vec,
                        xgb_model.predict_proba,
                        num_features=min(12, len(XGB_FEATURES)),
                        top_labels=1,
                    )
                    fig_lime = lime_exp.as_pyplot_figure(label=xgb_pred_idx)
                    fig_lime.set_size_inches(5, 3)
                    plt.tight_layout()
                    st.pyplot(fig_lime)
                    plt.close(fig_lime)
                except Exception as e:
                    st.info(f"LIME explanation not available: {e}")

            # -----------------------------
            # Hybrid SHAP + LIME
            # -----------------------------
            st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='small-label'>Hybrid SHAP + LIME Explanation</div>",
                unsafe_allow_html=True,
            )
            try:
                # SHAP importance
                if isinstance(shap_values_instance, list):
                    sv = shap_values_instance[xgb_pred_idx][0]
                else:
                    arr = np.array(shap_values_instance)
                    if arr.ndim == 3:
                        sv = arr[0, :, xgb_pred_idx]
                    else:
                        sv = arr[0]
                sv = np.asarray(sv).flatten()
                shap_imp = np.abs(sv)
                shap_imp_norm = shap_imp / (shap_imp.sum() + 1e-9)

                # LIME importance
                lime_local = dict(lime_exp.local_exp[xgb_pred_idx])
                lime_imp = np.zeros_like(shap_imp, dtype=float)
                for idx, weight in lime_local.items():
                    if 0 <= idx < len(lime_imp):
                        lime_imp[idx] = abs(weight)
                lime_imp_norm = lime_imp / (lime_imp.sum() + 1e-9)

                # Hybrid average
                hybrid_score = 0.5 * shap_imp_norm + 0.5 * lime_imp_norm

                df_hybrid = pd.DataFrame(
                    {
                        "feature": XGB_FEATURES,
                        "hybrid_score": hybrid_score,
                        "shap_norm": shap_imp_norm,
                        "lime_norm": lime_imp_norm,
                    }
                )
                df_hybrid = df_hybrid[
                    df_hybrid["feature"].isin(SIMPLIFIED_FEATURES.keys())
                ].copy()
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
                    "The hybrid score averages normalised SHAP and LIME contributions, highlighting features where both methods agree on local importance."
                )
            except Exception as e:
                st.info(f"Hybrid explanation not available: {e}")
        else:
            st.info("Explainability could not be initialised because the XGBoost model is unavailable.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # HTBT Temporal Attention
        # ============================================
        if htbt_available and htbt_model is not None:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-title'>⏱️ HTBT Temporal Attention</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='section-subtitle'>Relative attention assigned to each day in the 30-day activity sequence for this profile.</div>",
                unsafe_allow_html=True,
            )

            try:
                attn = htbt_model.attn_weights
                if attn is not None:
                    # attn: (B, heads, 1, L)
                    attn_np = (
                        attn.mean(dim=1)
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()[0]
                    )  # (L,)
                    fig_a, ax_a = plt.subplots(figsize=(8, 2.8))
                    ax_a.plot(
                        range(1, SEQ_LEN + 1),
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
                        "Higher attention weights indicate days that were especially influential for the HTBT prediction, given the entered activity pattern."
                    )
                else:
                    st.info("Attention weights are not available from the HTBT model.")
            except Exception as e:
                st.info(f"Could not visualise HTBT attention: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown(
    "<div class='footer-note'>Encoded classes (0–3) correspond to the label encoding used for the original <code>target_result</code> variable (e.g., Distinction, Fail, Pass, Withdrawn). Interpretations of categories should follow that mapping.</div>",
    unsafe_allow_html=True,
)
