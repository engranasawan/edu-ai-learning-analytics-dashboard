import math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# ============================================
# Streamlit basic config
# ============================================
st.set_page_config(
    page_title="Learning Analytics – HTBT & XGBoost Dashboard",
    layout="wide",
    page_icon="📊"
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
# HTBT Architecture (inference)
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
        seq = self.seq_proj(seq)  # (B, L, D)
        seq = self.pos_enc(seq)
        seq_enc = self.seq_encoder(seq)  # (B, L, D)
        seq_pool = seq_enc.mean(dim=1)  # (B, D)

        static_emb = self.static_proj(static_x)  # (B, D)

        q = static_emb.unsqueeze(1)  # (B, 1, D)
        k = seq_enc  # (B, L, D)
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
# Cached model loaders
# ============================================
@st.cache_resource
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_final.json")
    return model


@st.cache_resource
def load_htbt_model(n_static, seq_len, num_classes):
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
# Helper functions
# ============================================
def risk_label_from_probs(probs, classes):
    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))

    # assumes lower index = higher risk
    if pred_idx == 0:
        label = "High Risk"
        css = "risk-pill risk-high"
    elif pred_idx == 1:
        label = "Moderate Risk"
        css = "risk-pill risk-med"
    else:
        label = "Lower Risk"
        css = "risk-pill risk-low"
    return label, css, conf, pred_idx


# Simplified, user-facing subset of features
SIMPLIFIED_FEATURES = OrderedDict([
    ("total_clicks", "Total VLE Activity"),
    ("avg_clicks_per_visit", "Average Engagement per Visit"),
    ("active_days", "Active Study Days"),
    ("weighted_score", "Weighted Assessment Score"),
    ("avg_score", "Average Score"),
    ("performance_trend", "Performance Trend Over Time"),
    ("study_duration", "Study Duration"),
    ("engagement_efficiency", "Engagement Efficiency"),
    ("cbii", "Cognitive–Behavioural Index (CBII)"),
    ("tpi", "Temporal Persistence Index (TPI)"),
    ("dropout_risk_proxy", "Dropout Risk Indicator"),
    ("activity_entropy", "Study Pattern Entropy"),
])

SEQ_LEN = 30


def random_default(feature_name: str) -> float:
    """Generate a plausible random default for each feature type.
    Called at runtime -> changes with new sessions/reloads."""
    name = feature_name.lower()
    if "click" in name or "days" in name:
        return float(np.random.randint(20, 800))
    if "score" in name:
        return float(np.random.uniform(30, 90))
    if "trend" in name:
        return float(np.random.uniform(-5, 5))
    if "engagement_efficiency" in name:
        return float(np.random.uniform(0.1, 2.5))
    if "cbii" in name:
        return float(np.random.uniform(0.0, 1.0))
    if "tpi" in name:
        return float(np.random.uniform(0.0, 40.0))
    if "dropout_risk_proxy" in name:
        return float(np.random.uniform(0.0, 1.5))
    if "entropy" in name:
        return float(np.random.uniform(0.0, 3.0))
    if "duration" in name:
        return float(np.random.uniform(20, 250))
    return 0.0


def sample_background(all_feature_names, n_samples=300):
    """Synthetic background for SHAP/LIME (since training data are not loaded here)."""
    bg = np.zeros((n_samples, len(all_feature_names)), dtype=np.float32)
    for j, fname in enumerate(all_feature_names):
        if fname in SIMPLIFIED_FEATURES:
            vals = [random_default(fname) for _ in range(n_samples)]
            bg[:, j] = np.array(vals)
        else:
            bg[:, j] = 0.0
    return bg


# ============================================
# Load models & set up explainers
# ============================================
def load_xgb_model():
    model = xgb.XGBClassifier()
    model.load_model("xgb_final.json")
    return model
xgb_classes = getattr(xgb_model, "classes_", np.arange(4))

# Full static feature list as seen by the model
if hasattr(xgb_model, "feature_names_in_"):
    ALL_FEATURES = list(xgb_model.feature_names_in_)
else:
    # Fallback if not present
    ALL_FEATURES = list(SIMPLIFIED_FEATURES.keys())

num_static_features = len(ALL_FEATURES)
num_classes = len(xgb_classes)

htbt_model = load_htbt_model(
    n_static=num_static_features,
    seq_len=SEQ_LEN,
    num_classes=num_classes,
)

background_data = sample_background(ALL_FEATURES, n_samples=300)
shap_explainer = shap.TreeExplainer(xgb_model)
lime_explainer = LimeTabularExplainer(
    training_data=background_data,
    feature_names=ALL_FEATURES,
    class_names=[str(c) for c in xgb_classes],
    discretize_continuous=True,
    random_state=42,
)

# ============================================
# Header
# ============================================
st.markdown(
    "<h1 style='margin-bottom:0.25rem;'>📊 Hybrid Temporal–Behavioural Analytics Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#9ca3af;margin-top:0;'>Enter student indicators to obtain predictions from XGBoost and HTBT, together with layered SHAP, LIME, and hybrid explanations.</p>",
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
    st.markdown("<div class='section-title'>📝 Student Information</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-subtitle'>Adjust the indicators below to reflect a specific student profile. Default values change across sessions.</div>",
        unsafe_allow_html=True,
    )

    static_inputs = {}
    seq_inputs = []

    with st.form("feature_input_form"):
        st.markdown("#### 🔧 Key Indicators")

        for f, label in SIMPLIFIED_FEATURES.items():
            default_val = random_default(f)
            static_inputs[f] = st.number_input(
                label,
                value=float(default_val),
                step=0.1,
                format="%.3f",
            )

        st.markdown("#### 📊 Study Activity Over Past 30 Days")
        st.caption(
            "Daily virtual learning environment activity (e.g., number of clicks or interactions per day)."
        )

        with st.expander("View / Edit 30-Day Activity Sequence"):
            for i in range(SEQ_LEN):
                default_seq = float(np.random.randint(0, 60))
                seq_val = st.number_input(
                    f"Day {i+1} Activity",
                    value=default_seq,
                    min_value=0.0,
                    step=1.0,
                    key=f"seq_day_{i}",
                )
                seq_inputs.append(seq_val)

        run_inference = st.form_submit_button("🔍 Check Results")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# RIGHT: Predictions & Explanations
# ------------------------------------------------
with col_right:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'>🎯 Model Predictions</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-subtitle'>Predicted outcome, risk characterisation, and confidence from XGBoost and HTBT.</div>",
        unsafe_allow_html=True,
    )

    if not run_inference:
        st.info("Enter or adjust the inputs on the left and click **“Check Results”** to generate predictions and explanations.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # -------------------------
        # Build full feature vector
        # -------------------------
        x_full = np.zeros(len(ALL_FEATURES), dtype=np.float32)
        for j, fname in enumerate(ALL_FEATURES):
            if fname in SIMPLIFIED_FEATURES:
                x_full[j] = float(static_inputs[fname])
            else:
                x_full[j] = 0.0

        xgb_input = x_full.reshape(1, -1)

        # -------------------------
        # XGBoost predictions
        # -------------------------
        xgb_probs = xgb_model.predict_proba(xgb_input)[0]
        risk_label, risk_css, conf, pred_idx = risk_label_from_probs(
            xgb_probs, xgb_classes
        )

        c_pred_top, c_pred_bottom = st.columns(2)

        with c_pred_top:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(
                "<span class='small-label'>XGBoost Predicted Outcome</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='margin-top:0.25rem;'>Class {int(xgb_classes[pred_idx])}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='{risk_css}'>{risk_label} · {conf*100:.1f}% confidence</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # HTBT prediction & attention
        # -------------------------
        static_tensor = torch.tensor(x_full.reshape(1, -1), dtype=torch.float32)
        seq_tensor = torch.tensor(seq_inputs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits_htbt, _, _ = htbt_model(static_tensor, seq_tensor)
            probs_htbt = torch.softmax(logits_htbt, dim=1).cpu().numpy()[0]
            htbt_pred_idx = int(np.argmax(probs_htbt))
            htbt_conf = float(np.max(probs_htbt))
            attn = htbt_model.attn_weights  # (B, heads, 1, L)

        with c_pred_bottom:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(
                "<span class='small-label'>HTBT Predicted Outcome</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<h3 style='margin-top:0.25rem;'>Class {int(xgb_classes[htbt_pred_idx])}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span class='risk-pill risk-low'>Confidence · {htbt_conf*100:.1f}%</span>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # Explanations: SHAP, LIME, Hybrid
        # ============================================
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>🔍 Local Explanations (XGBoost)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Feature-level explanations for this specific profile using SHAP, LIME, and a hybrid aggregation.</div>",
            unsafe_allow_html=True,
        )

        c_shap, c_lime = st.columns(2)

        # -------------------------
        # SHAP local explanation
        # -------------------------
        with c_shap:
            st.markdown(
                "<div class='small-label'>SHAP Feature Contributions</div>",
                unsafe_allow_html=True,
            )
            try:
                shap_values_instance = shap_explainer.shap_values(xgb_input)
                # shap_values_instance is a list (per class) for multiclass
                if isinstance(shap_values_instance, list):
                    shap_vals = shap_values_instance[pred_idx][0]
                else:
                    shap_vals = shap_values_instance[0]

                shap_df = pd.DataFrame(
                    {
                        "feature": ALL_FEATURES,
                        "shap_value": shap_vals,
                    }
                )
                # Focus display on simplified features
                shap_df = shap_df[
                    shap_df["feature"].isin(SIMPLIFIED_FEATURES.keys())
                ].copy()
                shap_df["abs_val"] = shap_df["shap_value"].abs()
                shap_df = shap_df.sort_values("abs_val", ascending=False).head(10)

                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(
                    x="shap_value",
                    y="feature",
                    data=shap_df,
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

        # -------------------------
        # LIME local explanation
        # -------------------------
        with c_lime:
            st.markdown(
                "<div class='small-label'>LIME Local Explanation</div>",
                unsafe_allow_html=True,
            )
            try:
                lime_exp = lime_explainer.explain_instance(
                    x_full,
                    xgb_model.predict_proba,
                    num_features=min(12, len(ALL_FEATURES)),
                    top_labels=1,
                )
                fig_lime = lime_exp.as_pyplot_figure(label=pred_idx)
                fig_lime.set_size_inches(5, 3)
                plt.tight_layout()
                st.pyplot(fig_lime)
                plt.close(fig_lime)
            except Exception as e:
                st.info(f"LIME explanation not available: {e}")

        # -------------------------
        # Hybrid SHAP + LIME
        # -------------------------
        st.markdown("<hr style='border-color:#334155;'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='small-label'>Hybrid SHAP + LIME Explanation</div>",
            unsafe_allow_html=True,
        )
        try:
            # SHAP importance vector
            if isinstance(shap_values_instance, list):
                shap_vals = shap_values_instance[pred_idx][0]
            else:
                shap_vals = shap_values_instance[0]
            shap_imp = np.abs(shap_vals)
            shap_imp_norm = shap_imp / (shap_imp.sum() + 1e-9)

            # LIME importance vector (per feature index)
            lime_local = dict(lime_exp.local_exp[pred_idx])
            lime_imp = np.zeros_like(shap_imp, dtype=float)
            for idx, weight in lime_local.items():
                if 0 <= idx < len(lime_imp):
                    lime_imp[idx] = abs(weight)
            lime_imp_norm = lime_imp / (lime_imp.sum() + 1e-9)

            # Hybrid: simple average
            hybrid_score = 0.5 * shap_imp_norm + 0.5 * lime_imp_norm

            hybrid_df = pd.DataFrame(
                {
                    "feature": ALL_FEATURES,
                    "hybrid_score": hybrid_score,
                    "shap_norm": shap_imp_norm,
                    "lime_norm": lime_imp_norm,
                }
            )
            hybrid_df = hybrid_df[
                hybrid_df["feature"].isin(SIMPLIFIED_FEATURES.keys())
            ].copy()
            hybrid_df = hybrid_df.sort_values(
                "hybrid_score", ascending=False
            ).head(10)

            fig_h, ax_h = plt.subplots(figsize=(6, 3))
            sns.barplot(
                x="hybrid_score",
                y="feature",
                data=hybrid_df,
                ax=ax_h,
                palette="mako",
            )
            ax_h.set_xlabel("Hybrid importance (SHAP + LIME, normalised)")
            ax_h.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig_h)
            plt.close(fig_h)

            st.caption(
                "The hybrid importance score averages normalised SHAP and LIME contributions, encouraging agreement between model-consistent and perturbed local explanations."
            )
        except Exception as e:
            st.info(f"Hybrid explanation not available: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # ============================================
        # HTBT Temporal Attention Visualisation
        # ============================================
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-title'>⏱️ HTBT Temporal Attention</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='section-subtitle'>Relative attention assigned to each day in the 30-day activity sequence for this student profile.</div>",
            unsafe_allow_html=True,
        )

        try:
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
                    "Higher attention weights indicate days that were particularly influential for the HTBT prediction, given the entered activity pattern."
                )
            else:
                st.info("Attention weights not available from the HTBT model.")
        except Exception as e:
            st.info(f"Could not visualise HTBT attention: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown(
    "<div class='footer-note'>Dashboard powered by XGBoost and Hybrid Temporal–Behavioural Transformer (HTBT) with layered explainability (SHAP, LIME, Hybrid).</div>",
    unsafe_allow_html=True,
)
