import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import shap
from lime.lime_tabular import LimeTabularExplainer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Trustworthy AI in Education", layout="wide")
st.title("🎓 Trustworthy AI in Education Dashboard")
st.markdown("**XGBoost + Hybrid Temporal-Behavioral Transformer (HTBT) with SHAP • LIME • Hybrid Explanations**")

# ==============================
# 1. Load Models
# ==============================
@st.cache_resource
def load_models():
    with st.spinner("Loading models..."):
        xgb = joblib.load("xgb_final.joblib")
        htbt = HTBT(n_static=47, num_classes=4)
        htbt.load_state_dict(torch.load("htbt_best.pt", map_location="cpu"))
        htbt.eval()
    return xgb, htbt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class HTBT(nn.Module):
    def __init__(self, n_static, seq_len=30, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, num_classes=4):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.static_proj = nn.Sequential(nn.Linear(n_static, d_model), nn.ReLU(), nn.LayerNorm(d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.fusion_mlp = nn.Sequential(nn.Linear(d_model*2, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(d_ff), nn.Linear(d_ff, d_model))
        self.classifier = nn.Sequential(nn.Linear(d_model, d_ff//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff//2, num_classes))

    def forward(self, static_x, seq_x):
        seq = seq_x.unsqueeze(-1)
        seq = self.seq_proj(seq)
        seq = self.pos_enc(seq)
        seq_enc = self.seq_encoder(seq)
        static_emb = self.static_proj(static_x)
        q = static_emb.unsqueeze(1)
        attn_out, _ = self.cross_attn(q, seq_enc, seq_enc)
        attn_out = attn_out.squeeze(1)
        fused = torch.cat([static_emb, attn_out], dim=-1)
        fused = self.fusion_mlp(fused)
        logits = self.classifier(fused)
        return logits

xgb_model, htbt_model = load_models()

# ==============================
# 2. Explainers (SHAP + LIME)
# ==============================
@st.cache_resource
def get_explainers():
    # SHAP
    shap_explainer = shap.TreeExplainer(xgb_model)
    
    # LIME - create realistic background dataset (100 varied rows)
    np.random.seed(42)
    background = np.random.normal(0, 1, size=(100, 47))  # standardized scale like your training data
    feature_names = [f"feature_{i}" for i in range(47)]
    
    lime_explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=["Distinction", "Pass", "Fail", "Withdrawn"],
        mode="classification",
        discretize_continuous=True
    )
    return shap_explainer, lime_explainer

shap_explainer, lime_explainer = get_explainers()

# Nice feature names for display
nice_names = [
    "Gender", "Region", "Highest Education", "IMD Band", "Age Band", "Disability",
    "Previous Attempts", "Studied Credits", "Total Clicks", "Avg Score",
    "Avg Clicks/Visit", "Max Clicks", "Std Clicks", "Active Days", "Unique Sites",
    "Clicks per Day", "Revisit Ratio", "Activity Entropy", "Avg Gap Logins",
    "Burstiness", "Study Duration", "First Registration", "Last Activity Gap",
    "Num Assessments", "First Submission", "Last Submission", "Performance Trend",
    "Engagement Efficiency", "CBII", "TPI", "Dropout Risk Proxy"
] + [f"Extra_{i}" for i in range(47-len([
    "Gender", "Region", "Highest Education", "IMD Band", "Age Band", "Disability",
    "Previous Attempts", "Studied Credits", "Total Clicks", "Avg Score",
    "Avg Clicks/Visit", "Max Clicks", "Std Clicks", "Active Days", "Unique Sites",
    "Clicks per Day", "Revisit Ratio", "Activity Entropy", "Avg Gap Logins",
    "Burstiness", "Study Duration", "First Registration", "Last Activity Gap",
    "Num Assessments", "First Submission", "Last Submission", "Performance Trend",
    "Engagement Efficiency", "CBII", "TPI", "Dropout Risk Proxy"
]))]

class_names = ["Distinction", "Pass", "Fail", "Withdrawn"]
risk_level = ["Very Low", "Low", "High", "Very High"]

# ==============================
# Sidebar
# ==============================
st.sidebar.header("👤 Student Profile")
role = st.sidebar.selectbox("Role", ["Instructor", "Academic Advisor", "Administrator"])

c1, c2 = st.sidebar.columns(2)
age_band = c1.selectbox("Age Band", [0,1,2], format_func=lambda x: ["0-35", "35-55", "55+"][x])
gender = c2.selectbox("Gender", [0,1], format_func=lambda x: ["Male","Female"][x])

region = st.sidebar.selectbox("Region", list(range(13)), format_func=lambda x: ["East Anglian","Scotland","North Western","South East","West Midlands","Wales","North","South","Ireland","South West","East Midlands","Yorkshire","London"][x])
highest_education = st.sidebar.selectbox("Highest Education", [0,1,2,3,4], format_func=lambda x: ["No Formal","Lower Than A Level","A Level","HE Qualification","Post Graduate"][x])
imd_band = st.sidebar.slider("IMD Band (0=most deprived)",0,9,5)
disability = st.sidebar.selectbox("Disability",[0,1], format_func=lambda x: ["No","Yes"][x])

st.sidebar.header("Academic & Engagement")
prev_attempts = st.sidebar.slider("Previous Attempts",0,6,0)
studied_credits = st.sidebar.slider("Studied Credits",30,360,60)
total_clicks = st.sidebar.slider("Total VLE Clicks",0,10000,1200)
avg_score = st.sidebar.slider("Average Score",0,100,70)

st.sidebar.header("Last 30 Days Click Pattern")
sequence_input = st.sidebar.text_input(
    "Clicks per day (comma-separated)",
    value="50,30,0,80,120,90,0,0,150,200,180,160,0,0,220,250,300,280,0,0,100,90,120,140,160,180,200,220,250,300"
)

# ==============================
# Prediction
# ==============================
if st.sidebar.button("Predict & Explain", type="primary", use_container_width=True):
    with st.spinner("Running models + explanations..."):
        # Sequence
        try:
            seq = [float(x.strip()) for x in sequence_input.split(",")]
            seq = seq[:30] + [0]*(30-len(seq))
            seq_tensor = torch.tensor([seq], dtype=torch.float32)
        except:
            st.error("Invalid sequence. Using zeros.")
            seq_tensor = torch.zeros(1,30)

        # Static vector (47-dim)
        static_list = [
            gender, region, highest_education, imd_band, age_band, disability,
            prev_attempts, studied_credits, total_clicks, avg_score
        ] + [0.0] * (47 - 10)  # pad remaining
        static_np = np.array([static_list], dtype=np.float32)
        static_tensor = torch.tensor(static_np)

        # Predictions
        xgb_pred = xgb_model.predict(static_np)[0]
        xgb_prob = xgb_model.predict_proba(static_np)[0]
        xgb_conf = xgb_prob[xgb_pred]

        with torch.no_grad():
            logits = htbt_model(static_tensor, seq_tensor)
            htbt_prob = torch.softmax(logits, dim=1)[0].numpy()
            htbt_pred = int(htbt_prob.argmax())
            htbt_conf = htbt_prob[htbt_pred]

        final_pred = xgb_pred if xgb_conf > htbt_conf else htbt_pred

        # ==============================
        # Explanations: SHAP + LIME + Hybrid
        # ==============================
        # SHAP
        shap_values = shap_explainer.shap_values(static_np)
        if isinstance(shap_values, list):  # multiclass
            shap_vals = shap_values[xgb_pred]
        shap_abs = np.abs(shap_vals[0])
        
        # LIME
        lime_exp = lime_explainer.explain_instance(
            static_np[0], 
            xgb_model.predict_proba, 
            num_features=15
        )
        lime_dict = dict(lime_exp.as_list())
        lime_vals = np.array([lime_dict.get(f"feature_{i}", 0.0) for i in range(47)])
        lime_abs = np.abs(lime_vals)

        # Hybrid (normalized + weighted, alpha=0.6 for SHAP)
        shap_norm = shap_abs / (shap_abs.max() + 1e-8)
        lime_norm = lime_abs / (lime_abs.max() + 1e-8)
        hybrid = 0.6 * shap_norm + 0.4 * lime_norm

        # Create unified dataframe
        explain_df = pd.DataFrame({
            "Feature": nice_names,
            "SHAP": shap_vals[0],
            "LIME": lime_vals,
            "Hybrid": hybrid
        }).round(4)

        explain_df["SHAP_abs"] = explain_df["SHAP"].abs()
        explain_df["LIME_abs"] = explain_df["LIME"].abs()
        top_hybrid = explain_df.sort_values("Hybrid", ascending=False).head(12)

        # ==============================
        # Display Results
        # ==============================
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("XGBoost", class_names[xgb_pred], f"{xgb_conf:.1%}")
        with col2:
            st.metric("HTBT", class_names[htbt_pred], f"{htbt_conf:.1%}")
        with col3:
            st.metric("Final", class_names[final_pred], risk_level[final_pred], delta=None)

        st.success(f"**{role} View**: Student is at **{risk_level[final_pred]} risk** of dropout/withdrawal.")

        # Tabs for explanations
        tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Hybrid (SHAP+LIME)"])

        with tab1:
            fig_shap = px.bar(explain_df.sort_values("SHAP_abs", ascending=False).head(12),
                              x="SHAP", y="Feature", orientation='h', color="SHAP", color_continuous_scale="RdBu")
            st.plotly_chart(fig_shap, use_container_width=True)

        with tab2:
            fig_lime = px.bar(explain_df.sort_values("LIME_abs", ascending=False).head(12),
                              x="LIME", y="Feature", orientation='h', color="LIME", color_continuous_scale="RdBu")
            st.plotly_chart(fig_lime, use_container_width=True)

        with tab3:
            fig_hybrid = px.bar(top_hybrid, x="Hybrid", y="Feature", orientation='h', color="Hybrid", color_continuous_scale="Viridis",
                                title="Hybrid SHAP-LIME Importance (Top 12)")
            st.plotly_chart(fig_hybrid, use_container_width=True)

        st.dataframe(explain_df.style.background_gradient(cmap="RdBu", subset=["SHAP", "LIME"]), use_container_width=True)

st.info("👈 Adjust student data → Click **Predict & Explain** to see full SHAP + LIME + Hybrid explanations!")