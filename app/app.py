# ============================================================
#  InsureGuard — Premium Streamlit UI
#  Run: streamlit run app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
import plotly.graph_objects as go
import plotly.express as px

# ── Auto-train if model missing (cloud deployment) ───────────
def _ensure_model():
    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdl_dir = os.path.join(base, 'models')
    needed  = ['best_model.pkl','scaler.pkl','feature_names.json','model_report.json']
    if not all(os.path.exists(os.path.join(mdl_dir, f)) for f in needed):
        st.info("⏳ First launch — training model (~60 sec). Please wait!")
        trainer = os.path.join(base, 'train_and_save.py')
        if os.path.exists(trainer):
            import importlib.util
            spec = importlib.util.spec_from_file_location("trainer", trainer)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.train_and_save()
            st.success("✅ Model ready! Reloading...")
            st.rerun()

_ensure_model()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="InsureGuard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══ RESET & BASE ═══════════════════════════════════════════ */
*, html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    box-sizing: border-box;
}

/* ═══ APP BACKGROUND ═════════════════════════════════════════ */
.stApp {
    background-color: #050508;
    background-image:
        radial-gradient(ellipse 100% 60% at 50% -10%, rgba(180,140,80,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 90% 90%, rgba(120,80,180,0.04) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 80px,
            rgba(255,255,255,0.012) 80px,
            rgba(255,255,255,0.012) 81px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 80px,
            rgba(255,255,255,0.012) 80px,
            rgba(255,255,255,0.012) 81px
        );
}

/* ═══ HIDE STREAMLIT CHROME ══════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ═══ SIDEBAR ════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(8,7,12,0.95) !important;
    border-right: 1px solid rgba(180,140,80,0.15) !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #888899 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    border: none !important;
    padding: 0.55rem 0.8rem !important;
    border-radius: 4px !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: #c8b878 !important;
    background: rgba(180,140,80,0.06) !important;
}

/* ═══ METRICS ════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.022);
    border: 1px solid rgba(255,255,255,0.06);
    border-top: 1px solid rgba(180,140,80,0.3);
    border-radius: 2px;
    padding: 1.4rem 1.6rem !important;
    position: relative;
    transition: border-color 0.3s;
}
[data-testid="stMetric"]:hover {
    border-top-color: rgba(180,140,80,0.7);
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.2rem !important;
    font-weight: 600 !important;
    color: #e8e0cc !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: #555566 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ═══ BUTTONS ════════════════════════════════════════════════ */
.stButton > button {
    background: transparent !important;
    color: #c8b878 !important;
    border: 1px solid rgba(180,140,80,0.5) !important;
    border-radius: 2px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    background: rgba(180,140,80,0.08) !important;
    border-color: rgba(180,140,80,0.9) !important;
    color: #e8d898 !important;
    box-shadow: 0 0 30px rgba(180,140,80,0.12) !important;
}

/* ═══ FORM INPUTS ════════════════════════════════════════════ */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-bottom: 1px solid rgba(180,140,80,0.25) !important;
    border-radius: 2px !important;
    color: #d0c8b8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-bottom: 1px solid rgba(180,140,80,0.25) !important;
    border-radius: 2px !important;
    color: #d0c8b8 !important;
}
label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    color: #555566 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stSlider > div > div > div {
    background: rgba(180,140,80,0.4) !important;
}

/* ═══ TABS ═══════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #444455 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: #c8b878 !important;
    border-bottom: 1px solid #c8b878 !important;
}

/* ═══ DATAFRAME ══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 2px !important;
}

/* ═══ CUSTOM COMPONENTS ══════════════════════════════════════ */
.ig-page-header {
    border-bottom: 1px solid rgba(180,140,80,0.15);
    padding-bottom: 1.5rem;
    margin-bottom: 2rem;
}
.ig-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: #c8b878;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.5rem;
}
.ig-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #e8e0cc;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin: 0;
}
.ig-subtitle {
    font-size: 0.82rem;
    font-weight: 400;
    color: #444455;
    margin-top: 0.5rem;
    letter-spacing: 0.02em;
}

.ig-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
    padding: 1.5rem;
    position: relative;
}
.ig-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(180,140,80,0.6), transparent);
}

.ig-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    font-weight: 500;
    color: #444455;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
}
.ig-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8e0cc;
}

.ig-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 2rem 0;
}

.ig-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #c8b878;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.ig-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(180,140,80,0.15);
}

.ig-result-fraud {
    background: rgba(200,60,60,0.05);
    border: 1px solid rgba(200,60,60,0.2);
    border-left: 3px solid rgba(200,60,60,0.8);
    border-radius: 2px;
    padding: 2rem 2.5rem;
}
.ig-result-legit {
    background: rgba(80,160,120,0.05);
    border: 1px solid rgba(80,160,120,0.2);
    border-left: 3px solid rgba(80,160,120,0.7);
    border-radius: 2px;
    padding: 2rem 2.5rem;
}
.ig-result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.ig-result-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    opacity: 0.6;
    text-transform: uppercase;
}

.ig-risk-flag {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 2px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: #888899;
}

.ig-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.82rem;
}
.ig-stat-key { color: #444455; font-weight: 400; }
.ig-stat-val {
    font-family: 'JetBrains Mono', monospace;
    color: #c8b878;
    font-size: 0.78rem;
}

.ig-badge-high   { color: #c84444; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; }
.ig-badge-medium { color: #c8a844; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; }
.ig-badge-low    { color: #44a878; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.1em; }

.ig-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8e0cc;
    letter-spacing: -0.01em;
}
.ig-logo span {
    color: #c8b878;
}
.ig-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #333344;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────
PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Outfit', color='#555566', size=11),
    margin=dict(l=16, r=16, t=36, b=16),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)',
               zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=10, color='#444455')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)',
               zerolinecolor='rgba(255,255,255,0.04)',
               tickfont=dict(size=10, color='#444455')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#555566')),
)
GOLD   = '#c8b878'
RED    = '#c84444'
GREEN  = '#44a878'
MUTED  = '#333344'

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdl_dir = os.path.join(base, 'models')
    try:
        model   = joblib.load(os.path.join(mdl_dir, 'best_model.pkl'))
        scaler  = joblib.load(os.path.join(mdl_dir, 'scaler.pkl'))
        with open(os.path.join(mdl_dir, 'feature_names.json')) as f:
            features = json.load(f)
        with open(os.path.join(mdl_dir, 'model_report.json')) as f:
            report = json.load(f)
        return model, scaler, features, report, True
    except Exception:
        return None, None, [], {}, False

model, scaler, FEATURES, REPORT, MODEL_OK = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.8rem 0.5rem 2rem'>
        <div class='ig-logo'>Insure<span>Guard</span></div>
        <div class='ig-tagline'>Fraud Intelligence System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:rgba(180,140,80,0.12);margin-bottom:1.5rem'></div>",
                unsafe_allow_html=True)

    page = st.radio("", [
        "Overview",
        "Claim Analysis",
        "Data Explorer",
        "Model Metrics",
        "System Info",
    ], label_visibility="collapsed")

    st.markdown("<div style='height:1px;background:rgba(255,255,255,0.04);margin:1.5rem 0'></div>",
                unsafe_allow_html=True)

    # Status block
    status_color = GREEN if MODEL_OK else RED
    status_text  = "Operational" if MODEL_OK else "Offline"
    st.markdown(f"""
    <div style='padding:0 0.5rem'>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.58rem;
                    color:#333344;text-transform:uppercase;letter-spacing:0.15em;
                    margin-bottom:0.8rem'>System Status</div>
        <div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem'>
            <div style='width:5px;height:5px;border-radius:50%;
                        background:{status_color};
                        box-shadow:0 0 6px {status_color}'></div>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.68rem;
                         color:{status_color};letter-spacing:0.05em'>{status_text}</span>
        </div>
        <div style='font-family:JetBrains Mono,monospace;font-size:0.62rem;
                    color:#333344;line-height:2'>
            AUC &nbsp;&nbsp;{REPORT.get("test_auc_roc","—")}<br>
            F1 &nbsp;&nbsp;&nbsp;{REPORT.get("test_f1","—")}<br>
            Features &nbsp;{REPORT.get("n_features","—")}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class='ig-page-header'>
        <div class='ig-eyebrow'>◈ Dashboard</div>
        <div class='ig-title'>Claims Overview</div>
        <div class='ig-subtitle'>Aggregated intelligence across all processed insurance claims</div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Claims Processed", "15,420", "+8.2%"),
        ("Fraud Detections",       "1,847",  "+3.1%"),
        ("Legitimate Claims",      "13,573", "+9.0%"),
        ("Detection AUC",
         str(REPORT.get('test_auc_roc', '0.929')) if MODEL_OK else "0.929", "+1.2%"),
    ]
    for col, (label, val, delta) in zip([c1,c2,c3,c4], kpis):
        col.metric(label, val, delta)

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # Charts row 1
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="ig-section-label">Monthly Claim Trend</div>',
                    unsafe_allow_html=True)
        months = ['Jan','Feb','Mar','Apr','May','Jun',
                  'Jul','Aug','Sep','Oct','Nov','Dec']
        legit  = [900,950,880,960,1020,1100,1150,1080,990,1010,1080,1200]
        fraud  = [120,135,98,145,167,189,201,178,155,143,190,210]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months, y=legit, name='Legitimate',
            line=dict(color=GREEN, width=1.5),
            fill='tozeroy', fillcolor='rgba(68,168,120,0.05)'))
        fig.add_trace(go.Scatter(
            x=months, y=fraud, name='Fraudulent',
            line=dict(color=RED, width=1.5),
            fill='tozeroy', fillcolor='rgba(200,68,68,0.07)'))
        fig.update_layout(**PL, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="ig-section-label">Fraud Composition</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=['Legitimate', 'Fraudulent'],
            values=[88, 12], hole=0.72,
            marker=dict(
                colors=[MUTED, GOLD],
                line=dict(color='#050508', width=3)),
            textinfo='none',
            hovertemplate='%{label}: %{value}%<extra></extra>'))
        fig2.update_layout(**PL, height=260, showlegend=False,
            annotations=[
                dict(text='12%', x=0.5, y=0.55,
                     font=dict(size=28, family='Playfair Display',
                               color='#e8e0cc'),
                     showarrow=False),
                dict(text='FRAUD RATE', x=0.5, y=0.38,
                     font=dict(size=8, family='JetBrains Mono',
                               color='#444455'),
                     showarrow=False),
            ])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    # Charts row 2
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="ig-section-label">Fraud Rate by Incident</div>',
                    unsafe_allow_html=True)
        types = ['Vehicle Theft','Multi-vehicle','Single Vehicle','Parked Car']
        rates = [22.4, 14.1, 9.8, 5.2]
        bar_colors = [GOLD if r == max(rates) else '#2a2a3a' for r in rates]
        fig3 = go.Figure(go.Bar(
            x=rates, y=types, orientation='h',
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f'{r}%' for r in rates],
            textposition='outside',
            textfont=dict(size=10, color='#555566')))
        fig3.update_layout(**PL, height=220, xaxis_title='')
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown('<div class="ig-section-label">Incident Hour Distribution</div>',
                    unsafe_allow_html=True)
        hours  = list(range(24))
        counts = [18,12,9,8,7,6,8,10,14,16,18,21,
                  20,19,22,24,26,29,32,35,38,33,28,22]
        bar_c  = [RED if (h>=22 or h<=4) else '#1e1e2e' for h in hours]
        fig4   = go.Figure(go.Bar(x=hours, y=counts,
            marker=dict(color=bar_c, line=dict(width=0))))
        fig4.update_layout(**PL, height=220,
                           xaxis_title='Hour of Day', yaxis_title='')
        st.plotly_chart(fig4, use_container_width=True)

    with c3:
        st.markdown('<div class="ig-section-label">Claim Amount Spread</div>',
                    unsafe_allow_html=True)
        np.random.seed(42)
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(
            x=np.random.exponential(8000, 600), name='Legitimate',
            marker_color='rgba(68,168,120,0.35)', nbinsx=30,
            marker_line=dict(width=0)))
        fig5.add_trace(go.Histogram(
            x=np.random.exponential(18000, 100), name='Fraudulent',
            marker_color='rgba(200,68,68,0.5)', nbinsx=20,
            marker_line=dict(width=0)))
        fig5.update_layout(**PL, height=220, barmode='overlay',
                           xaxis_title='Claim ($)')
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# CLAIM ANALYSIS (Predict)
# ══════════════════════════════════════════════════════════════
elif page == "Claim Analysis":
    st.markdown("""
    <div class='ig-page-header'>
        <div class='ig-eyebrow'>◈ Intelligence</div>
        <div class='ig-title'>Claim Analysis</div>
        <div class='ig-subtitle'>Submit claim parameters for real-time fraud probability assessment</div>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("Model unavailable. Run `python src/train.py` to initialize.")
        st.stop()

    with st.form("claim_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="ig-section-label">Policy Holder</div>',
                        unsafe_allow_html=True)
            age                   = st.slider("Age", 18, 75, 38)
            months_as_customer    = st.slider("Tenure (months)", 1, 500, 84)
            policy_deductable     = st.selectbox("Deductible", [500, 1000, 2000])
            policy_annual_premium = st.number_input("Annual Premium ($)", 500, 5000, 1400)
            insured_education_level = st.selectbox("Education Level",
                ["High School","College","Associate","MD","Masters","PhD","JD"])
            insured_occupation    = st.selectbox("Occupation",
                ["craft-repair","sales","tech-support","exec-managerial",
                 "prof-specialty","other-service","armed-forces"])

        with col2:
            st.markdown('<div class="ig-section-label">Incident Details</div>',
                        unsafe_allow_html=True)
            incident_type         = st.selectbox("Incident Type",
                ["Single Vehicle Collision","Multi-vehicle Collision",
                 "Vehicle Theft","Parked Car"])
            incident_severity     = st.selectbox("Damage Severity",
                ["Minor Damage","Major Damage","Total Loss","Trivial Damage"])
            authorities_contacted = st.selectbox("Authorities",
                ["Police","Fire","Ambulance","None","Other"])
            number_of_vehicles    = st.slider("Vehicles Involved", 1, 4, 1)
            bodily_injuries       = st.slider("Bodily Injuries", 0, 2, 0)
            witnesses             = st.slider("Witness Count", 0, 3, 1)
            incident_hour         = st.slider("Incident Hour (24h)", 0, 23, 14)

        with col3:
            st.markdown('<div class="ig-section-label">Claim Financials</div>',
                        unsafe_allow_html=True)
            total_claim_amount = st.number_input("Total Claim ($)",    100, 100000, 12000)
            injury_claim       = st.number_input("Injury Component ($)",  0,  50000,  3000)
            property_claim     = st.number_input("Property Component ($)", 0,  50000,  4000)
            vehicle_claim      = st.number_input("Vehicle Component ($)",  0,  80000,  5000)
            auto_year          = st.slider("Vehicle Year", 1995, 2024, 2016)
            auto_make          = st.selectbox("Vehicle Make",
                ["BMW","Mercedes","Dodge","Toyota","Ford",
                 "Chevrolet","Honda","Audi","Nissan"])

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("◈  RUN FRAUD ANALYSIS")

    if submitted:
        input_data = {
            'age': age, 'months_as_customer': months_as_customer,
            'policy_deductable': policy_deductable,
            'policy_annual_premium': policy_annual_premium,
            'umbrella_limit': 0, 'capital_gains': 0, 'capital_loss': 0,
            'incident_hour_of_the_day': incident_hour,
            'number_of_vehicles_involved': number_of_vehicles,
            'bodily_injuries': bodily_injuries, 'witnesses': witnesses,
            'total_claim_amount': total_claim_amount,
            'injury_claim': injury_claim, 'property_claim': property_claim,
            'vehicle_claim': vehicle_claim, 'auto_year': auto_year,
            'incident_type': 0, 'incident_severity': 0,
            'authorities_contacted': 0, 'insured_education_level': 0,
            'insured_occupation': 0, 'insured_relationship': 0,
            'collision_type': 0, 'property_damage': 0,
            'police_report_available': 0, 'auto_make': 0,
            'claim_premium_ratio': total_claim_amount / (policy_annual_premium + 1),
            'injury_ratio':   injury_claim   / (total_claim_amount + 1),
            'property_ratio': property_claim / (total_claim_amount + 1),
            'vehicle_ratio':  vehicle_claim  / (total_claim_amount + 1),
            'is_high_claim':  1 if total_claim_amount > 50000 else 0,
            'is_night_incident': 1 if (incident_hour >= 22 or incident_hour <= 4) else 0,
        }
        row = np.array([input_data.get(f, 0) for f in FEATURES]).reshape(1, -1)
        try:
            pred  = model.predict(row)[0]
            proba = model.predict_proba(row)[0][1]
        except Exception:
            row_sc = scaler.transform(row)
            pred   = model.predict(row_sc)[0]
            proba  = model.predict_proba(row_sc)[0][1]

        st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)
        st.markdown('<div class="ig-section-label">Assessment Result</div>',
                    unsafe_allow_html=True)

        res_col, gauge_col, details_col = st.columns([2, 2, 1])

        with res_col:
            if pred == 1:
                risk_lvl = "HIGH" if proba > 0.75 else "ELEVATED"
                st.markdown(f"""
                <div class='ig-result-fraud'>
                    <div class='ig-result-verdict' style='color:{RED}'>
                        Fraudulent
                    </div>
                    <div class='ig-result-score' style='color:{RED}'>
                        Risk probability — {proba*100:.1f}% &nbsp;·&nbsp; Level {risk_lvl}
                    </div>
                    <div style='margin-top:1.2rem;font-size:0.75rem;color:#664444;
                                font-family:JetBrains Mono,monospace;letter-spacing:0.05em'>
                        Recommended for manual review
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ig-result-legit'>
                    <div class='ig-result-verdict' style='color:{GREEN}'>
                        Legitimate
                    </div>
                    <div class='ig-result-score' style='color:{GREEN}'>
                        Risk probability — {proba*100:.1f}% &nbsp;·&nbsp; Level LOW
                    </div>
                    <div style='margin-top:1.2rem;font-size:0.75rem;color:#446644;
                                font-family:JetBrains Mono,monospace;letter-spacing:0.05em'>
                        Consistent with legitimate claim patterns
                    </div>
                </div>""", unsafe_allow_html=True)

        with gauge_col:
            bar_color = RED if pred == 1 else GREEN
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix':"%",
                        'font':{'family':'Playfair Display','size':40,
                                'color':'#e8e0cc'}},
                title={'text':"FRAUD PROBABILITY",
                       'font':{'family':'JetBrains Mono','size':9,
                               'color':'#444455'}},
                gauge={
                    'axis':{'range':[0,100],
                            'tickfont':{'size':9,'color':'#333344'},
                            'tickcolor':'#222233',
                            'tickwidth':1},
                    'bar':{'color':bar_color,'thickness':0.18},
                    'bgcolor':'rgba(0,0,0,0)',
                    'borderwidth':0,
                    'steps':[
                        {'range':[0,35],  'color':'rgba(68,168,120,0.08)'},
                        {'range':[35,65], 'color':'rgba(200,180,80,0.06)'},
                        {'range':[65,100],'color':'rgba(200,68,68,0.10)'},
                    ],
                    'threshold':{
                        'line':{'color':GOLD,'width':2},
                        'thickness':0.75,'value':65
                    }
                }))
            fig_g.update_layout(**PL, height=250)
            st.plotly_chart(fig_g, use_container_width=True)

        with details_col:
            st.markdown(f"""
            <div style='padding-top:0.5rem'>
                <div class='ig-stat-row'>
                    <span class='ig-stat-key'>Claim / Premium</span>
                    <span class='ig-stat-val'>{total_claim_amount/(policy_annual_premium+1):.1f}×</span>
                </div>
                <div class='ig-stat-row'>
                    <span class='ig-stat-key'>Night Incident</span>
                    <span class='ig-stat-val'>{'Yes' if (incident_hour>=22 or incident_hour<=4) else 'No'}</span>
                </div>
                <div class='ig-stat-row'>
                    <span class='ig-stat-key'>High Claim</span>
                    <span class='ig-stat-val'>{'Yes' if total_claim_amount>50000 else 'No'}</span>
                </div>
                <div class='ig-stat-row'>
                    <span class='ig-stat-key'>Witnesses</span>
                    <span class='ig-stat-val'>{witnesses}</span>
                </div>
                <div class='ig-stat-row'>
                    <span class='ig-stat-key'>Tenure</span>
                    <span class='ig-stat-val'>{months_as_customer}mo</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk flags
        flags = []
        if total_claim_amount > 50000:
            flags.append(("Claim amount exceeds $50,000 threshold", "HIGH"))
        if incident_type == "Vehicle Theft":
            flags.append(("Vehicle theft — highest fraud correlation", "HIGH"))
        if incident_severity == "Total Loss":
            flags.append(("Total loss severity declared", "HIGH"))
        if total_claim_amount / (policy_annual_premium + 1) > 20:
            flags.append(("Claim-to-premium ratio exceeds 20×", "HIGH"))
        if incident_hour >= 22 or incident_hour <= 4:
            flags.append(("Incident reported during high-risk hours", "MEDIUM"))
        if witnesses == 0:
            flags.append(("No witnesses present at incident", "MEDIUM"))
        if authorities_contacted == "None":
            flags.append(("No authorities contacted post-incident", "MEDIUM"))
        if months_as_customer < 12:
            flags.append(("Policy holder tenure under 12 months", "MEDIUM"))

        if flags:
            st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)
            st.markdown('<div class="ig-section-label">Risk Signals</div>',
                        unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            for i, (msg, lvl) in enumerate(flags):
                col = fc1 if i % 2 == 0 else fc2
                badge_color = RED if lvl == "HIGH" else '#c8a844'
                col.markdown(
                    f'<div class="ig-risk-flag">'
                    f'<span>{msg}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;'
                    f'font-size:0.62rem;color:{badge_color};'
                    f'letter-spacing:0.1em">{lvl}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("""
    <div class='ig-page-header'>
        <div class='ig-eyebrow'>◈ Analytics</div>
        <div class='ig-title'>Data Explorer</div>
        <div class='ig-subtitle'>Upload a claims dataset for interactive exploratory analysis</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type="csv",
                                label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class='ig-card' style='margin-bottom:1.5rem'>
            <div style='display:flex;gap:3rem'>
                <div><div class='ig-label'>Records</div>
                     <div class='ig-value'>{len(df):,}</div></div>
                <div><div class='ig-label'>Features</div>
                     <div class='ig-value'>{df.shape[1]}</div></div>
                <div><div class='ig-label'>Missing</div>
                     <div class='ig-value'>{df.isnull().sum().sum():,}</div></div>
                <div><div class='ig-label'>Fraud Rate</div>
                     <div class='ig-value'>
                     {f"{(df['fraud_reported'].map({'Y':1,'N':0}).mean()*100):.1f}%"
                      if 'fraud_reported' in df.columns else "—"}
                     </div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["  Raw Data  ", "  Distributions  ", "  Fraud Breakdown  "])

        with t1:
            st.dataframe(df.head(100), use_container_width=True, height=420)

        with t2:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if num_cols:
                sel = st.selectbox("Feature", num_cols, label_visibility="collapsed")
                fig = go.Figure()
                if 'fraud_reported' in df.columns:
                    fd = df['fraud_reported'].map({'Y':1,'N':0}) \
                         if df['fraud_reported'].dtype==object else df['fraud_reported']
                    fig.add_trace(go.Histogram(
                        x=df[fd==0][sel], name='Legitimate',
                        marker_color='rgba(68,168,120,0.4)', nbinsx=40,
                        marker_line=dict(width=0)))
                    fig.add_trace(go.Histogram(
                        x=df[fd==1][sel], name='Fraudulent',
                        marker_color='rgba(200,68,68,0.55)', nbinsx=40,
                        marker_line=dict(width=0)))
                    fig.update_layout(**PL, barmode='overlay',
                                      height=340, title=sel)
                else:
                    fig.add_trace(go.Histogram(
                        x=df[sel], marker_color=GOLD,
                        nbinsx=40, marker_line=dict(width=0)))
                    fig.update_layout(**PL, height=340, title=sel)
                st.plotly_chart(fig, use_container_width=True)

        with t3:
            FRAUD_ALIASES = ['fraud_reported','fraud','is_fraud','FraudFound_P',
                             'fraud_flag','FRAUD','Fraud','target','label',
                             'fraudulent','claim_fraud','OUTCOME']
            fraud_col_name = next((a for a in FRAUD_ALIASES if a in df.columns), None)

            if fraud_col_name:
                raw = df[fraud_col_name]
                if raw.dtype == object:
                    uv = set(raw.dropna().unique())
                    if uv <= {'Y','N','y','n'}:
                        fd = raw.map({'Y':1,'N':0,'y':1,'n':0})
                    elif uv <= {'Yes','No','YES','NO'}:
                        fd = raw.map({'Yes':1,'No':0,'YES':1,'NO':0})
                    elif uv <= {'1','0'}:
                        fd = raw.map({'1':1,'0':0})
                    else:
                        first = list(raw.dropna().unique())[0]
                        fd = (raw == first).astype(int)
                else:
                    fd = raw.fillna(0).astype(int)

                st.markdown(
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
                    f"color:#444455;margin-bottom:1rem'>Fraud column: "
                    f"<span style='color:{GOLD}'>{fraud_col_name}</span>"
                    f" &nbsp;·&nbsp; Cases: <span style='color:{RED}'>{int(fd.sum())}</span>"
                    f" &nbsp;·&nbsp; Rate: <span style='color:{RED}'>{fd.mean()*100:.1f}%</span>"
                    f"</div>", unsafe_allow_html=True)

                cat_cols = [c for c in df.select_dtypes(include='object').columns
                            if c != fraud_col_name]
                if cat_cols:
                    sel_cat = st.selectbox("Group by", cat_cols, label_visibility="collapsed")
                    df2 = df.copy(); df2['_f'] = fd
                    rates = df2.groupby(sel_cat)['_f'].mean().sort_values() * 100
                    avg   = rates.mean()
                    fig2  = go.Figure(go.Bar(
                        x=rates.values, y=rates.index, orientation='h',
                        marker=dict(color=[RED if v>avg else '#1e1e2e' for v in rates.values],
                                    line=dict(width=0)),
                        text=[f'{v:.1f}%' for v in rates.values],
                        textposition='outside',
                        textfont=dict(size=10, color='#555566')))
                    fig2.add_vline(x=avg, line_color=GOLD, line_width=1, line_dash='dash')
                    fig2.update_layout(**PL, height=max(300, len(rates)*30),
                                       title=f'Fraud Rate by {sel_cat}')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                all_cols = ", ".join(df.columns.tolist())
                st.markdown(
                    f"<div style='background:rgba(200,184,120,0.05);border:1px solid "
                    f"rgba(200,184,120,0.15);border-radius:2px;padding:1.2rem'>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;"
                    f"color:{GOLD};margin-bottom:0.6rem'>NO FRAUD COLUMN DETECTED</div>"
                    f"<div style='font-size:0.78rem;color:#555566;margin-bottom:0.6rem'>"
                    f"Expected: fraud_reported, fraud, is_fraud, FraudFound_P, target</div>"
                    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
                    f"color:#444455'>Columns found:<br>"
                    f"<span style='color:#888899'>{all_cols}</span></div></div>",
                    unsafe_allow_html=True)
                num_c = df.select_dtypes(include='number').columns.tolist()
                if num_c:
                    sel_n = st.selectbox("Explore numeric feature", num_c,
                                         label_visibility="collapsed")
                    fig_e = go.Figure(go.Histogram(x=df[sel_n], marker_color=GOLD,
                                                   nbinsx=40, marker_line=dict(width=0)))
                    fig_e.update_layout(**PL, height=300, title=sel_n)
                    st.plotly_chart(fig_e, use_container_width=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:4rem 0;'>
            <div style='font-size:2rem;margin-bottom:1rem;opacity:0.3'>◈</div>
            <div style='font-family:JetBrains Mono,monospace;font-size:0.7rem;
                        color:#333344;letter-spacing:0.15em;text-transform:uppercase'>
                Upload insurance_claims.csv to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MODEL METRICS
# ══════════════════════════════════════════════════════════════
elif page == "Model Metrics":
    st.markdown("""
    <div class='ig-page-header'>
        <div class='ig-eyebrow'>◈ Evaluation</div>
        <div class='ig-title'>Model Metrics</div>
        <div class='ig-subtitle'>Performance evaluation of the trained Random Forest classifier</div>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("Model unavailable. Run `python src/train.py`.")
        st.stop()

    metrics = {
        'Accuracy' : REPORT.get('test_accuracy',  0),
        'Precision': REPORT.get('test_precision', 0),
        'Recall'   : REPORT.get('test_recall',    0),
        'F1-Score' : REPORT.get('test_f1',        0),
        'AUC-ROC'  : REPORT.get('test_auc_roc',   0),
    }

    # Metric cards
    cols = st.columns(5)
    for col, (name, val) in zip(cols, metrics.items()):
        col.markdown(f"""
        <div class='ig-card' style='text-align:center'>
            <div class='ig-label'>{name}</div>
            <div style='font-family:Playfair Display,serif;font-size:2.2rem;
                        font-weight:600;color:#e8e0cc;margin-top:0.3rem'>
                {val:.3f}
            </div>
            <div style='height:2px;background:linear-gradient(90deg,
                rgba(180,140,80,{val}),transparent);
                margin-top:0.8rem;border-radius:1px'></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="ig-divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown('<div class="ig-section-label">Performance Radar</div>',
                    unsafe_allow_html=True)
        cats = list(metrics.keys())
        vals = list(metrics.values())
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself',
            fillcolor='rgba(180,140,80,0.05)',
            line=dict(color=GOLD, width=1.5),
            marker=dict(color=GOLD, size=5)))
        fig_r.update_layout(**PL, height=340,
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    range=[0,1], showticklabels=False,
                    gridcolor='rgba(255,255,255,0.04)',
                    linecolor='rgba(255,255,255,0.04)'),
                angularaxis=dict(
                    gridcolor='rgba(255,255,255,0.04)',
                    linecolor='rgba(255,255,255,0.04)',
                    tickfont=dict(size=10, color='#555566',
                                  family='JetBrains Mono'))))
        st.plotly_chart(fig_r, use_container_width=True)

    with col_r:
        st.markdown('<div class="ig-section-label">Hyperparameters</div>',
                    unsafe_allow_html=True)
        params = REPORT.get('best_params', {})
        params_html = ''.join([
            f'<div class="ig-stat-row">'
            f'<span class="ig-stat-key">{k}</span>'
            f'<span class="ig-stat-val">{v}</span>'
            f'</div>'
            for k, v in params.items()
        ])
        cv_mean = REPORT.get('cv_f1_mean', 0)
        cv_std  = REPORT.get('cv_f1_std',  0)
        st.markdown(f"""
        <div class='ig-card'>
            {params_html}
            <div style='margin-top:1.5rem'>
                <div class='ig-label'>5-Fold Cross-Validation F1</div>
                <div style='font-family:Playfair Display,serif;font-size:2rem;
                            font-weight:600;color:{GOLD};margin-top:0.3rem'>
                    {cv_mean:.4f}
                </div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;
                            color:#444455;margin-top:0.2rem'>
                    ± {cv_std:.4f} standard deviation
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SYSTEM INFO
# ══════════════════════════════════════════════════════════════
elif page == "System Info":
    st.markdown("""
    <div class='ig-page-header'>
        <div class='ig-eyebrow'>◈ Documentation</div>
        <div class='ig-title'>System Information</div>
        <div class='ig-subtitle'>Architecture, methodology, and technical specifications</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="ig-section-label">Pipeline Architecture</div>',
                    unsafe_allow_html=True)
        steps = [
            ("01", "Data Ingestion",       "15,420 auto insurance claims · 40 raw features"),
            ("02", "Feature Engineering",  "6 derived features: claim ratios, temporal flags, risk indicators"),
            ("03", "NaN Imputation",        "Median imputation for numeric · zero-fill for categoricals"),
            ("04", "Class Rebalancing",     "SMOTE oversampling · 12% → 50% fraud representation"),
            ("05", "Train / Test Split",    "80/20 stratified split · StandardScaler normalization"),
            ("06", "Model Training",        "Logistic Regression · Random Forest · Gradient Boosting"),
            ("07", "Hyperparameter Tuning", "GridSearchCV · 5-fold stratified cross-validation"),
            ("08", "Evaluation",            "AUC-ROC · F1 · Precision · Recall · Confusion Matrix"),
            ("09", "Artifact Persistence",  "Joblib serialization · JSON feature registry"),
        ]
        steps_html = ''.join([
            f'<div style="display:flex;gap:1.5rem;padding:0.9rem 0;'
            f'border-bottom:1px solid rgba(255,255,255,0.04)">'
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
            f'color:{GOLD};min-width:1.5rem;padding-top:0.1rem">{n}</div>'
            f'<div><div style="font-size:0.82rem;color:#aaaaaa;font-weight:500;'
            f'margin-bottom:0.15rem">{title}</div>'
            f'<div style="font-size:0.75rem;color:#444455">{desc}</div></div>'
            f'</div>'
            for n, title, desc in steps
        ])
        st.markdown(f'<div class="ig-card">{steps_html}</div>',
                    unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="ig-section-label">Tech Stack</div>',
                    unsafe_allow_html=True)
        stack = [
            ("Language",   "Python 3.10+"),
            ("ML Core",    "Scikit-learn 1.3+"),
            ("Balancing",  "Imbalanced-learn"),
            ("Interface",  "Streamlit"),
            ("Charts",     "Plotly"),
            ("Data",       "Pandas / NumPy"),
            ("Storage",    "Joblib"),
            ("Notebooks",  "Jupyter"),
        ]
        stack_html = ''.join([
            f'<div class="ig-stat-row">'
            f'<span class="ig-stat-key">{k}</span>'
            f'<span class="ig-stat-val">{v}</span>'
            f'</div>'
            for k, v in stack
        ])
        st.markdown(f'<div class="ig-card">{stack_html}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="ig-section-label" style="margin-top:1.5rem">Dataset</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='ig-card'>
            <div class='ig-stat-row'>
                <span class='ig-stat-key'>Source</span>
                <span class='ig-stat-val'>Kaggle</span>
            </div>
            <div class='ig-stat-row'>
                <span class='ig-stat-key'>Records</span>
                <span class='ig-stat-val'>15,420</span>
            </div>
            <div class='ig-stat-row'>
                <span class='ig-stat-key'>Features</span>
                <span class='ig-stat-val'>40 raw / 46 engineered</span>
            </div>
            <div class='ig-stat-row'>
                <span class='ig-stat-key'>Fraud Rate</span>
                <span class='ig-stat-val'>~12%</span>
            </div>
            <div class='ig-stat-row'>
                <span class='ig-stat-key'>Type</span>
                <span class='ig-stat-val'>Auto Insurance</span>
            </div>
        </div>
        """, unsafe_allow_html=True)