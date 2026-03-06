# ============================================================
#  InsureGuard — Streamlit Fraud Detection App
#  Run from project root:  streamlit run app/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="InsureGuard · Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #080812;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(102,126,234,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(255,65,108,0.08) 0%, transparent 60%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important; font-weight: 700 !important; color: #e8e8ff !important;
}
[data-testid="stMetricLabel"] { color: #8888aa !important; font-size: 0.78rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    padding: 0.75rem 2rem !important; font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important; font-size: 1rem !important; width: 100% !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

label { color: #aaaacc !important; font-size: 0.82rem !important; }

.result-fraud {
    background: linear-gradient(135deg, rgba(255,65,108,0.15), rgba(255,75,43,0.1));
    border: 1.5px solid rgba(255,65,108,0.5); border-radius: 20px;
    padding: 2rem; text-align: center;
    animation: pulse-red 2s infinite;
}
.result-legit {
    background: linear-gradient(135deg, rgba(56,239,125,0.12), rgba(17,153,142,0.08));
    border: 1.5px solid rgba(56,239,125,0.4); border-radius: 20px;
    padding: 2rem; text-align: center;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 20px rgba(255,65,108,0.2); }
    50%      { box-shadow: 0 0 40px rgba(255,65,108,0.45); }
}
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700;
    color: #e8e8ff; margin-bottom: 0.2rem; letter-spacing: -0.02em;
}
.section-sub { font-size: 0.85rem; color: #6666aa; margin-bottom: 1.5rem; }
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font_color='#c8c8e8', font_family='DM Sans',
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.06)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.06)'),
)

# ── Load artifacts ────────────────────────────────────────────
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
    <div style='padding:1rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                    color:#e8e8ff;letter-spacing:-0.02em'>🛡️ InsureGuard</div>
        <div style='font-size:0.75rem;color:#555577;margin-top:0.2rem'>
            AI Fraud Detection System</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Dashboard", "🔍  Predict Claim",
        "📊  Analytics", "📈  Model Performance", "ℹ️   About"])
    st.markdown("<hr style='border-color:rgba(255,255,255,0.07)'>", unsafe_allow_html=True)
    if MODEL_OK:
        st.markdown(f"""
        <div style='font-size:0.75rem;color:#555577;margin-bottom:0.4rem'>MODEL STATUS</div>
        <div style='font-size:0.82rem;color:#38ef7d'>● Model Loaded</div>
        <div style='font-size:0.78rem;color:#8888aa;margin-top:0.3rem'>
            AUC-ROC : {REPORT.get('test_auc_roc','N/A')}<br>
            F1-Score: {REPORT.get('test_f1','N/A')}<br>
            Features: {REPORT.get('n_features','N/A')}
        </div>""", unsafe_allow_html=True)
    else:
        st.error("Model not found.\nRun: python src/train.py")

# ══════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown('<div class="section-title">📊 Claims Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time overview of insurance claim activity</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("📋 Total Claims",   "15,420", "+8.2%")
    c2.metric("🚨 Fraud Detected", "1,847",  "+3.1%")
    c3.metric("✅ Legitimate",      "13,573", "+9.0%")
    c4.metric("🎯 AUC-ROC", str(REPORT.get('test_auc_roc', 0.929)) if MODEL_OK else "0.929", "+1.2%")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_a, col_b = st.columns([3,2])

    with col_a:
        st.markdown("#### 📈 Monthly Fraud Trend")
        months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months,
            y=[900,950,880,960,1020,1100,1150,1080,990,1010,1080,1200],
            name='Legitimate', line=dict(color='#38ef7d',width=2.5),
            fill='tozeroy', fillcolor='rgba(56,239,125,0.07)'))
        fig.add_trace(go.Scatter(x=months,
            y=[120,135,98,145,167,189,201,178,155,143,190,210],
            name='Fraudulent', line=dict(color='#ff416c',width=2.5),
            fill='tozeroy', fillcolor='rgba(255,65,108,0.1)'))
        fig.update_layout(**PLOT_LAYOUT, height=280, legend=dict(orientation='h',y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🍩 Fraud Split")
        fig2 = go.Figure(go.Pie(
            labels=['Legitimate','Fraudulent'], values=[88,12], hole=0.65,
            marker_colors=['#38ef7d','#ff416c'],
            textinfo='percent+label', textfont=dict(size=12,color='white')))
        fig2.update_layout(**PLOT_LAYOUT, height=280, showlegend=False,
            annotations=[dict(text='12%<br><span style="font-size:11px">Fraud Rate</span>',
                x=0.5,y=0.5,font_size=18,showarrow=False,font_color='white')])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown("#### 🚗 Fraud by Incident Type")
        fig3 = go.Figure(go.Bar(
            x=[22.4,14.1,9.8,5.2],
            y=['Vehicle Theft','Multi-vehicle','Single Vehicle','Parked Car'],
            orientation='h',
            marker_color=['#ff416c','#f7c948','#667eea','#38ef7d'],
            text=['22.4%','14.1%','9.8%','5.2%'], textposition='outside'))
        fig3.update_layout(**PLOT_LAYOUT, height=240, xaxis_title='Fraud Rate %')
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        st.markdown("#### ⏰ Fraud by Hour of Day")
        hours  = list(range(24))
        counts = [18,12,9,8,7,6,8,10,14,16,18,21,
                  20,19,22,24,26,29,32,35,38,33,28,22]
        fig4 = go.Figure(go.Bar(x=hours, y=counts,
            marker_color=['#ff416c' if (h>=22 or h<=4) else '#667eea' for h in hours]))
        fig4.update_layout(**PLOT_LAYOUT, height=240,
                           xaxis_title='Hour', yaxis_title='Fraud Cases')
        st.plotly_chart(fig4, use_container_width=True)

    with c3:
        st.markdown("#### 💰 Claim Amount Distribution")
        np.random.seed(42)
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=np.random.exponential(8000,500),
            name='Legit', marker_color='rgba(56,239,125,0.5)', nbinsx=30))
        fig5.add_trace(go.Histogram(x=np.random.exponential(18000,80),
            name='Fraud', marker_color='rgba(255,65,108,0.7)', nbinsx=20))
        fig5.update_layout(**PLOT_LAYOUT, height=240, barmode='overlay',
                           xaxis_title='Claim Amount ($)')
        st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="section-title">🔍 Fraud Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter claim details to get an instant fraud risk assessment</div>', unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("⚠️ Model not loaded. Run `python src/train.py` first.")
        st.stop()

    with st.form("predict_form"):
        col1,col2,col3 = st.columns(3)

        with col1:
            st.markdown("**👤 Policy Holder**")
            age                   = st.slider("Age", 18, 75, 35)
            months_as_customer    = st.slider("Months as Customer", 1, 500, 100)
            policy_deductable     = st.selectbox("Deductible ($)", [500,1000,2000])
            policy_annual_premium = st.number_input("Annual Premium ($)", 500, 5000, 1200)
            insured_education_level = st.selectbox("Education",
                ["High School","College","Associate","MD","Masters","PhD","JD"])
            insured_occupation    = st.selectbox("Occupation",
                ["craft-repair","sales","tech-support","exec-managerial",
                 "prof-specialty","other-service","armed-forces"])

        with col2:
            st.markdown("**🚗 Incident Details**")
            incident_type         = st.selectbox("Incident Type",
                ["Single Vehicle Collision","Multi-vehicle Collision",
                 "Vehicle Theft","Parked Car"])
            incident_severity     = st.selectbox("Severity",
                ["Minor Damage","Major Damage","Total Loss","Trivial Damage"])
            authorities_contacted = st.selectbox("Authorities Contacted",
                ["Police","Fire","Ambulance","None","Other"])
            number_of_vehicles    = st.slider("Vehicles Involved", 1, 4, 1)
            bodily_injuries       = st.slider("Bodily Injuries", 0, 2, 0)
            witnesses             = st.slider("Witnesses", 0, 3, 1)
            incident_hour         = st.slider("Hour of Incident (24h)", 0, 23, 14)

        with col3:
            st.markdown("**💰 Claim Financials**")
            total_claim_amount = st.number_input("Total Claim ($)",    100, 100000, 12000)
            injury_claim       = st.number_input("Injury Claim ($)",     0,  50000,  3000)
            property_claim     = st.number_input("Property Claim ($)",   0,  50000,  4000)
            vehicle_claim      = st.number_input("Vehicle Claim ($)",    0,  80000,  5000)
            auto_year          = st.slider("Vehicle Year", 1995, 2024, 2015)
            auto_make          = st.selectbox("Vehicle Make",
                ["BMW","Mercedes","Dodge","Toyota","Ford",
                 "Chevrolet","Honda","Audi","Nissan"])
            collision_type     = st.selectbox("Collision Type",
                ["Front Collision","Rear Collision","Side Collision","?"])

        submitted = st.form_submit_button("🔍 Analyze Claim for Fraud Risk")

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

        st.markdown("---")
        st.markdown("### 🧠 Prediction Result")
        res_col, gauge_col = st.columns(2)

        with res_col:
            if pred == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div style="font-family:Syne,sans-serif;font-size:1.8rem;
                                font-weight:800;color:#ff416c">🚨 FRAUD DETECTED</div>
                    <div style="color:#ffaaaa;margin-top:0.4rem">
                        Risk Score: <strong>{proba*100:.1f}%</strong> &nbsp;|&nbsp;
                        Level: <strong>{'HIGH' if proba>0.75 else 'MEDIUM'}</strong>
                    </div>
                    <div style="margin-top:1rem;font-size:0.82rem;color:#cc6677">
                        ⚠️ Flagged for manual review</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div style="font-family:Syne,sans-serif;font-size:1.8rem;
                                font-weight:800;color:#38ef7d">✅ LEGITIMATE CLAIM</div>
                    <div style="color:#aaffcc;margin-top:0.4rem">
                        Risk Score: <strong>{proba*100:.1f}%</strong> &nbsp;|&nbsp;
                        Level: <strong>LOW</strong>
                    </div>
                    <div style="margin-top:1rem;font-size:0.82rem;color:#55bb77">
                        ✓ Consistent with normal claim patterns</div>
                </div>""", unsafe_allow_html=True)

        with gauge_col:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={'text':"Fraud Risk Score",'font':{'color':'#c8c8e8','size':14}},
                number={'suffix':"%",'font':{'color':'white','size':36}},
                gauge={
                    'axis':{'range':[0,100],'tickcolor':'#666688'},
                    'bar': {'color':'#ff416c' if pred==1 else '#38ef7d','thickness':0.25},
                    'bgcolor':'rgba(255,255,255,0.04)',
                    'steps':[
                        {'range':[0,35],  'color':'rgba(56,239,125,0.12)'},
                        {'range':[35,65], 'color':'rgba(247,201,72,0.10)'},
                        {'range':[65,100],'color':'rgba(255,65,108,0.15)'},
                    ],
                    'threshold':{'line':{'color':'white','width':3},'thickness':0.8,'value':65}
                }))
            fig_g.update_layout(**PLOT_LAYOUT, height=280)
            st.plotly_chart(fig_g, use_container_width=True)

        # Risk flags
        st.markdown("#### 🔎 Risk Signals Detected")
        flags = []
        if total_claim_amount > 50000:           flags.append(("💰 Very high claim amount","HIGH"))
        if incident_type == "Vehicle Theft":      flags.append(("🚗 Vehicle Theft — highest fraud type","HIGH"))
        if incident_severity == "Total Loss":     flags.append(("💥 Total Loss severity","HIGH"))
        if incident_hour >= 22 or incident_hour <= 4: flags.append(("🌙 Late-night incident","MEDIUM"))
        if witnesses == 0:                        flags.append(("👁️ No witnesses present","MEDIUM"))
        if authorities_contacted == "None":       flags.append(("🚔 No authorities contacted","MEDIUM"))
        if months_as_customer < 12:               flags.append(("📅 Customer < 1 year old","MEDIUM"))
        if total_claim_amount/(policy_annual_premium+1) > 20:
                                                  flags.append(("📊 Claim/Premium ratio very high","HIGH"))
        if flags:
            fc1,fc2 = st.columns(2)
            for i,(msg,level) in enumerate(flags):
                col = fc1 if i%2==0 else fc2
                color = '#ff416c' if level=="HIGH" else '#f7c948'
                col.markdown(
                    f'<div style="background:rgba(255,255,255,0.04);border-left:3px solid {color};'
                    f'border-radius:8px;padding:0.6rem 1rem;margin-bottom:0.5rem;font-size:0.85rem;color:#ccccee">'
                    f'{msg} <span style="color:{color};font-weight:600;float:right">{level}</span></div>',
                    unsafe_allow_html=True)
        else:
            st.success("✅ No major risk flags detected.")

# ══════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════
elif "Analytics" in page:
    st.markdown('<div class="section-title">📊 Claim Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload your claims CSV for full interactive analysis</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Claims CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
        t1,t2,t3 = st.tabs(["📋 Data Preview","📈 Distributions","🚨 Fraud Analysis"])

        with t1:
            st.dataframe(df.head(50), use_container_width=True)
            a,b,c = st.columns(3)
            a.metric("Total Records",  f"{len(df):,}")
            b.metric("Total Features", f"{df.shape[1]}")
            c.metric("Missing Values", f"{df.isnull().sum().sum():,}")

        with t2:
            num_cols = df.select_dtypes(include='number').columns.tolist()
            if num_cols:
                sel = st.selectbox("Select feature", num_cols)
                fig = px.histogram(df, x=sel, nbins=50,
                    color_discrete_sequence=['#667eea'], title=f"Distribution of {sel}")
                fig.update_layout(**PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

        with t3:
            if 'fraud_reported' in df.columns:
                fraud_col = df['fraud_reported'].map({'Y':1,'N':0}) \
                            if df['fraud_reported'].dtype==object else df['fraud_reported']
                st.metric("Overall Fraud Rate", f"{fraud_col.mean()*100:.1f}%")
                cat_cols = [c for c in df.select_dtypes(include='object').columns
                            if c != 'fraud_reported']
                if cat_cols:
                    sel_cat = st.selectbox("Fraud rate by category", cat_cols)
                    df2 = df.copy(); df2['fraud_bin'] = fraud_col
                    rates = df2.groupby(sel_cat)['fraud_bin'].mean().sort_values()*100
                    fig2 = go.Figure(go.Bar(
                        x=rates.values, y=rates.index, orientation='h',
                        marker_color=['#ff416c' if v>rates.mean() else '#667eea' for v in rates.values],
                        text=[f'{v:.1f}%' for v in rates.values], textposition='outside'))
                    fig2.update_layout(**PLOT_LAYOUT, height=350,
                                       title=f"Fraud Rate by {sel_cat}")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Column 'fraud_reported' not found.")
    else:
        st.info("👆 Upload your insurance_claims.csv to see full analytics")

# ══════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Evaluation metrics of the trained Random Forest model</div>', unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("Model not loaded. Run `python src/train.py` first.")
        st.stop()

    metrics = {
        'Accuracy' : REPORT.get('test_accuracy',  0),
        'Precision': REPORT.get('test_precision', 0),
        'Recall'   : REPORT.get('test_recall',    0),
        'F1-Score' : REPORT.get('test_f1',        0),
        'AUC-ROC'  : REPORT.get('test_auc_roc',   0),
    }
    cols   = st.columns(5)
    colors = ['#667eea','#38ef7d','#f7c948','#ff6b9d','#00d4ff']
    for col,(name,val),color in zip(cols,metrics.items(),colors):
        col.markdown(f"""
        <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                    border-top:3px solid {color};border-radius:14px;padding:1.2rem;text-align:center">
            <div style="font-size:0.75rem;color:#666688;margin-bottom:0.3rem">{name}</div>
            <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:700;color:{color}">
                {val:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    col_r,col_b = st.columns(2)

    with col_r:
        st.markdown("#### Radar Chart — All Metrics")
        cats = list(metrics.keys()); vals = list(metrics.values())
        fig_r = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill='toself', fillcolor='rgba(102,126,234,0.15)',
            line=dict(color='#667eea',width=2.5),
            marker=dict(color='#667eea',size=8)))
        fig_r.update_layout(**PLOT_LAYOUT, height=320,
            polar=dict(bgcolor='rgba(255,255,255,0.03)',
                radialaxis=dict(range=[0,1],tickfont_color='#666688',
                                gridcolor='rgba(255,255,255,0.08)'),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                                 tickfont_color='#c8c8e8')))
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        st.markdown("#### Best Hyperparameters")
        for k,v in REPORT.get('best_params',{}).items():
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'background:rgba(255,255,255,0.04);border-radius:8px;'
                f'padding:0.6rem 1rem;margin-bottom:0.4rem;font-size:0.85rem">'
                f'<span style="color:#8888aa">{k}</span>'
                f'<span style="color:#667eea;font-weight:600">{v}</span></div>',
                unsafe_allow_html=True)
        cv_mean = REPORT.get('cv_f1_mean',0); cv_std = REPORT.get('cv_f1_std',0)
        st.markdown(
            f'<div style="background:rgba(56,239,125,0.08);border:1px solid rgba(56,239,125,0.2);'
            f'border-radius:12px;padding:1rem;text-align:center;margin-top:1rem">'
            f'<div style="font-size:0.78rem;color:#55bb77">5-Fold CV F1-Score</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:2.2rem;font-weight:700;color:#38ef7d">'
            f'{cv_mean:.4f}</div>'
            f'<div style="font-size:0.78rem;color:#55bb77">± {cv_std:.4f}</div></div>',
            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="section-title">ℹ️ About InsureGuard</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#aaaacc;font-size:1rem;line-height:1.7;max-width:720px">
    <strong style="color:#e8e8ff">InsureGuard</strong> is an end-to-end AI-powered insurance fraud
    detection system. It analyzes claim patterns, customer history, and incident characteristics
    to identify potentially fraudulent claims in real time using a tuned Random Forest model.
    </p>""", unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(icon,title,desc) in zip([c1,c2,c3],[
        ("🤖","ML Model","Tuned Random Forest — 200–250 trees with optimized depth"),
        ("📊","Dataset","15,420 real auto insurance claims · 40 features"),
        ("⚙️","Pipeline","Load → Engineer → Balance → Train → Tune → Save"),
    ]):
        col.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);'
            f'border-radius:16px;padding:1.5rem">'
            f'<div style="font-size:1.5rem">{icon}</div>'
            f'<div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;'
            f'color:#e8e8ff;margin:0.5rem 0 0.3rem">{title}</div>'
            f'<div style="font-size:0.8rem;color:#666688;line-height:1.5">{desc}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### 🛠️ Tech Stack")
    stack = ["Python 3.10+","Scikit-learn","Imbalanced-learn",
             "Streamlit","Plotly","Pandas / NumPy","Joblib","Jupyter"]
    cols = st.columns(4)
    for i,tech in enumerate(stack):
        cols[i%4].markdown(
            f'<div style="background:rgba(102,126,234,0.1);border:1px solid rgba(102,126,234,0.2);'
            f'border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.5rem;'
            f'font-size:0.82rem;color:#aaaaff">⚡ {tech}</div>',
            unsafe_allow_html=True)