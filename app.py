import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="VitalAI — Health Prediction System",
    page_icon="🫀",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #F0F4F8;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A2540 0%, #103561 60%, #0D4A6B 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

section[data-testid="stSidebar"] .stSuccess {
    background: rgba(16,185,129,0.15) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    color: #6EE7B7 !important;
}

section[data-testid="stSidebar"] .stInfo {
    background: rgba(59,130,246,0.12) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 12px !important;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0 18px 0;
}

.sidebar-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.3px;
}

.sidebar-logo-dot {
    color: #38BDF8 !important;
}

.sidebar-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56,189,248,0.18);
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #7DD3FC !important;
    margin-bottom: 16px;
}

.sidebar-section {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #64748B !important;
    margin: 18px 0 8px 0;
}

.sidebar-biomarker {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 9px 12px;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #94A3B8 !important;
    border: 1px solid rgba(255,255,255,0.07);
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0A2540 0%, #0E3A6E 45%, #0B5394 100%);
    border-radius: 24px;
    padding: 52px 60px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(10,37,64,0.22);
}

.hero-wrap::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(56,189,248,0.18) 0%, transparent 70%);
    pointer-events: none;
}

.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #7DD3FC;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.15;
    letter-spacing: -0.8px;
    margin-bottom: 14px;
}

.hero-title span {
    color: #38BDF8;
}

.hero-sub {
    font-size: 1.05rem;
    color: #94A3B8;
    max-width: 560px;
    line-height: 1.65;
    margin-bottom: 28px;
}

.hero-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.hero-pill {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    font-weight: 500;
    color: #CBD5E1;
}

/* ── Stat Cards ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}

.stat-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 2px 12px rgba(10,37,64,0.07);
    border: 1px solid #E2E8F0;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: box-shadow 0.22s, transform 0.22s;
}

.stat-card:hover {
    box-shadow: 0 8px 28px rgba(10,37,64,0.12);
    transform: translateY(-2px);
}

.stat-icon {
    width: 46px; height: 46px;
    border-radius: 13px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}

.stat-icon.blue  { background: #EFF6FF; }
.stat-icon.teal  { background: #F0FDFA; }
.stat-icon.green { background: #F0FDF4; }
.stat-icon.amber { background: #FFFBEB; }

.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1;
    margin-bottom: 3px;
}

.stat-label {
    font-size: 0.73rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Section Cards ── */
.section-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 28px 30px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(10,37,64,0.06);
    border: 1px solid #E2E8F0;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
    padding-bottom: 14px;
    border-bottom: 1px solid #F1F5F9;
}

.section-icon {
    width: 38px; height: 38px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}

.section-icon.blue  { background: #EFF6FF; }
.section-icon.rose  { background: #FFF1F2; }
.section-icon.emerald { background: #ECFDF5; }
.section-icon.purple  { background: #F5F3FF; }
.section-icon.orange  { background: #FFF7ED; }

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
}

.section-sub {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 1px;
}

/* ── Inputs ── */
label {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

div[data-baseweb="select"] {
    border-radius: 12px !important;
}

input[type="number"] {
    border-radius: 12px !important;
    border-color: #E2E8F0 !important;
    background: #F8FAFC !important;
}

div[data-baseweb="input"] {
    border-radius: 12px !important;
    background: #F8FAFC !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #0EA5E9) !important;
}

/* ── CTA Button ── */
.stButton > button {
    width: 100%;
    height: 58px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(135deg, #0A2540 0%, #0E3A6E 50%, #1B5EA0 100%);
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.2px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 18px rgba(10,37,64,0.25);
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(10,37,64,0.35);
    color: white !important;
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── Result Cards ── */
.result-healthy {
    background: linear-gradient(135deg, #065F46 0%, #047857 50%, #059669 100%);
    border-radius: 20px;
    padding: 32px 36px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 12px 40px rgba(5,150,105,0.3);
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.result-unhealthy {
    background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 50%, #DC2626 100%);
    border-radius: 20px;
    padding: 32px 36px;
    color: white;
    margin-top: 20px;
    box-shadow: 0 12px 40px rgba(220,38,38,0.3);
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.result-healthy::before, .result-unhealthy::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
    pointer-events: none;
}

.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.result-headline {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 6px;
}

.result-confidence {
    font-size: 0.95rem;
    opacity: 0.85;
}

.result-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.15);
    margin: 18px 0;
}

.result-meta {
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
}

.result-meta-item label {
    font-size: 0.72rem !important;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 600 !important;
}

.result-meta-item .val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
}

/* ── Prediction Metric Cards ── */
.pred-metric-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(10,37,64,0.07);
    border: 1px solid #E2E8F0;
    transition: all 0.22s;
}

.pred-metric-card:hover {
    box-shadow: 0 8px 24px rgba(10,37,64,0.12);
    transform: translateY(-2px);
}

.pred-metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #0A2540;
    line-height: 1;
    margin-bottom: 6px;
}

.pred-metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ── Wellness Tips ── */
.tip-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    background: #F8FAFC;
    border-radius: 14px;
    padding: 14px 16px;
    border: 1px solid #E2E8F0;
    transition: background 0.18s;
}

.tip-item:hover {
    background: #F1F5F9;
}

.tip-emoji {
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 1px;
}

.tip-text strong {
    display: block;
    font-size: 0.85rem;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 3px;
}

.tip-text span {
    font-size: 0.78rem;
    color: #64748B;
    line-height: 1.5;
}

/* ── Footer ── */
.footer {
    background: #0A2540;
    border-radius: 20px;
    padding: 28px 36px;
    margin-top: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}

.footer-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #FFFFFF;
}

.footer-brand span { color: #38BDF8; }

.footer-meta {
    font-size: 0.78rem;
    color: #64748B;
    line-height: 1.7;
}

.footer-tech {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.footer-tag {
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 4px 11px;
    font-size: 0.72rem;
    font-weight: 500;
    color: #7DD3FC;
}

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

.section-card { animation: fadeIn 0.35s ease; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Caching Resource Architecture
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load("health_model.pkl")

@st.cache_resource
def load_columns():
    return joblib.load("feature_columns.pkl")

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <span style="font-size:1.6rem;">🫀</span>
        <span class="sidebar-logo-text">Vital<span class="sidebar-logo-dot">AI</span></span>
    </div>
    <div class="sidebar-badge">⚡ Live Model</div>
    """, unsafe_allow_html=True)

    st.success("🤖 AdaBoost · Recall **96.79%**")

    st.markdown('<div class="sidebar-section">How to use</div>', unsafe_allow_html=True)
    st.info(
        "Fill in the patient's demographics, vitals, and lifestyle metrics across each section. "
        "Hit **Run Prediction** to get an instant wellness assessment."
    )

    st.markdown('<div class="sidebar-section">Evaluated Biomarkers</div>', unsafe_allow_html=True)

    biomarkers = [
        ("🩸", "Blood Pressure & Vitals"),
        ("💊", "Cholesterol & Glucose"),
        ("🧘", "Lifestyle & Mental Health"),
        ("📋", "Medical Records & Allergies"),
    ]
    for icon, label in biomarkers:
        st.markdown(f"""
        <div class="sidebar-biomarker">{icon}&nbsp;&nbsp;{label}</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Disclaimer</div>', unsafe_allow_html=True)
    st.caption("This tool is for research and educational purposes. It does not replace professional medical advice.")

# ==========================================
# 3. HERO BANNER
# ==========================================
st.markdown("""
<div class="hero-wrap">
    <div class="hero-eyebrow">🏥 Healthcare AI &nbsp;·&nbsp; AdaBoost Classifier</div>
    <div class="hero-title">
        Predict Health Risks<br>with <span>Clinical Precision</span>
    </div>
    <div class="hero-sub">
        Enter patient biomarkers and lifestyle data to receive an instant,
        ML-powered wellness assessment — backed by a 96.79% recall score.
    </div>
    <div class="hero-pills">
        <span class="hero-pill">🔬 25 Clinical Features</span>
        <span class="hero-pill">⚙️ Feature Engineering</span>
        <span class="hero-pill">📊 Confidence Scoring</span>
        <span class="hero-pill">🟢 Real-time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. STAT CARDS
# ==========================================
st.markdown("""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-icon blue">🤖</div>
        <div>
            <div class="stat-value">AdaBoost</div>
            <div class="stat-label">Model</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon teal">🎯</div>
        <div>
            <div class="stat-value">96.79%</div>
            <div class="stat-label">Recall Score</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon green">🧬</div>
        <div>
            <div class="stat-value">25</div>
            <div class="stat-label">Features</div>
        </div>
    </div>
    <div class="stat-card">
        <div class="stat-icon amber">✅</div>
        <div>
            <div class="stat-value">Ready</div>
            <div class="stat-label">Model Status</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 5. PATIENT FORM INPUTS
# ==========================================

# — Demographics —
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon blue">👤</div>
        <div>
            <div class="section-title">Patient Demographics</div>
            <div class="section-sub">Basic identification and physical profile</div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
with col3:
    blood_group = st.selectbox("Blood Group", ["O", "A", "B", "AB"])
st.markdown("</div>", unsafe_allow_html=True)

# — Clinical Vitals —
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon rose">❤️</div>
        <div>
            <div class="section-title">Vital Clinical Metrics</div>
            <div class="section-sub">Core cardiovascular and metabolic markers</div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    bp = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
with col2:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=72.0)
with col3:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=50.0, max_value=500.0, value=190.0)
with col4:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=95.0)
st.markdown("</div>", unsafe_allow_html=True)

# — Lifestyle —
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon emerald">🧘</div>
        <div>
            <div class="section-title">Routine & Lifestyle Habits</div>
            <div class="section-sub">Daily activity, nutrition, and stress levels</div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    sleep = st.number_input("Sleep Duration (Hours)", min_value=0.0, max_value=24.0, value=7.5, step=0.5)
with col2:
    exercise = st.number_input("Daily Exercise (Hours)", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
with col3:
    water = st.number_input("Water Intake (Liters)", min_value=0.0, max_value=10.0, value=2.5, step=0.5)
with col4:
    diet_type = st.selectbox("Dietary Preference", ["Vegetarian", "Non Vegetarian", "Vegan"])

stress = st.slider("😓 Stress Assessment Level", 1, 10, 4, help="1 = Calm, 10 = High Stress")
st.markdown("</div>", unsafe_allow_html=True)

# — Risk Factors —
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon purple">🚬</div>
        <div>
            <div class="section-title">Risk Factors & Clinical History</div>
            <div class="section-sub">Behavioural risk indicators and pre-existing conditions</div>
        </div>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    smoking_option  = st.selectbox("Smoking Habits Scale", ["Low", "Medium", "High"])
    alcohol_option  = st.selectbox("Alcohol Consumption Frequency", ["Low", "Medium", "High"])
    diet_option     = st.selectbox("Nutritional Diet Quality Status", ["Good", "Average", "Poor"])
with col2:
    mental_option   = st.selectbox("Subjective Mental Health Rating", ["Good", "Average", "Poor"])
    physical_option = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
    medical_option  = st.selectbox("Pre-existing Medical History Severity", ["Low", "Moderate", "High"])

allergy_option = st.selectbox("Allergy Manifestation Profiles", ["Low", "Moderate", "High"])
st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 6. DATA ENCODING & FEATURE ENGINEERING
# ==========================================
risk_map     = {"Low": 0, "Medium": 1, "Moderate": 1, "High": 2}
health_map   = {"Poor": 0, "Average": 1, "Good": 2}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

smoking  = risk_map[smoking_option]
alcohol  = risk_map[alcohol_option]
diet     = health_map[diet_option]
mental   = health_map[mental_option]
physical = activity_map[physical_option]
medical  = risk_map[medical_option]
allergies = risk_map[allergy_option]

blood_ab = 1 if blood_group == "AB" else 0
blood_b  = 1 if blood_group == "B"  else 0
blood_o  = 1 if blood_group == "O"  else 0

vegan      = 1 if diet_type == "Vegan"      else 0
vegetarian = 1 if diet_type == "Vegetarian" else 0

internal_body_health = bp + heart_rate + chol + glucose + stress + bmi
daily_activity       = sleep + exercise + water
medical_status       = mental + physical + medical + allergies

# ==========================================
# 7. PREDICTION SECTION
# ==========================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-icon orange">📊</div>
        <div>
            <div class="section-title">Prediction Engine</div>
            <div class="section-sub">Run the AdaBoost model against the entered patient profile</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if st.button("🚀 Run Health Prediction Analysis"):
    try:
        model = load_model()

        input_data = {
            'Age':                    [age],
            'BMI':                    [bmi],
            'Blood_Pressure':         [bp],
            'Cholesterol':            [chol],
            'Glucose_Level':          [glucose],
            'Heart_Rate':             [heart_rate],
            'Sleep_Hours':            [sleep],
            'Exercise_Hours':         [exercise],
            'Water_Intake':           [water],
            'Stress_Level':           [stress],
            'Smoking':                [smoking],
            'Alcohol':                [alcohol],
            'Diet':                   [diet],
            'MentalHealth':           [mental],
            'PhysicalActivity':       [physical],
            'MedicalHistory':         [medical],
            'Allergies':              [allergies],
            'internal_body_health':   [internal_body_health],
            'daily_activity':         [daily_activity],
            'Medical_status':         [medical_status],
            'Diet_Type__Vegan_True':       [vegan],
            'Diet_Type__Vegetarian_True':  [vegetarian],
            'Blood_Group_AB_True':    [blood_ab],
            'Blood_Group_B_True':     [blood_b],
            'Blood_Group_O_True':     [blood_o],
        }

        input_df = pd.DataFrame(input_data)
        feature_columns = load_columns()
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        confidence  = max(probability) * 100

        health_score = max(0, 100 - (stress * 2 + smoking * 8 + alcohol * 6 + (chol / 10)))
        health_score = min(100, health_score)

        # — Metric mini-cards —
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="pred-metric-card">
                <div class="pred-metric-value">{confidence:.1f}%</div>
                <div class="pred-metric-label">Confidence</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="pred-metric-card">
                <div class="pred-metric-value">{health_score:.0f}</div>
                <div class="pred-metric-label">Wellness Score</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="pred-metric-card">
                <div class="pred-metric-value">{100 - health_score:.0f}%</div>
                <div class="pred-metric-label">Risk Level</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # — Probability bars —
        st.caption("Prediction Probabilities")
        col_a, col_b = st.columns(2)
        with col_a:
            st.progress(float(probability[1]))
            st.caption(f"🟢 Healthy — {probability[1]*100:.2f}%")
        with col_b:
            st.progress(float(probability[0]))
            st.caption(f"🔴 Unhealthy — {probability[0]*100:.2f}%")

        # — Result card —
        if prediction == 1:
            st.markdown(f"""
            <div class="result-healthy">
                <div class="result-badge">✅ Assessment Result</div>
                <div class="result-headline">🟢 Healthy</div>
                <div class="result-confidence">The model predicts a healthy profile with {confidence:.2f}% confidence.</div>
                <hr class="result-divider"/>
                <div class="result-meta">
                    <div class="result-meta-item">
                        <label>Wellness Score</label>
                        <div class="val">{health_score:.0f} / 100</div>
                    </div>
                    <div class="result-meta-item">
                        <label>Risk Level</label>
                        <div class="val">{100 - health_score:.0f}%</div>
                    </div>
                    <div class="result-meta-item">
                        <label>Confidence</label>
                        <div class="val">{confidence:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unhealthy">
                <div class="result-badge">⚠️ Assessment Result</div>
                <div class="result-headline">🔴 High Health Risk</div>
                <div class="result-confidence">The model flags elevated health risk with {confidence:.2f}% confidence. Please consult a physician.</div>
                <hr class="result-divider"/>
                <div class="result-meta">
                    <div class="result-meta-item">
                        <label>Wellness Score</label>
                        <div class="val">{health_score:.0f} / 100</div>
                    </div>
                    <div class="result-meta-item">
                        <label>Risk Level</label>
                        <div class="val">{100 - health_score:.0f}%</div>
                    </div>
                    <div class="result-meta-item">
                        <label>Confidence</label>
                        <div class="val">{confidence:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(health_score))
        st.caption(f"Estimated Wellness Score: {health_score:.0f} / 100")

        with st.expander("📋 View Transformed Patient Model Features"):
            st.dataframe(input_df)

    except FileNotFoundError:
        st.error("🚨 `health_model.pkl` or `feature_columns.pkl` not found. Verify runtime environment assets.")
    except Exception as e:
        st.error(f"An unexpected inference pipeline error occurred: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# 8. WELLNESS RECOMMENDATIONS
# ==========================================
st.markdown("""
<div class="section-card" style="border-top: 3px solid #0EA5E9;">
    <div class="section-header">
        <div class="section-icon blue">💡</div>
        <div>
            <div class="section-title">General Wellness Recommendations</div>
            <div class="section-sub">Evidence-based tips for a healthier lifestyle</div>
        </div>
    </div>
    <div class="tip-grid">
        <div class="tip-item">
            <div class="tip-emoji">💧</div>
            <div class="tip-text">
                <strong>Hydration Maintenance</strong>
                <span>Aim for 2.5–3 litres of water daily to support metabolic function.</span>
            </div>
        </div>
        <div class="tip-item">
            <div class="tip-emoji">😴</div>
            <div class="tip-text">
                <strong>Sleep Optimisation</strong>
                <span>7–8 hours of consistent sleep supports cognitive health and recovery.</span>
            </div>
        </div>
        <div class="tip-item">
            <div class="tip-emoji">🏋️</div>
            <div class="tip-text">
                <strong>Active Routine</strong>
                <span>Prioritise daily movement and structured physical activity.</span>
            </div>
        </div>
        <div class="tip-item">
            <div class="tip-emoji">🧘</div>
            <div class="tip-text">
                <strong>Stress Interventions</strong>
                <span>Mindfulness and breathing breaks reduce cognitive overexertion.</span>
            </div>
        </div>
        <div class="tip-item">
            <div class="tip-emoji">🥗</div>
            <div class="tip-text">
                <strong>Balanced Nutrition</strong>
                <span>High nutrient-density meals sustain consistent daily energy levels.</span>
            </div>
        </div>
        <div class="tip-item">
            <div class="tip-emoji">🩺</div>
            <div class="tip-text">
                <strong>Routine Assessments</strong>
                <span>Regular check-ups with qualified providers catch risks early.</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 9. MODEL INFO BANNER
# ==========================================
col_i1, col_i2, col_i3, col_i4 = st.columns(4)
with col_i1:
    st.info("🤖 **Model:** AdaBoost")
with col_i2:
    st.info("📊 **Recall:** 96.79%")
with col_i3:
    st.info("🧠 **Features:** 25")
with col_i4:
    st.info("⚙️ **Feature Engineering:** Enabled")

# ==========================================
# 10. FOOTER
# ==========================================
st.markdown("""
<div class="footer">
    <div>
        <div class="footer-brand">Vital<span>AI</span></div>
        <div class="footer-meta">
            Built by <strong style="color:#CBD5E1;">Ayush Kumar</strong><br>
            AI Health Prediction System · For research & educational purposes only
        </div>
    </div>
    <div class="footer-tech">
        <span class="footer-tag">Python</span>
        <span class="footer-tag">Scikit-Learn</span>
        <span class="footer-tag">AdaBoost</span>
        <span class="footer-tag">Feature Engineering</span>
        <span class="footer-tag">Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)
