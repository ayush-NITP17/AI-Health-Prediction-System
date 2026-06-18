import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="AI Health Prediction System",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background:
    linear-gradient(135deg,#071B2F 0%,#0A2540 50%,#123C69 100%);
}

/* Make Input Framework Elements Highly Visible */
div[data-baseweb="select"]{
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
}

input{
    border-radius:12px !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Glass Card */
.glass-card{
    background:rgba(255,255,255,0.12);
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,0.15);
    border-radius:24px;
    padding:25px;
    margin-bottom:20px;
    box-shadow:0 8px 30px rgba(0,0,0,0.25);
}

/* Section Title & Form Polish */
.section-title {
    color: #00E5FF;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 15px;
}
label {
    color: #E2E8F0 !important;
}

/* Hero Banner */
.hero{
    background:linear-gradient(135deg,#00C6FF,#0072FF);
    border-radius:30px;
    padding:50px;
    color:white;
    text-align:center;
    box-shadow:0 12px 40px rgba(0,114,255,.35);
    margin-bottom: 25px;
}

.hero h1{
    font-size:3rem;
    font-weight:700;
}

.hero p{
    font-size:1.2rem;
    opacity:.9;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:65px;
    border:none;
    border-radius:18px;
    background:linear-gradient(135deg,#00E5FF,#00B4D8);
    color:white;
    font-size:1.15rem;
    font-weight:700;
    transition:.3s;
    box-shadow: 0px 4px 15px rgba(0, 229, 255, 0.3);
}

.stButton>button:hover{
    transform:translateY(-4px);
    box-shadow: 0px 6px 20px rgba(0, 229, 255, 0.5);
    color: white !important;
}

/* Metric Cards */
.metric-card{
    background:white;
    border-radius:20px;
    padding:20px;
    text-align:center;
    box-shadow:0 8px 25px rgba(0,0,0,.1);
}

.metric-value{
    font-size:2rem;
    font-weight:700;
    color:#0072FF;
}

.metric-label{
    color:gray;
}

/* Healthy */
.healthy{
    background:linear-gradient(135deg,#00C853,#69F0AE);
    color:white;
    border-radius:20px;
    padding:25px;
    text-align:center;
    font-size:1.5rem;
    font-weight:700;
    margin-top: 15px;
}

/* Unhealthy */
.unhealthy{
    background:linear-gradient(135deg,#D50000,#FF5252);
    color:white;
    border-radius:20px;
    padding:25px;
    text-align:center;
    font-size:1.5rem;
    font-weight:700;
    margin-top: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
}
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
# 2. SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.markdown("### 🏥 System Dashboard")
    st.success("🤖 AdaBoost Recall: **96.79%**")
    
    st.info(
        "**Instructions:**\n"
        "Enter patient details and lifestyle metrics in the main panel. "
        "The system will compute risk factors and output a wellness assessment prediction."
    )
    
    st.markdown("---")
    st.markdown("📋 **Evaluated Biomarkers:**")
    st.caption("✔ Blood Pressure & Vitals\n\n"
               "✔ Cholesterol & Glucose\n\n"
               "✔ Lifestyle & Mental Well-being\n\n"
               "✔ Medical Records & Allergies")

# ==========================================
# 3. HERO BANNER & ANALYTICS ROW
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🏥 AI Health Prediction System</h1>
    <p>
        Advanced Machine Learning Powered Healthcare Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model","AdaBoost")

with c2:
    st.metric("Features","25")

with c3:
    st.metric("Recall","96.79%")

with c4:
    st.metric("Status","Ready")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 4. PATIENT FORM INPUTS (GLASS CARDS)
# ==========================================

# Form Card 1: Demographics
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 Patient Demographics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=22.5, step=0.1)
with col3:
    blood_group = st.selectbox("Blood Group", ["O", "A", "B", "AB"])
st.markdown('</div>', unsafe_allow_html=True)

# Form Card 2: Clinical Vitals
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">❤️ Vital Clinical Metrics</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    bp = st.number_input("Systolic BP (mmHg)", min_value=50.0, max_value=250.0, value=120.0)
with col2:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=72.0)
with col3:
    chol = st.number_input("Cholesterol (mg/dL)", min_value=50.0, max_value=500.0, value=190.0)
with col4:
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50.0, max_value=400.0, value=95.0)
st.markdown('</div>', unsafe_allow_html=True)

# Form Card 3: Lifestyle & Environment
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧘 Routine & Lifestyle Habits</div>', unsafe_allow_html=True)
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
st.markdown('</div>', unsafe_allow_html=True)

# Form Card 4: Clinical History & Risk Factors
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚬 Risk Factors & Clinical History</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    smoking_option = st.selectbox("Smoking Habits Scale", ["Low", "Medium", "High"])
    alcohol_option = st.selectbox("Alcohol Consumption Frequency", ["Low", "Medium", "High"])
    diet_option = st.selectbox("Nutritional Diet Quality Status", ["Good", "Average", "Poor"])

with col2:
    mental_option = st.selectbox("Subjective Mental Health Rating", ["Good", "Average", "Poor"])
    physical_option = st.selectbox("Physical Activity Level", ["High", "Moderate", "Low"])
    medical_option = st.selectbox("Pre-existing Medical History Severity", ["Low", "Moderate", "High"])

allergy_option = st.selectbox("Allergy Manifestation Profiles", ["Low", "Moderate", "High"])
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. DATA ENCODING & FEATURE ENGINEERING
# ==========================================
risk_map = {"Low": 0, "Medium": 1, "Moderate": 1, "High": 2}
health_map = {"Poor": 0, "Average": 1, "Good": 2}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

smoking = risk_map[smoking_option]
alcohol = risk_map[alcohol_option]
diet = health_map[diet_option]
mental = health_map[mental_option]
physical = activity_map[physical_option]
medical = risk_map[medical_option]
allergies = risk_map[allergy_option]

# Categorical Feature One-Hot Encoding (Matches precise training schema)
blood_ab = 1 if blood_group == "AB" else 0
blood_b  = 1 if blood_group == "B" else 0
blood_o  = 1 if blood_group == "O" else 0

vegan = 1 if diet_type == "Vegan" else 0
vegetarian = 1 if diet_type == "Vegetarian" else 0

# Feature Aggregations
internal_body_health = bp + heart_rate + chol + glucose + stress + bmi
daily_activity = sleep + exercise + water
medical_status = mental + physical + medical + allergies

# ==========================================
# 6. MODEL INFERENCE ENGINE
# ==========================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Prediction Output</div>', unsafe_allow_html=True)

if st.button("🚀 Run Health Prediction Analysis"):
    try:
        # Load the trained model pipeline securely using resource cache optimization
        model = load_model()
        
        # EXACT column layout mapping expected by your Scikit-Learn Pipeline
        input_data = {
            'Age': [age],
            'BMI': [bmi],
            'Blood_Pressure': [bp],
            'Cholesterol': [chol],
            'Glucose_Level': [glucose],
            'Heart_Rate': [heart_rate],
            'Sleep_Hours': [sleep],
            'Exercise_Hours': [exercise],
            'Water_Intake': [water],
            'Stress_Level': [stress],
            'Smoking': [smoking],
            'Alcohol': [alcohol],
            'Diet': [diet],
            'MentalHealth': [mental],
            'PhysicalActivity': [physical],
            'MedicalHistory': [medical],
            'Allergies': [allergies],
            'internal_body_health': [internal_body_health],
            'daily_activity': [daily_activity],
            'Medical_status': [medical_status],
            
            'Diet_Type__Vegan_True': [vegan],
            'Diet_Type__Vegetarian_True': [vegetarian],
            'Blood_Group_AB_True': [blood_ab],
            'Blood_Group_B_True': [blood_b],
            'Blood_Group_O_True': [blood_o]
        }
        
        input_df = pd.DataFrame(input_data)

        feature_columns = load_columns()
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Real-time Inference Execution
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        st.subheader("Prediction Probabilities")

        st.progress(float(probability[1]))

        st.write(
            f"Healthy: {probability[1]*100:.2f}%"
        )

        st.write(
            f"Unhealthy: {probability[0]*100:.2f}%"
        )
        confidence = max(probability) * 100
        
        # Dynamic Feature Engineering Metrics
        health_score = max(0, 100 - (stress * 2 + smoking * 8 + alcohol * 6 + (chol / 10)))
        health_score = min(100, health_score)
        
        # Prediction Output Dashboard Layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{confidence:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{health_score:.0f}</div>
                <div class="metric-label">Estimated Wellness Score</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{100 - health_score:.0f}%</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(health_score))
        
        # Display Styled Result Custom Card based on prediction value
        if prediction == 1:
            st.markdown(f"""
            <div class="healthy">
                🟢 HEALTHY<br>
                Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="unhealthy">
                🔴 HIGH HEALTH RISK<br>
                Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View Transformed Patient Model Features"):
            st.dataframe(input_df)
            
    except FileNotFoundError:
        st.error("🚨 `health_model.pkl` or feature structural mapping files not found! Please verify runtime environment assets.")
    except Exception as e:
        st.error(f"An unexpected inference pipeline error occurred: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. GENERAL WELLNESS RECOMMENDATIONS
# ==========================================
st.markdown("---")
st.markdown('<div class="glass-card" style="border-top: 4px solid #E07A5F;">', unsafe_allow_html=True)
st.markdown('<div class="section-title" style="color: #E07A5F;">💡 General Wellness Recommendations</div>', unsafe_allow_html=True)

rec_col1, rec_col2 = st.columns(2)
with rec_col1:
    st.markdown("""
    * 💧 **Hydration Maintenance:** Aim for 2.5 to 3 Liters of standard fresh water daily.
    * 🏋️ **Active Routine:** Prioritize moving or routine basic physical activity during the week.
    * 🥗 **Balanced Fuel:** Focus on high nutrient-density macro structures to support consistent daily energy.
    """)
with rec_col2:
    st.markdown("""
    * 😴 **Sleep Optimization:** Strive for 7-8 hours of sound sleep windows to ensure cognitive health recovery.
    * 🧘 **Stress Interventions:** Implement intentional mindfulness breathing breaks to reduce cognitive overexertion.
    * 🩺 **Routine Assessment:** Routinely book physical examinations with qualified wellness providers.
    """)
st.markdown('</div>', unsafe_allow_html=True)

st.info("""
🤖 Model Used: AdaBoost Classifier

📊 Recall: 96.79%

🧠 Features: 25

⚙️ Feature Engineering Enabled
""")

# ==========================================
# 8. APPLICATION FOOTER
# ==========================================
st.markdown("""
<div style='text-align:center;color:#94A3B8;padding:10px;'>
Built by <b>Ayush Kumar</b> using Python, Scikit-Learn, AdaBoost, Feature Engineering, and Streamlit
</div>
""", unsafe_allow_html=True)