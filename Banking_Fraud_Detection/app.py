import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Banking Fraud Analytics",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# CUSTOM DASHBOARD THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0b1020, #101a35, #111827);
    color: #f1f5f9;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

h1, h2, h3 {
    color: #f8fafc !important;
}

p, label, span {
    color: #e2e8f0;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172554, #312e81);
    border: 1px solid #6366f1;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 5px 15px rgba(99,102,241,0.15);
}

[data-testid="stMetricLabel"] {
    color: #c4b5fd !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1e1b4b);
    border-right: 1px solid #4f46e5;
}

div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    border-radius: 10px;
    border: none;
}

div.stDownloadButton > button {
    background: linear-gradient(90deg, #06b6d4, #10b981);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #6366f1;
    border-radius: 12px;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# COLOR PALETTE
# ==========================================

COLORS = {
    "Genuine": "#06D6A0",
    "Fraud": "#FF4D6D",
    "High Risk": "#FF4D6D",
    "Medium Risk": "#FFB703",
    "Low Risk": "#06D6A0",
    "Blue": "#3A86FF",
    "Purple": "#8338EC",
    "Cyan": "#00B4D8",
    "Orange": "#FB8500"
}

CHART_TEMPLATE = "plotly_dark"

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_files():
    model = joblib.load("fraud_detection_model.pkl")
    scaler = joblib.load("fraud_detection_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_files()

except Exception as e:
    st.error("Model or scaler could not be loaded.")
    st.write(e)
    st.stop()

# ==========================================
# FEATURES
# ==========================================

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div style="
    background: linear-gradient(100deg, #312e81, #4f46e5, #0891b2);
    padding: 28px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #818cf8;
">

<h1 style="color:white; margin:0;">
🏦 AI Banking Fraud Analytics
</h1>

<p style="color:#e0f2fe; font-size:17px; margin-top:10px;">
Intelligent Transaction Monitoring & Machine Learning Fraud Detection
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("""
<h1 style="color:#a5b4fc;">⚙️ Control Center</h1>
""", unsafe_allow_html=True)

st.sidebar.write("Upload transaction data for AI-powered fraud analysis.")

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

st.sidebar.divider()

st.sidebar.markdown("### 🧠 Model Performance")

st.sidebar.metric("Precision", "91%")
st.sidebar.metric("Recall", "79%")
st.sidebar.metric("F1 Score", "84%")
st.sidebar.metric("ROC-AUC", "95%")

st.sidebar.divider()

st.sidebar.markdown("""
<div style="
background:linear-gradient(135deg,#312e81,#0891b2);
padding:15px;
border-radius:12px;
text-align:center;
">
<p style="color:white;font-weight:bold;">
AI Powered Banking Security
</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# UPLOAD SCREEN
# ==========================================

if uploaded_file is None:

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#172554,#312e81);
    padding:25px;
    border-radius:16px;
    border:1px solid #818cf8;
    ">
    <h2 style="color:#a5b4fc;">
    📊 Dashboard Overview
    </h2>
    <p>
    Upload your transaction dataset to explore fraud detection,
    transaction risk and financial analytics.
    </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    a.info("📊 Transaction Analysis")
    b.success("🛡️ Fraud Detection")
    c.warning("⚠️ Risk Monitoring")

    st.stop()

# ==========================================
# READ CSV
# ==========================================

try:
    df = pd.read_csv(uploaded_file)

except Exception as e:
    st.error("Could not read uploaded CSV.")
    st.write(e)
    st.stop()

# ==========================================
# VALIDATION
# ==========================================

missing_features = [
    col for col in FEATURES if col not in df.columns
]

if missing_features:
    st.error("Required model features are missing.")
    st.write(missing_features)
    st.stop()

if df.empty:
    st.warning("The dataset is empty.")
    st.stop()

# ==========================================
# PREDICTION
# ==========================================

try:

    X = df[FEATURES].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isnull().any().any():
        st.error("Missing or invalid values found in features.")
        st.stop()

    X = X.astype(float)

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    probabilities = model.predict_proba(X_scaled)

    fraud_probability = probabilities[:, 1]

except Exception as e:

    st.error("Prediction failed.")
    st.write(e)
    st.stop()

# ==========================================
# RESULT DATA
# ==========================================

result_df = df.copy()

result_df["Predicted_Class"] = predictions

result_df["Fraud_Probability"] = fraud_probability

result_df["Prediction_Status"] = np.where(
    predictions == 1,
    "Fraud",
    "Genuine"
)

result_df["Risk_Level"] = np.where(
    fraud_probability >= 0.80,
    "High Risk",
    np.where(
        fraud_probability >= 0.50,
        "Medium Risk",
        "Low Risk"
    )
)

# ==========================================
# KPI CALCULATIONS
# ==========================================

total_transactions = len(result_df)

fraud_count = int((predictions == 1).sum())

genuine_count = int((predictions == 0).sum())

fraud_percentage = fraud_count / total_transactions * 100

average_amount = result_df["Amount"].mean()

high_risk_count = int(
    (result_df["Fraud_Probability"] >= 0.80).sum()
)

# ==========================================
# KPI DASHBOARD
# ==========================================

st.markdown("## 📌 Transaction Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "💳 Total Transactions",
    f"{total_transactions:,}"
)

k2.metric(
    "🚨 Fraud Detected",
    f"{fraud_count:,}"
)

k3.metric(
    "🟢 Genuine Transactions",
    f"{genuine_count:,}"
)

k4.metric(
    "📈 Fraud Rate",
    f"{fraud_percentage:.2f}%"
)

st.divider()

# ==========================================
# CHART 1 - FRAUD VS GENUINE
# ==========================================

st.markdown("## 📊 Transaction Analytics")

col1, col2 = st.columns(2)

count_df = pd.DataFrame({
    "Status": ["Genuine", "Fraud"],
    "Count": [genuine_count, fraud_count]
})

with col1:

    st.markdown("### 🟢 Fraud vs Genuine")

    fig1 = px.bar(
        count_df,
        x="Status",
        y="Count",
        color="Status",
        text="Count",
        color_discrete_map=COLORS,
        template=CHART_TEMPLATE
    )

    fig1.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    fig1.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

# ==========================================
# CHART 2 - PROBABILITY
# ==========================================

with col2:

    st.markdown("### 🔮 Fraud Probability")

    fig2 = px.histogram(
        result_df,
        x="Fraud_Probability",
        nbins=25,
        template=CHART_TEMPLATE,
        color_discrete_sequence=[COLORS["Purple"]]
    )

    fig2.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================================
# CHART 3 - AMOUNT
# ==========================================

col3, col4 = st.columns(2)

with col3:

    st.markdown("### 💰 Transaction Amount")

    fig3 = px.histogram(
        result_df,
        x="Amount",
        nbins=30,
        template=CHART_TEMPLATE,
        color_discrete_sequence=[COLORS["Cyan"]]
    )

    fig3.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ==========================================
# CHART 4 - RISK CATEGORY
# ==========================================

with col4:

    st.markdown("### 🎯 Risk Categories")

    risk_counts = (
        result_df["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["Risk_Level", "Count"]

    fig4 = px.pie(
        risk_counts,
        names="Risk_Level",
        values="Count",
        hole=0.50,
        template=CHART_TEMPLATE,
        color="Risk_Level",
        color_discrete_map=COLORS
    )

    fig4.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# ==========================================
# ADVANCED ANALYTICS
# ==========================================

st.markdown("## 🔍 Advanced Analytics")

col5, col6 = st.columns(2)

# ==========================================
# CHART 5
# ==========================================

with col5:

    st.markdown("### ⚠️ Fraud Probability vs Amount")

    fig5 = px.scatter(
        result_df,
        x="Amount",
        y="Fraud_Probability",
        color="Prediction_Status",
        color_discrete_map=COLORS,
        opacity=0.65,
        template=CHART_TEMPLATE
    )

    fig5.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# ==========================================
# CHART 6
# ==========================================

with col6:

    st.markdown("### ⏰ Transaction Activity")

    time_df = result_df.copy()

    time_df["Time_Bucket"] = (
        time_df["Time"] // 3600
    )

    time_counts = (
        time_df.groupby("Time_Bucket")
        .size()
        .reset_index(name="Count")
    )

    fig6 = px.line(
        time_counts,
        x="Time_Bucket",
        y="Count",
        markers=True,
        template=CHART_TEMPLATE,
        color_discrete_sequence=[COLORS["Orange"]]
    )

    fig6.update_traces(
        line=dict(width=3)
    )

    fig6.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# ==========================================
# CHART 7 - HIGH RISK
# ==========================================

st.markdown("## 🚨 High Risk Monitoring")

col7, col8 = st.columns(2)

with col7:

    high_risk_summary = pd.DataFrame({
        "Category": ["High Risk", "Other Transactions"],
        "Count": [
            high_risk_count,
            total_transactions - high_risk_count
        ]
    })

    fig7 = px.bar(
        high_risk_summary,
        x="Category",
        y="Count",
        color="Category",
        text="Count",
        template=CHART_TEMPLATE,
        color_discrete_sequence=[
            COLORS["Fraud"],
            COLORS["Blue"]
        ]
    )

    fig7.update_layout(
        height=350,
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    fig7.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig7,
        use_container_width=True
    )

# ==========================================
# RISK KPIs
# ==========================================

with col8:

    st.metric(
        "🔴 High Risk Transactions",
        f"{high_risk_count:,}"
    )

    st.metric(
        "💵 Average Transaction Amount",
        f"{average_amount:,.2f}"
    )

    st.metric(
        "📊 Maximum Fraud Probability",
        f"{fraud_probability.max():.2%}"
    )

# ==========================================
# FRAUD TABLE
# ==========================================

st.divider()

st.markdown("## 🚨 Fraudulent Transactions")

fraud_df = result_df[
    result_df["Predicted_Class"] == 1
].sort_values(
    "Fraud_Probability",
    ascending=False
)

if not fraud_df.empty:

    st.dataframe(
        fraud_df.head(100),
        use_container_width=True,
        height=350
    )

else:

    st.success("No fraudulent transactions detected.")

# ==========================================
# COMPLETE DATA
# ==========================================

st.markdown("## 📋 Transaction Records")

st.dataframe(
    result_df.head(500),
    use_container_width=True,
    height=350
)

# ==========================================
# DOWNLOAD
# ==========================================

st.divider()

st.markdown("## 📥 Export Analysis")

csv_data = result_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "⬇️ Download Fraud Analysis Report",
    data=csv_data,
    file_name="fraud_predictions.csv",
    mime="text/csv"
)

st.markdown("""
<div style="
background:linear-gradient(90deg,#312e81,#4f46e5,#0891b2);
padding:18px;
border-radius:14px;
text-align:center;
margin-top:20px;
">

<h3 style="color:white;">
🛡️ AI Banking Fraud Detection System
</h3>

<p style="color:#e0f2fe;">
Machine Learning • Financial Analytics • Fraud Monitoring
</p>

</div>
""", unsafe_allow_html=True)

st.success("✅ Analysis completed successfully!")