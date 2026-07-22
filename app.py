import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------------------------------------------------------
# 1. PAGE LAYOUT & APP CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="APL Logistics | Risk Prediction Desk",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚢 APL Logistics Late Delivery Risk Prediction Desk")
st.caption("Global Supply Chain Operations Optimization Engine")
st.write("---")

# -----------------------------------------------------------------------------
# 2. DATA PIPELINE WITH GOOGLE DRIVE DIRECT LOAD
# -----------------------------------------------------------------------------
# ⚠️ REPLACE THIS WITH YOUR ACTUAL GOOGLE DRIVE FILE ID FROM STEP 1
GOOGLE_DRIVE_FILE_ID = https://drive.google.com/file/d/1KoQrXqioQnKdcLbnuOr52EWew-XVw7VB/view?usp=sharing

@st.cache_data
def run_data_pipeline():
    file_path = 'APL_Logistics.csv'
    
    # If the CSV isn't on the server, download it directly from Google Drive!
    if not os.path.exists(file_path):
        drive_url = 'https://drive.google.com/uc?id=https://drive.google.com/file/d/1KoQrXqioQnKdcLbnuOr52EWew-XVw7VB/view?usp=sharing
        df = pd.read_csv(drive_url, encoding='latin1')
    else:
        df = pd.read_csv(file_path, encoding='latin1')
    
    # Preprocessing: Handle missing values safely
    df['Customer Lname'] = df['Customer Lname'].fillna('')
    df['Customer Zipcode'] = df['Customer Zipcode'].fillna(0)
    
    # 1. Shipping Pressure Index
    df['Shipping_Pressure_Index'] = df['Days for shipment (scheduled)'] / (df['Order Item Quantity'] + 0.1)
    
    # 2. Mode Risk Flags
    mode_map = {'Same Day': 0.9, 'First Class': 0.7, 'Second Class': 0.4, 'Standard Class': 0.2}
    df['Mode_Risk_Flag'] = df['Shipping Mode'].map(mode_map).fillna(0.3)
    
    # 3. Order Complexity Score
    df['Order_Complexity_Score'] = (df['Order Item Quantity'] * df['Order Item Product Price']) / 100.0
    
    return df

try:
    data = run_data_pipeline()
except Exception as e:
    st.error(f"❌ Data Load Error: Ensure the Google Drive link is set to 'Anyone with the link can view'. Details: {e}")
    st.stop()
# -----------------------------------------------------------------------------
# 3. GLOBAL USER CAPABILITIES (SIDEBAR FILTERS)
# -----------------------------------------------------------------------------
st.sidebar.header("🕹️ Control Room Filters")

# Capability 1: Shipping Mode Filter
selected_mode = st.sidebar.selectbox("Shipping Mode", ["All Modes"] + list(data['Shipping Mode'].unique()))

# Capability 2: Region Selector
selected_market = st.sidebar.selectbox("Global Market", ["All Markets"] + list(data['Market'].unique()))

# Capability 3: Customer Segment Filter
selected_segment = st.sidebar.selectbox("Customer Segment", ["All Segments"] + list(data['Customer Segment'].unique()))

# Capability 4: Risk Threshold Slider
risk_threshold = st.sidebar.slider("Operational Risk Threshold", 0.10, 0.90, 0.50, 0.05)

# Execute filtering framework
df_filtered = data.copy()
if selected_mode != "All Modes":
    df_filtered = df_filtered[df_filtered['Shipping Mode'] == selected_mode]
if selected_market != "All Markets":
    df_filtered = df_filtered[df_filtered['Market'] == selected_market]
if selected_segment != "All Segments":
    df_filtered = df_filtered[df_filtered['Customer Segment'] == selected_segment]

# -----------------------------------------------------------------------------
# 4. PREDICTIVE ENGINE PIPELINE (TRAINING & EVALUATION)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_and_evaluate_models(df):
    features = ['Days for shipment (scheduled)', 'Shipping_Pressure_Index', 'Mode_Risk_Flag', 'Order_Complexity_Score', 'Sales']
    X = df[features].fillna(0)
    y = df['Late_delivery_risk']
    
    # Use downsampled subset for rapid runtime calculations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=30000, random_state=42)
    
    # Baseline Model: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sub, y_train_sub)
    
    # Advanced Model: Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_sub, y_train_sub)
    
    # Compute performance evaluation matrices
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }
    
    return rf, lr, metrics

rf_model, lr_model, eval_metrics = train_and_evaluate_models(data)

# Inject predictions into filtered space
features_list = ['Days for shipment (scheduled)', 'Shipping_Pressure_Index', 'Mode_Risk_Flag', 'Order_Complexity_Score', 'Sales']
df_filtered['Risk_Probability'] = rf_model.predict_proba(df_filtered[features_list].fillna(0))[:, 1]

# Dynamic Risk Categorization
df_filtered['Risk_Tier'] = np.where(
    df_filtered['Risk_Probability'] >= risk_threshold + 0.15, 'High Risk',
    np.where(df_filtered['Risk_Probability'] >= risk_threshold, 'Medium Risk', 'Low Risk')
)

# -----------------------------------------------------------------------------
# 5. RENDER SYSTEM WORKSPACE TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Delay Risk Overview",
    "🎯 Order-Level Risk Prediction",
    "🗺️ Region & Mode Risk Analysis",
    "⚙️ Model Evaluation Matrices",
    "🚨 Operations Action Panel"
])

# MODULE 1: DELAY RISK OVERVIEW
with tab1:
    st.subheader("Fulfillment Fleet Health Indicators")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="card-box"><b>Total Tracked Shipments</b><br><span style="font-size:1.8rem; font-weight:700;">{len(df_filtered):,}</span></div>', unsafe_allow_html=True)
    with m2:
        critical_count = len(df_filtered[df_filtered['Risk_Tier'] == 'High Risk'])
        st.markdown(f'<div class="card-box" style="border-top: 4px solid #EF4444;"><b>Critical Exception Queue</b><br><span style="font-size:1.8rem; font-weight:700; color:#EF4444;">{critical_count:,}</span></div>', unsafe_allow_html=True)
    with m3:
        mean_risk = df_filtered['Risk_Probability'].mean() * 100 if len(df_filtered) > 0 else 0
        st.markdown(f'<div class="card-box" style="border-top: 4px solid #F59E0B;"><b>Network Risk Average</b><br><span style="font-size:1.8rem; font-weight:700; color:#F59E0B;">{mean_risk:.2f}%</span></div>', unsafe_allow_html=True)
        
    st.write("")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.histplot(data=df_filtered, x='Risk_Probability', bins=30, color='#1E3A8A', kde=True, ax=ax)
    ax.axvline(risk_threshold, color='#EF4444', linestyle='--', linewidth=2, label=f'Intervention Bar ({risk_threshold:.2f})')
    ax.set_title("Probability Distribution Profile for Shipment Delays", fontsize=11, fontweight='bold')
    ax.set_xlabel("Calculated Late Delivery Probability")
    ax.set_ylabel("Order Volumetrics")
    ax.legend()
    st.pyplot(fig)

# MODULE 2: ORDER-LEVEL RISK PREDICTION
with tab2:
    st.subheader("Shipment Routing Risk Diagnostics")
    
    if len(df_filtered) == 0:
        st.info("No records match your selected filters.")
    else:
        sample_rows = df_filtered.index[:10].tolist()
        selected_index = st.selectbox("Select Shipment Row ID for Forensic Breakdown:", sample_rows)
        row = df_filtered.loc[selected_index]
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### 📋 Routing Log Details")
            st.write(f"**Customer:** {row['Customer Fname']} {row['Customer Lname']} ({row['Customer Segment']})")
            st.write(f"**Destination Hub:** {row['Order City']}, {row['Order Country']}")
            st.write(f"**Product Category:** {row['Category Name']}")
            st.write(f"**Item Ordered:** {row['Product Name']}")
            st.write(f"**Scheduled Delivery Buffer:** {row['Days for shipment (scheduled)']} Days")
            st.write(f"**Financial Sale Value:** ${row['Sales']:.2f}")
            
        with c_right:
            st.markdown("#### 🧠 AI Explainability Engine Insights")
            st.write(f"**Calculated Late Probability Index:** `{row['Risk_Probability']:.4f}`")
            
            # Key Contributing Drivers Breakdown
            st.write("**Identified Risk Driver Weights:**")
            st.write(f"* *Shipping Pressure Index:* `{row['Shipping_Pressure_Index']:.3f}`")
            st.write(f"* *Mode Congestion Risk Factor:* `{row['Mode_Risk_Flag']:.3f}`")
            st.write(f"* *Order Complexity Score:* `{row['Order_Complexity_Score']:.3f}`")
            
            if row['Risk_Tier'] == 'High Risk':
                st.markdown('<div class="directive-alert" style="background-color: #FEE2E2; color: #991B1B;">🔴 RISK PROFILE: CRITICAL SHIPMENT DELAY HAZARD<br><br><b>Prescriptive Action Required:</b> Immediately execute premium logistics rerouting. Apply high-priority cross-docking protocol at the local hub to avoid SLA penalties.</div>', unsafe_allow_html=True)
            elif row['Risk_Tier'] == 'Medium Risk':
                st.markdown('<div class="directive-alert" style="background-color: #FEF3C7; color: #92400E;">🟡 RISK PROFILE: MONITOR STATUS WARNING<br><br><b>Prescriptive Action Required:</b> Audit warehouse loading priority list within 12 hours. Track potential port or regional transit corridor slowdowns.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="directive-alert" style="background-color: #D1FAE5; color: #065F46;">🟢 RISK PROFILE: OPTIMAL METRICS CONFIRMED<br><br><b>Prescriptive Action Required:</b> No operational adjustment required. Standard background logging active. Proceed through regular line tracking.</div>', unsafe_allow_html=True)

# MODULE 3: REGION & MODE RISK ANALYSIS
with tab3:
    st.subheader("Macro Congestion Bottleneck Assessments")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🗺️ Market Regional Congestion Heatmap Matrix")
        reg_data = df_filtered.groupby('Order Region')['Risk_Probability'].mean().reset_index().sort_values(by='Risk_Probability', ascending=False)
        fig_reg, ax_reg = plt.subplots(figsize=(6, 4))
        sns.barplot(data=reg_data.head(8), y='Order Region', x='Risk_Probability', palette='ch:start=.2,rot=-.3_r', ax=ax_reg)
        ax_reg.set_xlabel("Mean Delay Likelihood Index")
        ax_reg.set_ylabel("")
        st.pyplot(fig_reg)
        
    with col_r:
        st.markdown("#### 🚢 Fulfillment Performance Across Shipping Modes")
        mode_data = df_filtered.groupby('Shipping Mode')['Risk_Probability'].mean().reset_index().sort_values(by='Risk_Probability', ascending=False)
        fig_mode, ax_mode = plt.subplots(figsize=(6, 4))
        sns.barplot(data=mode_data, x='Shipping Mode', y='Risk_Probability', palette='magma', ax=ax_mode)
        ax_mode.set_ylabel("Mean Delay Likelihood Index")
        ax_mode.set_xlabel("")
        st.pyplot(fig_mode)

# MODULE 4: MODEL EVALUATION MATRICES
with tab4:
    st.subheader("Machine Learning Performance Verification")
    
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Overall Prediction Quality (ROC-AUC)", f"{eval_metrics['ROC-AUC']:.4f}")
    with c_m2:
        st.metric("Precision Score (False Alarm Prevention)", f"{eval_metrics['Precision']:.4f}")
    with c_m3:
        st.metric("Recall Rating (True Capture Rate)", f"{eval_metrics['Recall']:.4f}")
    with c_m4:
        st.metric("F1-Score Benchmark", f"{eval_metrics['F1 Score']:.4f}")
        
    st.write("---")
    st.markdown("#### 📉 Algorithm Interpretability Benchmarking Matrix (Confusion Matrix)")
    
    cm = eval_metrics['Confusion Matrix']
    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted On-Time', 'Predicted Late'], yticklabels=['Actual On-Time', 'Actual Late'], ax=ax_cm)
    st.pyplot(fig_cm)

# MODULE 5: OPERATIONS ACTION PANEL
with tab5:
    st.subheader("🚨 Urgent Action Rerouting Queue")
    st.markdown("The following records represent deliveries violating active risk threshold baselines. Operations teams should treat this view as a priority action dispatch desk.")
    
    action_queue = df_filtered[df_filtered['Risk_Tier'].isin(['High Risk', 'Medium Risk'])].sort_values(by='Risk_Probability', ascending=False)
    
    if len(action_queue) == 0:
        st.success("🟢 All operations running within optimal target fulfillment profiles.")
    else:
        out_cols = ['Customer City', 'Market', 'Shipping Mode', 'Product Name', 'Risk_Probability', 'Risk_Tier']
        st.dataframe(action_queue[out_cols].head(100), use_container_width=True)
        st.caption(f"Showing top 100 entries requiring attention out of {len(action_queue)} active exceptions.")
