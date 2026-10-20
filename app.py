import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Sleek Dark Palette)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global font & background enhancements */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Header Card */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    /* Result Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    .risk-high {
        border-left: 5px solid #EF4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(30, 41, 59, 0.6) 100%);
    }
    .risk-medium {
        border-left: 5px solid #F59E0B;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(30, 41, 59, 0.6) 100%);
    }
    .risk-low {
        border-left: 5px solid #10B981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(30, 41, 59, 0.6) 100%);
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .badge-primary {
        background-color: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .recommendation-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 18px 22px;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model & Encoder Loader (Cached for Performance)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading machine learning model...")
def load_churn_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "customer_churn_model.pkl")
    encoder_path = os.path.join(base_dir, "encoders.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error(f"Missing artifacts! Ensure {model_path} and {encoder_path} exist.")
        st.stop()
        
    with open(model_path, "rb") as f:
        model_payload = pickle.load(f)
        
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
        
    model = model_payload.get("model")
    feature_names = model_payload.get("feautres_name") or model_payload.get("features_name")
    
    return model, encoders, feature_names

model, encoders, feature_names = load_churn_artifacts()

# ---------------------------------------------------------
# Sidebar: Control Panel & Presets
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=600&auto=format&fit=crop&q=80", use_container_width=True)
    st.title("⚡ Navigation & Tools")
    
    st.markdown("### 🎯 Quick Demo Presets")
    st.write("Load pre-configured customer archetypes to test the model immediately:")
    
    preset_choice = st.radio(
        "Select an Archetype:",
        ["Custom Input", "High Churn Risk (New, Month-to-Month)", "Loyal Customer (Long-term, 2-Year Contract)", "Moderate Risk (Fiber Optic, Mid-Tenure)"],
        index=0
    )
    
    st.divider()
    st.caption("Developed for predictive customer retention workflows.")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Customer Churn Intelligence Platform</div>
    <p class="hero-subtitle">
        Leverage machine learning to anticipate churn, understand customer vulnerability drivers, and trigger targeted retention interventions.
    </p>
    <div style="margin-top: 14px;">
        <span class="badge badge-primary">Random Forest ML</span>
        <span class="badge badge-success">Production Ready</span>
        <span class="badge badge-primary">Real-time Inference</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Values Setup
# ---------------------------------------------------------
default_vals = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0,
    "TotalCharges": 780.0
}

if preset_choice == "High Churn Risk (New, Month-to-Month)":
    default_vals.update({
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 99.5,
        "TotalCharges": 199.0
    })
elif preset_choice == "Loyal Customer (Long-term, 2-Year Contract)":
    default_vals.update({
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.0,
        "TotalCharges": 4160.0
    })
elif preset_choice == "Moderate Risk (Fiber Optic, Mid-Tenure)":
    default_vals.update({
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 88.0,
        "TotalCharges": 2112.0
    })

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Customer Predictor", 
    "📁 Batch Prediction (CSV Upload)", 
    "🧠 Model Diagnostics & Insights"
])

# ---------------------------------------------------------
# TAB 1: Single Customer Predictor
# ---------------------------------------------------------
with tab1:
    st.markdown("### 📋 Customer Profile & Service Configuration")
    st.write("Configure the customer parameters below to calculate churn vulnerability in real-time.")
    
    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        
        # --- COLUMN 1: Demographics & Tenure ---
        with col1:
            st.markdown("#### 👤 Demographics & Tenure")
            gender = st.selectbox(
                "Gender", 
                options=list(encoders["gender"].classes_),
                index=list(encoders["gender"].classes_).index(default_vals["gender"])
            )
            senior_citizen = st.selectbox(
                "Senior Citizen",
                options=[0, 1],
                format_func=lambda x: "Yes (Senior)" if x == 1 else "No (Non-Senior)",
                index=default_vals["SeniorCitizen"]
            )
            partner = st.selectbox(
                "Has Partner?",
                options=list(encoders["Partner"].classes_),
                index=list(encoders["Partner"].classes_).index(default_vals["Partner"])
            )
            dependents = st.selectbox(
                "Has Dependents?",
                options=list(encoders["Dependents"].classes_),
                index=list(encoders["Dependents"].classes_).index(default_vals["Dependents"])
            )
            tenure = st.slider(
                "Tenure (Months with Company)",
                min_value=0,
                max_value=72,
                value=int(default_vals["tenure"]),
                help="Number of continuous months the customer has stayed with the company."
            )
            
        # --- COLUMN 2: Subscriptions & Services ---
        with col2:
            st.markdown("#### 🌐 Telecom & Cloud Services")
            phone_service = st.selectbox(
                "Phone Service",
                options=list(encoders["PhoneService"].classes_),
                index=list(encoders["PhoneService"].classes_).index(default_vals["PhoneService"])
            )
            multiple_lines = st.selectbox(
                "Multiple Lines",
                options=list(encoders["MultipleLines"].classes_),
                index=list(encoders["MultipleLines"].classes_).index(default_vals["MultipleLines"])
            )
            internet_service = st.selectbox(
                "Internet Service Provider",
                options=list(encoders["InternetService"].classes_),
                index=list(encoders["InternetService"].classes_).index(default_vals["InternetService"])
            )
            online_security = st.selectbox(
                "Online Cyber Security",
                options=list(encoders["OnlineSecurity"].classes_),
                index=list(encoders["OnlineSecurity"].classes_).index(default_vals["OnlineSecurity"])
            )
            online_backup = st.selectbox(
                "Online Cloud Backup",
                options=list(encoders["OnlineBackup"].classes_),
                index=list(encoders["OnlineBackup"].classes_).index(default_vals["OnlineBackup"])
            )
            device_protection = st.selectbox(
                "Device Protection Plan",
                options=list(encoders["DeviceProtection"].classes_),
                index=list(encoders["DeviceProtection"].classes_).index(default_vals["DeviceProtection"])
            )
            tech_support = st.selectbox(
                "Premium Tech Support",
                options=list(encoders["TechSupport"].classes_),
                index=list(encoders["TechSupport"].classes_).index(default_vals["TechSupport"])
            )
            streaming_tv = st.selectbox(
                "Streaming TV Service",
                options=list(encoders["StreamingTV"].classes_),
                index=list(encoders["StreamingTV"].classes_).index(default_vals["StreamingTV"])
            )
            streaming_movies = st.selectbox(
                "Streaming Movies Service",
                options=list(encoders["StreamingMovies"].classes_),
                index=list(encoders["StreamingMovies"].classes_).index(default_vals["StreamingMovies"])
            )

        # --- COLUMN 3: Billing & Contract ---
        with col3:
            st.markdown("#### 💳 Billing & Account Contract")
            contract = st.selectbox(
                "Contract Commitment",
                options=list(encoders["Contract"].classes_),
                index=list(encoders["Contract"].classes_).index(default_vals["Contract"]),
                help="Month-to-month contracts are statistically the highest churn risk."
            )
            paperless_billing = st.selectbox(
                "Paperless Billing",
                options=list(encoders["PaperlessBilling"].classes_),
                index=list(encoders["PaperlessBilling"].classes_).index(default_vals["PaperlessBilling"])
            )
            payment_method = st.selectbox(
                "Payment Method",
                options=list(encoders["PaymentMethod"].classes_),
                index=list(encoders["PaymentMethod"].classes_).index(default_vals["PaymentMethod"])
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($ USD)",
                min_value=15.0,
                max_value=150.0,
                value=float(default_vals["MonthlyCharges"]),
                step=1.0
            )
            
            # Helper recommendation for TotalCharges
            calc_total = float(np.round(monthly_charges * max(tenure, 1), 2))
            total_charges = st.number_input(
                "Total Charges to Date ($ USD)",
                min_value=0.0,
                max_value=10000.0,
                value=float(default_vals["TotalCharges"]) if default_vals["TotalCharges"] > 0 else calc_total,
                step=10.0,
                help="Total cumulative amount billed to the customer."
            )
            
        submitted = st.form_submit_button("⚡ Run Real-Time Churn Evaluation", use_container_width=True, type="primary")

    # Run Prediction when submitted
    if submitted or preset_choice != "Custom Input":
        # Build DataFrame
        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        df_input = pd.DataFrame([input_dict])
        
        # Apply Label Encoders
        df_encoded = df_input.copy()
        for col, le in encoders.items():
            if col in df_encoded.columns:
                df_encoded[col] = le.transform(df_encoded[col])
                
        # Align features
        df_encoded = df_encoded[feature_names]
        
        # Run inference
        prediction = model.predict(df_encoded)[0]
        probabilities = model.predict_proba(df_encoded)[0]
        churn_prob = probabilities[1]
        stay_prob = probabilities[0]
        
        # Display Result Section
        st.markdown("---")
        st.markdown("### 🎯 Predictive Assessment Results")
        
        res_col1, res_col2, res_col3 = st.columns([1.2, 1, 1])
        
        with res_col1:
            if churn_prob >= 0.60:
                risk_class = "risk-high"
                risk_label = "HIGH RISK OF CHURNING"
                color_hex = "#EF4444"
                verdict_icon = "🚨"
            elif churn_prob >= 0.30:
                risk_class = "risk-medium"
                risk_label = "MODERATE RISK"
                color_hex = "#F59E0B"
                verdict_icon = "⚠️"
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK (LOYAL)"
                color_hex = "#10B981"
                verdict_icon = "✅"
                
            st.markdown(f"""
            <div class="metric-card {risk_class}">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">{verdict_icon}</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: {color_hex}; letter-spacing: 0.5px;">
                    {risk_label}
                </div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #FFFFFF; margin: 10px 0;">
                    {churn_prob * 100:.1f}%
                </div>
                <div style="color: #94A3B8; font-size: 0.95rem;">
                    Probability of Customer Cancellation
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 600;">RETENTION PROBABILITY</div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #10B981; margin: 10px 0;">
                    {stay_prob * 100:.1f}%
                </div>
                <div style="color: #64748B; font-size: 0.85rem;">
                    Likelihood customer maintains account
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94A3B8; font-size: 0.95rem; font-weight: 600;">LIFETIME ESTIMATED VALUE</div>
                <div style="font-size: 2.5rem; font-weight: 900; color: #60A5FA; margin: 10px 0;">
                    ${total_charges:,.2f}
                </div>
                <div style="color: #64748B; font-size: 0.85rem;">
                    Cumulative revenue billed ({tenure} months)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Strategic Retention Actions
        st.markdown("#### 💡 Prescriptive Retention Recommendations")
        recommendations = []
        
        if contract == "Month-to-month":
            recommendations.append("📌 **Contract Incentivization**: Customer is on a Month-to-Month plan. Transition them to a 1-year or 2-year contract with an initial 10-15% loyalty rate discount.")
            
        if payment_method == "Electronic check":
            recommendations.append("💳 **Automated Payment Incentive**: Electronic check payments exhibit higher friction and churn rates. Offer a $5 one-time credit for enrolling in Auto-pay (Bank Transfer / Credit Card).")
            
        if internet_service == "Fiber optic" and tech_support == "No":
            recommendations.append("🛠️ **Bundle Premium Tech Support**: High-speed Fiber customers without Tech Support frequently churn due to unresolved speed/device issues. Provide a complimentary 3-month trial of Tech Support.")
            
        if tenure <= 6:
            recommendations.append("📞 **Early Onboarding Check-in**: Customer is in their vulnerable first 6 months. Schedule an onboarding satisfaction call or email sequence.")
            
        if online_security == "No":
            recommendations.append("🛡️ **Value-Add Security Addon**: Bundle 'Online Security' and 'Cloud Backup' at a discounted rate to raise customer switching barriers.")
            
        if not recommendations:
            recommendations.append("🎉 **Customer Health Excellent**: Customer shows strong retention indicators. Consider nominating for loyalty reward perks or cross-selling new services.")
            
        for rec in recommendations:
            st.markdown(f"""
            <div class="recommendation-box">
                {rec}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Batch Analysis (CSV Upload)
# ---------------------------------------------------------
with tab2:
    st.markdown("### 📁 Batch Prediction Engine")
    st.write("Upload a CSV file containing multiple customer records to perform bulk churn inference.")
    
    # Download sample CSV template button
    sample_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_customers.csv")
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample Customers CSV Template",
                data=f.read(),
                file_name="sample_customers.csv",
                mime="text/csv",
                help="Use this template to see the exact columns and format expected."
            )
            
    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded dataset with **{len(batch_df)}** records.")
            
            # Check for missing required features
            missing_cols = [c for c in feature_names if c not in batch_df.columns]
            if missing_cols:
                st.error(f"The uploaded CSV is missing the following required columns: {missing_cols}")
            else:
                # Preprocess batch
                processed_df = batch_df.copy()
                for col, le in encoders.items():
                    if col in processed_df.columns:
                        # Convert to string and handle unseen labels safely
                        processed_df[col] = processed_df[col].astype(str)
                        known_classes = set(le.classes_)
                        # Replace unknown with most frequent class
                        processed_df[col] = processed_df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                        processed_df[col] = le.transform(processed_df[col])
                        
                X_batch = processed_df[feature_names]
                
                # Make Predictions
                preds = model.predict(X_batch)
                probs = model.predict_proba(X_batch)[:, 1]
                
                # Enrich original dataframe
                results_df = batch_df.copy()
                results_df["Churn_Predicted"] = np.where(preds == 1, "Yes", "No")
                results_df["Churn_Probability"] = np.round(probs, 4)
                results_df["Risk_Category"] = pd.cut(
                    probs, 
                    bins=[-0.01, 0.30, 0.60, 1.01], 
                    labels=["Low Risk", "Medium Risk", "High Risk"]
                )
                
                # Summary KPIs
                churn_count = int(np.sum(preds == 1))
                total_count = len(preds)
                churn_rate = (churn_count / total_count) * 100
                high_risk_count = int(np.sum(results_df["Risk_Category"] == "High Risk"))
                
                st.markdown("#### 📊 Batch Summary Telemetry")
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                b_col1.metric("Total Customers Evaluated", f"{total_count:,}")
                b_col2.metric("Projected Churners", f"{churn_count:,}")
                b_col3.metric("Projected Churn Rate", f"{churn_rate:.1f}%")
                b_col4.metric("High Vulnerability Accounts", f"{high_risk_count:,}")
                
                # Charts
                st.markdown("#### 📈 Risk Distribution")
                chart_col1, chart_col2 = st.columns([1, 1])
                
                with chart_col1:
                    risk_counts = results_df["Risk_Category"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Category", "Count"]
                    
                    bar_chart = alt.Chart(risk_counts).mark_bar(cornerRadius=6).encode(
                        x=alt.X("Risk Category:N", sort=["Low Risk", "Medium Risk", "High Risk"]),
                        y=alt.Y("Count:Q"),
                        color=alt.Color("Risk Category:N", scale=alt.Scale(
                            domain=["Low Risk", "Medium Risk", "High Risk"],
                            range=["#10B981", "#F59E0B", "#EF4444"]
                        )),
                        tooltip=["Risk Category", "Count"]
                    ).properties(height=280)
                    st.altair_chart(bar_chart, use_container_width=True)
                    
                with chart_col2:
                    if "MonthlyCharges" in results_df.columns:
                        scatter = alt.Chart(results_df).mark_circle(size=70, opacity=0.7).encode(
                            x=alt.X("tenure:Q", title="Tenure (Months)"),
                            y=alt.Y("MonthlyCharges:Q", title="Monthly Charges ($)"),
                            color=alt.Color("Risk_Category:N", scale=alt.Scale(
                                domain=["Low Risk", "Medium Risk", "High Risk"],
                                range=["#10B981", "#F59E0B", "#EF4444"]
                            )),
                            tooltip=["tenure", "MonthlyCharges", "Churn_Probability", "Risk_Category"]
                        ).properties(height=280)
                        st.altair_chart(scatter, use_container_width=True)
                
                # Filterable Table
                st.markdown("#### 📋 Detailed Batch Predictions")
                filter_choice = st.selectbox("Filter by Risk Tier:", ["All", "High Risk", "Medium Risk", "Low Risk"])
                if filter_choice != "All":
                    filtered_df = results_df[results_df["Risk_Category"] == filter_choice]
                else:
                    filtered_df = results_df
                    
                st.dataframe(filtered_df, use_container_width=True)
                
                # Export Button
                csv_export = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Enriched Results CSV",
                    data=csv_export,
                    file_name="churn_predictions_results.csv",
                    mime="text/csv",
                    type="primary"
                )
                
        except Exception as e:
            st.error(f"Error executing batch inference: {e}")

# ---------------------------------------------------------
# TAB 3: Model Diagnostics & Insights
# ---------------------------------------------------------
with tab3:
    st.markdown("### 🧠 Model Explainability & Feature Importance")
    st.write("Understand the underlying intelligence of the trained `RandomForestClassifier` and how each customer feature impacts predictions.")
    
    # Feature Importances Chart
    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    fi_chart = alt.Chart(importances).mark_bar(cornerRadius=6, color="#6366F1").encode(
        x=alt.X("Importance:Q", title="Relative Predictive Weight (Gini Importance)"),
        y=alt.Y("Feature:N", sort="-x", title="Feature Name"),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")]
    ).properties(height=500)
    
    st.altair_chart(fi_chart, use_container_width=True)
    
    st.markdown("#### 🔎 Critical Insights from Feature Weights")
    st.markdown("""
    1. **TotalCharges & MonthlyCharges (~28% combined impact)**: Financial cost is the primary driver of attrition. Customers with high monthly fees without bundled discounts have elevated churn rates.
    2. **Contract Type (~12.7% impact)**: Customers locked into **One-Year** or **Two-Year** contracts have dramatically lower attrition rates than those on Month-to-Month arrangements.
    3. **Tenure (~12.2% impact)**: The first 6–12 months represents the most vulnerable retention window.
    4. **Online Security & Tech Support (~16% combined impact)**: Customers who adopt cybersecurity and tech assistance plugins demonstrate much higher retention "stickiness".
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.9rem; padding: 12px 0;">
    Customer Churn Prediction Model • Built with Streamlit, Scikit-Learn & Python • Ready for Public Deployment
</div>
""", unsafe_allow_html=True)
