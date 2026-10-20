# ⚡ Customer Churn Prediction & Retention Platform

An end-to-end Machine Learning web application built with **Streamlit** and **Scikit-Learn** (`RandomForestClassifier`) to predict customer churn, identify vulnerability risk factors, and provide prescriptive retention recommendations.

---

## 🌟 Features

- **⚡ Real-Time Single Customer Evaluation**: Interactive input form across Demographics, Telecom Services, and Billing/Contract details.
- **🎯 1-Click Demo Presets**: Test archetypes immediately (High Risk, Loyal Long-Term, Moderate Risk).
- **📁 Bulk Batch Prediction Engine**: Upload CSV customer files, run batch inference, view distribution analytics, and download enriched result datasets with churn probabilities.
- **💡 Prescriptive Retention Recommendations**: Actionable strategies automatically generated based on high-impact risk factors (e.g., month-to-month contracts, missing tech support, electronic check payments).
- **🧠 Model Diagnostics & Feature Weights**: Transparent explainability showing Gini feature importance from the 100-tree Random Forest ensemble.

---

## 🚀 How to Run Locally

### 1. Clone or Open the Repository
```bash
cd customer_churn_model
```

### 2. Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud (Public)

Deploying this app publicly for free on [Streamlit Community Cloud](https://streamlit.io/cloud) takes less than 2 minutes:

### Step 1: Push this Project to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g. `customer-churn-model`). You can make it **Public**.
2. Link your local project and push to GitHub:
   ```bash
   git add .
   git commit -m "feat: complete customer churn prediction streamlit app"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
2. Click **"New app"**.
3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will automatically detect `requirements.txt`, install dependencies, load the cached model, and give you a public URL (e.g., `https://<your-app-name>.streamlit.app`) to share with anyone!

---

## 📊 Model Information

- **Algorithm**: `RandomForestClassifier` (100 Decision Trees)
- **Framework**: `scikit-learn==1.6.1`
- **Trained Features**: 19 Features
  - *Demographics*: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`
  - *Services*: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  - *Contract & Billing*: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Artifacts**:
  - `customer_churn_model.pkl`: Model weights and feature mappings.
  - `encoders.pkl`: Fitted LabelEncoders for all categorical features.
