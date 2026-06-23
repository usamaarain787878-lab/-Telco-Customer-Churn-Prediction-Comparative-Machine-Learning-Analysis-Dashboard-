# 📊 Telco Customer Churn Prediction & Comparative ML Analysis Dashboard

### 🌐 Live Application URL: [https://share.streamlit.io/your-username/telco-churn-project2](https://share.streamlit.io/your-username/telco-churn-project2)

## 🎯 Business Problem & Context 
Acquiring new clients carries a significantly higher financial premium than retaining an active customer base. When customers depart, businesses suffer immediate operational losses and recurring monthly contract revenue slips. By proactively mapping high-risk customer profiles early, marketing teams and customer success managers can introduce precision retention interventions to stabilize revenue retention.

This comprehensive end-to-end Machine Learning project framework is designed to construct a predictive solution capable of forecasting subscriber churn risks within a major telecommunications enterprise.

## 📂 Dataset Details 
The model training is executed on the industry-standard **Telco Customer Churn Dataset (Kaggle)**.
* **Dimensions:** ~7,043 subscriber records with 21 data dimensions.
* **Feature Profiles:** The dataset contains numerical features (Tenure, MonthlyCharges) and multiple categorical facets (Contract, InternetService, PaymentMethod).
* **Target Variable:** `Churn` (Yes/No binary classification).

## 💻 Technologies Summary 
* **Environment:** Python 3.9+
* **Libraries:** Scikit-Learn (for ML pipeline), Pandas/Numpy (for data structure management), XGBoost (advanced boosting capability), Seaborn/Plotly (for dashboard visual graphics), Joblib (for binary model serialization).
* **Framework:** **Streamlit** (interactive dashboard deployment).
* **Deployment:** Hosted via **Streamlit Community Cloud / Render / Hugging Face Spaces**.

## 🚀 Key EDA Discoveries & Insights 
* **Contract Analysis:** High churn velocities are statistically visible in subscribers executing **Month-to-month contracts**.
* **Billing Dynamics:** Customers with higher `MonthlyCharges` bounds show elevated churn variances.
* **Target Balance:** The dataset displays significant class imbalance (approx. 27% Churn rate), managed through stratified splitting and evaluation metrics precision.

## 🏆 Model Evaluation Leaderboard 
Based on Phase 6 comparison analysis, the optimal solution was identified using the following performance metrics baseline:

| Model Group | Analytical Purpose / Characteristics | Accuracy Score | Precision Score | Recall Score | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | Applies sequential boosting trees for high-accuracy tabular predictions. | **~80.5%** | **~67.2%** | **~53.8%** | **~59.7%** |
| Logistic Regression | Establishes a solid, interpretable linear classification baseline. | ~80.1% | ~66.1% | ~53.1% | ~58.9% |
| Random Forest | Leverages bagging ensemble methods to mitigate variance. | ~79.8% | ~66.0% | ~51.2% | ~57.7% |

*The Tuned XGBoost engine is chosen as the **Champion Model** for final production deployment due to its superior trade-off balancing Precision and Recall.*

## 🛠️ Local Installation & Execution 

Follow these precise steps to deploy the application on your local terminal environment:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-username/telco-churn-project2.git](https://github.com/your-username/telco-churn-project2.git)
   cd telco-churn-project2