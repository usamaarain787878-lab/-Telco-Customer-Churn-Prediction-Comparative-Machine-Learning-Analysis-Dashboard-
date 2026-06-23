import pandas as pd
import numpy as np
import os
import joblib
import re

# Machine Learning Models
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC  # Sabse tez SVM model
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_and_serialize_models():
    print("🔄 Step 1: Loading Dataset...")
    
    possible_paths = [
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/Telco-Customer-Churn.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            data_path = path
            break
            
    if not data_path:
        print("❌ Error: Valid Churn CSV file empty hai ya data/ folder ma nahi mili!")
        print("💡 Please check karein ke aapki CSV file data/ folder ma mojood ho aur khali na ho.")
        return
        
    print(f"📖 Reading data from: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Error while reading CSV: {e}")
        return

    print("🔄 Step 2: Cleaning Data & Handling Missing Values...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    X = pd.get_dummies(X, drop_first=True)

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns]
    X = X.astype(float)

    # Step 3: Train-Test Partition
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 4: Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('models', exist_ok=True)
    os.makedirs('visuals', exist_ok=True)

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')

    # 🔥 Step 5: Super-Fast Models Settings
    base_svm = LinearSVC(dual=False, max_iter=1000, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "SVM": CalibratedClassifierCV(base_svm, cv=2), # Super fast execution
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }

    # Step 6: Hyperparameter Tuning (Fewer CV loops to make it faster)
    print("🔄 Step 3: Hyperparameter Tuning (Random Forest & XGBoost)...")
    
    rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    rf_grid = GridSearchCV(models["Random Forest"], rf_param_grid, cv=2, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf_grid.best_estimator_

    xgb_param_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    xgb_grid = GridSearchCV(models["XGBoost"], xgb_param_grid, cv=2, scoring='f1', n_jobs=-1)
    xgb_grid.fit(X_train_scaled, y_train)
    models["XGBoost"] = xgb_grid.best_estimator_

    # Step 7: Train, Evaluate, and Serialize All Models with Confusion Matrices
    print("🔄 Step 4: Training & Saving All 6 Models...")
    metrics_results = []

    for name, model in models.items():
        if name not in ["Random Forest", "XGBoost"]:
            model.fit(X_train_scaled, y_train)
            
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        metrics_results.append({
            "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "ROC-AUC": auc
        })

        # Confusion Matrix Save
        cm = confusion_matrix(y_test, y_pred)
        cm_filename = f"models/{name.lower().replace(' ', '_')}_confusion_matrix.pkl"
        joblib.dump(cm, cm_filename)

        # Model Save
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)
        print(f" ✅ Saved: {model_filename} aur uski Confusion Matrix")

    # 📌 DIRECT CSV EXPORT INSIDE THE MODELS DIRECTORY
    df_metrics = pd.DataFrame(metrics_results)
    output_csv_path = os.path.join("models", "model_comparison_metrics.csv")
    df_metrics.to_csv(output_csv_path, index=False)
    
    print(f"✅ CSV Saved Successfully at: {output_csv_path}")
    print("🚀 Success: All 6 models and Confusion Matrices saved successfully!")

if __name__ == "__main__":
    train_and_serialize_models()