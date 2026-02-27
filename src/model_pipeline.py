import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

def preprocess_data(df, is_training=True, expected_columns=None):
    df = df.copy()
    
    if is_training and "Customer Status" in df.columns:
        df["Churn"] = df["Customer Status"].apply(lambda x: 1 if x == "Churned" else 0)
        
    cols_to_drop = ["Customer ID", "Customer Status", "Churn Category", "Churn Reason", "City", "Zip Code", "Latitude", "Longitude"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    
    df["Avg Monthly Long Distance Charges"] = df["Avg Monthly Long Distance Charges"].fillna(0)
    df["Avg Monthly GB Download"] = df["Avg Monthly GB Download"].fillna(0)
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Multiple Lines"] = df["Multiple Lines"].fillna("No Phone")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet Service")
    
    internet_cols = ["Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data"]
    for col in internet_cols:
        if col in df.columns:
            df[col] = df[col].fillna("No Internet Service")
            
    if "Total Revenue" in df.columns and "Tenure in Months" in df.columns:
        df["Revenue_Per_Month"] = df["Total Revenue"] / df["Tenure in Months"].replace(0, 1)
    if "Contract" in df.columns:
        df["Is_Monthly_Contract"] = (df["Contract"] == "Month-to-Month").astype(int)
        df = df.drop(columns=["Contract"])
        
    binary_map = {"Gender": {"Male": 1, "Female": 0}, "Married": {"Yes": 1, "No": 0}, "Phone Service": {"Yes": 1, "No": 0}, "Paperless Billing": {"Yes": 1, "No": 0}, "Internet Service": {"Yes": 1, "No": 0}}
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    dummy_cols = ["Offer", "Multiple Lines", "Internet Type", "Online Security", "Online Backup", "Device Protection Plan", "Premium Tech Support", "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data", "Payment Method"]
    df = pd.get_dummies(df, columns=[c for c in dummy_cols if c in df.columns], drop_first=True)
    
    if not is_training and expected_columns is not None:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]
        
    return df

def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    print(f"--- {model_name} Evaluation (Threshold: {threshold}) ---")
    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    # plt.title(f"Confusion Matrix: {model_name}")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.show()
    
    return auc

def train_and_evaluate(data_path):
    df = pd.read_csv(data_path)
    df_processed = preprocess_data(df, is_training=True)
    
    X = df_processed.drop(columns=["Churn"])
    y = df_processed["Churn"]
    
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(X.columns.tolist(), os.path.join(artifacts_dir, "feature_columns.pkl"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    dt_model = DecisionTreeClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42, class_weight="balanced")
    dt_model.fit(X_train_scaled, y_train)
    
    lr_auc = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression", threshold=0.3)
    dt_auc = evaluate_model(dt_model, X_test_scaled, y_test, "Decision Tree")
    
    best_model, best_name = (lr_model, "Logistic Regression") if lr_auc >= dt_auc else (dt_model, "Decision Tree")
    
    print(f"Best model: {best_name}. Saving...")
    joblib.dump(best_model, os.path.join(artifacts_dir, "best_churn_model.pkl"))
    print("Model training and evaluation completed. Artifacts saved.")


def predict_churn(new_data_df):
    artifacts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../artifacts")
    try:
        model = joblib.load(os.path.join(artifacts_dir, "best_churn_model.pkl"))
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler.pkl"))
        expected_columns = joblib.load(os.path.join(artifacts_dir, "feature_columns.pkl"))
    except FileNotFoundError as e:
        raise FileNotFoundError("Artifacts not found. Run train_and_evaluate() first.") from e

    processed_df = preprocess_data(new_data_df, is_training=False, expected_columns=expected_columns)
    X_scaled = scaler.transform(processed_df)
    
    probabilities = model.predict_proba(X_scaled)[:, 1]
    threshold = 0.3 
    predictions = (probabilities >= threshold).astype(int)
    
    final_df = new_data_df.copy()
    final_df["Churned"] = ["Yes" if pred == 1 else "No" for pred in predictions]
    final_df["Probability"] = [f"{prob*100:.2f}%" for prob in probabilities]
    final_df.reset_index(drop=True, inplace=True)
    return final_df


if __name__ == "__main__":

    DATA_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/raw/telecom_customer_churn.csv")
    if os.path.exists(DATA_PATH):
        # train_and_evaluate(DATA_PATH)
        
        print("\n--- Testing predict_churn with sample data ---")

        df_raw = pd.read_csv(DATA_PATH)
        sample_df = df_raw.sample(5) 
        
        result_df = predict_churn(sample_df)
        print(result_df[["Customer ID", "Churned", "Probability"]])
        
        # for i, row in result_df.reset_index().iterrows():
        #     print(f"Sample {i+1}: Prediction = {row['Churned']}, Churn Probability = {row['Probability']:.4f}")
            
    else:
        print(f"Data file not found at {DATA_PATH}. Please check the path.")





