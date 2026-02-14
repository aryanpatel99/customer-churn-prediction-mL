import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BINARY_MAP = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Phone Service": {"Yes": 1, "No": 0},
    "Paperless Billing": {"Yes": 1, "No": 0},
    "Internet Service": {"Yes": 1, "No": 0},
}

DUMMY_COLS = [
    "Offer",
    "Multiple Lines",
    "Internet Type",
    "Online Security",
    "Online Backup",
    "Device Protection Plan",
    "Premium Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Streaming Music",
    "Unlimited Data",
    "Payment Method",
]


def encode(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    df = df.copy()

    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    df = pd.get_dummies(df, columns=DUMMY_COLS, drop_first=True)

    df = df.drop(columns=["Contract"], errors="ignore")

    if training:
        joblib.dump(df.columns, "artifacts/feature_columns.pkl")
    else:
        saved_cols = joblib.load("artifacts/feature_columns.pkl")
        df = df.reindex(columns=saved_cols, fill_value=0)

    return df




def split_data(df, test_size=0.2, random_state=42):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)



def scale_train(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    joblib.dump(scaler, "artifacts/scaler.pkl")

    return X_train_scaled


def scale_test(X_test):
    scaler = joblib.load("artifacts/scaler.pkl")
    return scaler.transform(X_test)
